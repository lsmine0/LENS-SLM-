import torch
import pandas as pd
from tqdm import tqdm
from EnsembledChat import EnsembleChatBot
from Datasets import prepare_lens_data

def run_benchmark(model_paths, dataset_names, num_samples=100):
    device = "cuda"
    # 1. Initialize
    bot = EnsembleChatBot(model_paths, cmlp_dir="./cmlp_weights", device=device)
    
    results = []

    for d_name in dataset_names:
        print(f"\n--- Benchmarking {d_name} ---")
        test_data = prepare_lens_data(bot.tokenizer, d_name, split=f"train[:{num_samples}]")
        
        # Track statistics
        stats = {m.model_path: {"correct": 0} for m in bot.wrappers}
        stats["Ensemble"] = {"correct": 0}

        for item in tqdm(test_data):
            input_ids = item['input_ids'].to(device)
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
                
            target_id = item['labels'].item()
            
            model_probs = []
            model_confs = []

            # 2. Iterate through wrappers carefully
            for slm in bot.wrappers:
                # Cache clearing is vital when mixing 1B and 3B RoPE dimensions
                torch.cuda.empty_cache() 
                
                with torch.no_grad():
                    outputs = slm.slm(input_ids, output_hidden_states=True, return_dict=True)
                    logits = outputs.logits[:, -1, :]
                    
                    # --- Use the Wrapper's Binary Logic ---
                    # max_p is filtered prob; pred_id is either TRUE_ID or FALSE_ID
                    max_p, pred_id = slm.get_binary_prediction(logits)
                    
                    # Individual Accuracy
                    if pred_id.item() == target_id:
                        stats[slm.model_path]["correct"] += 1
                    
                    # CMLP Feature Extraction
                    hiddens = torch.cat([lyr[:, -1, :] for lyr in outputs.hidden_states[-slm.num_layers:]], dim=-1)
                    
                    # Pass the filtered max_p to the CMLP
                    feat = torch.cat([hiddens, max_p.unsqueeze(-1)], dim=-1).to(torch.bfloat16)
                    conf = slm.cmlp(feat)
                    
                    # For ensemble, we only care about the True/False distribution
                    # Index 0 = True, Index 1 = False
                    binary_dist = torch.softmax(logits[:, [slm.true_id, slm.false_id]], dim=-1)
                    
                    model_probs.append(binary_dist)
                    model_confs.append(conf)

            # --- Ensemble Integration ---
            stacked_probs = torch.cat(model_probs, dim=0) # [num_models, 2]
            stacked_conf = torch.cat(model_confs, dim=0)   # [num_models, 1]
            
            weights = torch.softmax(stacked_conf, dim=0)
            # Final probs over [True, False]
            final_probs = torch.sum(stacked_probs * weights, dim=0)
            
            # If Index 0 is higher, ensemble picked TRUE_ID
            ens_local_idx = torch.argmax(final_probs).item()
            # Map back to the tokenizer's ID (using the first wrapper's IDs is safe as they match)
            ens_pred_id = bot.wrappers[0].true_id if ens_local_idx == 0 else bot.wrappers[0].false_id

            if ens_pred_id == target_id:
                stats["Ensemble"]["correct"] += 1

        # Aggregate Results
        total_items = len(test_data)
        for path in stats:
            results.append({
                "Dataset": d_name.split("/")[-1], # Clean name
                "Model": path.split("/")[-1] if "/" in path else path,
                "Accuracy": (stats[path]["correct"] / total_items) * 100
            })

    return pd.DataFrame(results)

if __name__ == "__main__":
    MODELS = [
        "unsloth/Llama-3.2-1B-Instruct",
        "ai-nexuz/llama-3.2-1b-instruct-fine-tuned",
        "NousResearch/Hermes-3-Llama-3.2-3B",
        "keeeeenw/Llama-3.2-1B-Instruct-Open-R1-Distill",
        "EpistemeAI/Reasoning-Llama-3.2-1B-Instruct-v1.2"
    ]
    DATASETS = ["google/boolq", "smoorsmith/prontoqa", "skrishna/coin_flip", "tasksource/proofwriter"]

    report_df = run_benchmark(MODELS, DATASETS, num_samples=50)
    
    # Create a nice pivot table for comparison
    comparison = report_df.pivot(index="Model", columns="Dataset", values="Accuracy")
    print("\n--- BENCHMARK COMPARISON (Accuracy %) ---")
    print(comparison)