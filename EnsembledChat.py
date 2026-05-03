import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from SLMWrapper import SLMWrapper
from pathlib import Path

class EnsembleChatBot:
    def __init__(self, model_paths, cmlp_dir, device="cuda"):
        self.device = device
        self.wrappers = []
        self.cmlp_dir = Path(cmlp_dir)
        
        print(f"--- Initializing Ensemble on {device} ---")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_paths[0])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        for path in model_paths:
            print(f"Loading Ensemble Member: {path}...")
            slm = SLMWrapper(model_path=path, device=self.device)
            
            clean_name = path.replace("/", "_").replace("\\", "_")
            weight_path = self.cmlp_dir / f"{clean_name}_cmlp.pt"

            if weight_path.exists():
                slm.cmlp.load_state_dict(torch.load(weight_path, map_location=self.device))
                slm.cmlp.eval()
                print(f"   Successfully loaded confidence weights.")
            else:
                print(f"   WARNING: No weights found at {weight_path}. Using uncalibrated CMLP.")
            
            self.wrappers.append(slm)

    def chat(self, user_input, max_new_tokens=100, temperature=0.7):
        """
        Generates a response using token-level ensemble weighting.
        """
        # Prepare context (simple format for now)
        input_text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        
        generated = input_ids

        for _ in range(max_new_tokens):
            all_probs = []
            all_confidences = []

            # Step 1: Collect Logits and Confidence from every model
            for slm in self.wrappers:
                with torch.no_grad():
                    outputs = slm.slm(generated, output_hidden_states=True, return_dict=True)
                    logits = outputs.logits[:, -1, :] / temperature
                    
                    hiddens = torch.cat(
                        [lyr[:, -1, :] for lyr in outputs.hidden_states[-slm.num_layers:]], 
                        dim=-1
                    )

                    probs = torch.softmax(logits, dim=-1)
                    max_prob = probs.max(dim=-1, keepdim=True)[0]

                    feature_vector = torch.cat([hiddens, max_prob], dim=-1).to(torch.bfloat16)

                    conf = slm.cmlp(feature_vector) # Shape: [1, 1]
                    
                    all_probs.append(probs)
                    all_confidences.append(conf)

            stacked_probs = torch.cat(all_probs, dim=0)
            stacked_conf = torch.stack(all_confidences).unsqueeze(-1) 
            
            weights = F.softmax(stacked_conf, dim=0)
            
            final_probs = torch.sum(stacked_probs * weights, dim=0, keepdim=True)

            next_token = torch.multinomial(final_probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=-1)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        response = self.tokenizer.decode(generated[0][input_ids.shape[-1]:], skip_special_tokens=True)
        return response

# --- Main Interaction Loop ---

if __name__ == "__main__":
    models = [
    "unsloth/Llama-3.2-1B-Instruct",       
    "ai-nexuz/llama-3.2-1b-instruct-fine-tuned",
    "NousResearch/Hermes-3-Llama-3.2-3B",    
    "keeeeenw/Llama-3.2-1B-Instruct-Open-R1-Distill",
    "EpistemeAI/Reasoning-Llama-3.2-1B-Instruct-v1.2" 
    ]
    
    # Initialize the Ensemble
    bot = EnsembleChatBot(
        model_paths=models, 
        cmlp_dir="./cmlp_weights",
        device="cuda"
    )

    print("\n--- Ensemble Bot Ready! (Type 'exit' to quit) ---")
    while True:
        user_msg = input("You: ")
        if user_msg.lower() in ["exit", "quit"]:
            break
            
        reply = bot.chat(user_msg)
        print(f"\nEnsemble: {reply}\n")