from torch.utils.data import DataLoader
import torch
import os
from pathlib import Path
from SLMWrapper import SLMWrapper
from transformers import AutoTokenizer
from Datasets import prepare_lens_data, Calibration_Dataset
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
        input_ids = [item['input_ids'] for item in batch]
        labels = [item['labels'] for item in batch]
        
        input_ids_padded = pad_sequence(
            input_ids, 
            batch_first=True, 
            padding_value=tokenizer.pad_token_id
        )
        
        return {
            'input_ids': input_ids_padded,
            'labels': torch.stack(labels)
        }

def train_slms(model_paths, device="cuda", CMLPPath=None):
    tokenizer = AutoTokenizer.from_pretrained(model_paths[0])

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. Load
    tasks = [
        ("google/boolq", "train[:2000]"),
        ("smoorsmith/prontoqa", "train[:500]"), 
        ("skrishna/coin_flip", "train[:2000]"),
        ("tasksource/proofwriter", "train[:2000]") 
    ]
    
    full_data_list = []
    for name, split in tasks:
        print(f"Loading {name}...")
        dataset = prepare_lens_data(tokenizer, name, split)
        full_data_list.extend(dataset.data)

    master_dataloader = DataLoader(Calibration_Dataset(full_data_list), batch_size=50, shuffle=True, collate_fn=collate_fn)

    save_dir = Path(CMLPPath).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    # 2. Train
    for path in model_paths:
        slm = SLMWrapper(model_path=path, device=device)
        
        slm.train_cmlp(master_dataloader, epochs=5, lr=5e-5)

        model_name_clean = path.replace("/", "_").replace("\\", "_")
        save_file = os.path.join(CMLPPath, f"{model_name_clean}_cmlp.pt")

        slm.saveCMLP(save_file)
        del slm
        torch.cuda.empty_cache()


models = [
    "unsloth/Llama-3.2-1B-Instruct",       
    "ai-nexuz/llama-3.2-1b-instruct-fine-tuned",
    "NousResearch/Hermes-3-Llama-3.2-3B",    
    "keeeeenw/Llama-3.2-1B-Instruct-Open-R1-Distill",
    "EpistemeAI/Reasoning-Llama-3.2-1B-Instruct-v1.2" 
]

tokenizer = AutoTokenizer.from_pretrained(models[0])

train_slms(
            model_paths=models, 
            device="cuda", 
            CMLPPath="./cmlp_weights"
        )