import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

class Calibration_Dataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'input_ids': self.data[idx]['input_ids'].squeeze(0),
            'labels': self.data[idx]['labels']
        }

def prepare_lens_data(tokenizer, dataset_name="boolq", split="train[:1000]"):
    ds = load_dataset(dataset_name, split=split)
    formatted_data = []
     
    for item in ds:
        prompt, target = "", None
           
        if dataset_name == "google/boolq":
            prompt = f"Question: {item['question']} \nTrue or false? Answer:"
            target = "True" if item['answer'] is True else "False" # 'True' or 'False'
            
        elif dataset_name == "smoorsmith/prontoqa":
            prompt = f"Context: {item['context']}\nQuestion: {item['question']}\nTrue or false? Answer:"
            target = "True" if item['answer'] is True else "False" # 'True' or 'False'
            
        elif dataset_name == "skrishna/coin_flip":
            prompt = f"{item['inputs']}\nTrue or false? Answer:"
            target =  "True" if item['targets'] == "yes" else "False" # 'True' or 'False'
            
        elif dataset_name == "tasksource/proofwriter":
            prompt = f"Theory: {item['theory']}\nQuestion: {item['question']}\nTrue or false? Answer:"
            target = "True" if item['answer'] is True else "False" # 'True' or 'False'

        # Tokenization Logic
        if prompt:
            inputs = tokenizer(prompt, return_tensors="pt")
            target_tokens = tokenizer(target, add_special_tokens=False).input_ids
            if len(target_tokens) > 0:
                formatted_data.append({
                    'input_ids': inputs.input_ids,
                    'labels': torch.tensor(target_tokens[0]) # Target the first token of the answer
                })
        
       
    return Calibration_Dataset(formatted_data)