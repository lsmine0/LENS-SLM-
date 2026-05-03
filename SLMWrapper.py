import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class SLMWrapper(nn.Module):
    def __init__(self, model_path, device="cuda"):
        super().__init__()
        self.device = device
        self.model_path = model_path
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.true_id = tokenizer.encode("True", add_special_tokens=False)[-1]
        self.false_id = tokenizer.encode("False", add_special_tokens=False)[-1]

        self.slm = AutoModelForCausalLM.from_pretrained(
            model_path, 
            dtype=torch.bfloat16, 
            device_map=device
        )
        for p in self.slm.parameters(): 
            p.requires_grad = False
            
    
        self.hidden_dim = self.slm.config.hidden_size
        self.num_layers = 4 
        self.cmlp = nn.Sequential(
            nn.Linear(self.hidden_dim * self.num_layers + 1, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        ).to(device).to(torch.bfloat16)

        self._last_conf = 0.0

    @property
    def conf(self):
        return self._last_conf

    def saveCMLP(self, path=None):
        torch.save(self.cmlp.state_dict(), path)

    def loadCMLP(self, path=None):
        if os.path.exists(path):
            self.cmlp.load_state_dict(torch.load(path, map_location=self.device))
            self.cmlp.eval() 
        else:
            raise FileNotFoundError(f"No CMLP weights found at {path}")
    def get_binary_prediction(self, logits):

        binary_logits = logits[:, [self.true_id, self.false_id]]
        
        probs = torch.softmax(binary_logits, dim=-1)
        max_prob, local_idx = probs.max(dim=-1)

        # Map local 0/1 back to global True/False IDs
        pred_token_ids = torch.where(local_idx == 0, self.true_id, self.false_id)
        
        return max_prob, pred_token_ids

    def predict(self, input_ids, attention_mask=None):
        outputs = self.slm(
            input_ids, 
            attention_mask=attention_mask,
            output_hidden_states=True, 
            return_dict=True
        )
        logits = outputs.logits[:, -1, :]
        
        hiddens = torch.cat(
            [lyr[:, -1, :] for lyr in outputs.hidden_states[-self.num_layers:]], 
            dim=-1
        )
        
        max_prob = torch.softmax(logits, dim=-1).max(dim=-1, keepdim=True)[0]
        
        with torch.no_grad():
            self._last_conf = self.cmlp(torch.cat([hiddens, max_prob], dim=-1)).item()
            
        return logits

    def train_cmlp(self, dataloader, epochs=3, lr=5e-5):
        self.cmlp.train()
        optimizer = torch.optim.AdamW(self.cmlp.parameters(), lr=lr)
        criterion = nn.BCELoss()

        for epoch in range(epochs):
            print(epoch)
            for batch in dataloader:
                input_ids, labels = batch['input_ids'].to(self.device), batch['labels'].to(self.device)
                
                with torch.no_grad():
                    out = self.slm(input_ids, output_hidden_states=True)
                    hiddens = torch.cat([lyr[:, -1, :] for lyr in out.hidden_states[-self.num_layers:]], dim=-1)
                    max_prob, preds = self.get_binary_prediction(out.logits[:, -1, :])
                    target = (preds == labels).float().unsqueeze(-1)
                
                optimizer.zero_grad()
                feat = torch.cat([hiddens, max_prob.unsqueeze(-1)], dim=-1).to(torch.bfloat16)
                pred_conf = self.cmlp(feat)
                
                loss = criterion(pred_conf, target.to(pred_conf.dtype))
                loss.backward()
                optimizer.step()
                
        self.cmlp.eval()