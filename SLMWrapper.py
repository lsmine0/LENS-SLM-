import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

class SLMWrapper(nn.Module):
    def __init__(self, model_path, device="cuda"):
        super().__init__()
        self.device = device
        self.model_path = model_path
        
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
                    probs = torch.softmax(out.logits[:, -1, :], dim=-1)
                    max_prob, preds = probs.max(dim=-1)
                    target = (preds == labels).float().unsqueeze(-1)
                
                optimizer.zero_grad()
                pred_conf = self.cmlp(torch.cat([hiddens, max_prob.unsqueeze(-1)], dim=-1))
                target = target.to(pred_conf.dtype)
                loss = criterion(pred_conf, target)
                loss.backward()
                optimizer.step()
                
        self.cmlp.eval()