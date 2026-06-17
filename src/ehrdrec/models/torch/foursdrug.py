import torch
import torch.nn as nn

class FourSDrug(nn.Module):
    def __init__(self, num_symptoms, num_drugs, emb_dim=64):
        super().__init__()

        self.symptom_emb = nn.Embedding(num_symptoms, emb_dim)
        self.drug_emb = nn.Embedding(num_drugs, emb_dim)
        self.symptom_importance = nn.Parameter(torch.zeros(num_symptoms))

    def encode_symptoms(self, symptoms):
        symptoms = symptoms.float()

        importance = torch.exp(self.symptom_importance)  # [S]
        weighted = symptoms * importance.unsqueeze(0)    # [B, S]

        denom = weighted.sum(dim=1, keepdim=True).clamp_min(1e-8)
        h_s = weighted @ self.symptom_emb.weight
        h_s = h_s / denom

        return h_s

    def forward(self, symptoms):
        h_s = self.encode_symptoms(symptoms)
        return {"predictions": h_s @ self.drug_emb.weight.T}