import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple, OrderedDict
from imblearn.over_sampling import SMOTE

class FederatedClient:
    def __init__(self, client_id, dataloader, device, model):
        self.client_id, self.dataloader, self.device = client_id, dataloader, device
        self.model_class, self.local_model = type(model), None
    
    def _smote(self, X, y):
        classes = np.unique(y)
        if len(classes) < 2: return X, y
        min_cnt = min([np.sum(y==c) for c in classes])
        if min_cnt <= 5: return X, y
        return SMOTE(random_state=42, k_neighbors=min(5, min_cnt-1)).fit_resample(X, y)
    
    def train_local(self, epochs, lr, apply_smote=False):
        X = np.vstack([i.numpy() for i, _ in self.dataloader])
        y = np.concatenate([l.numpy() for _, l in self.dataloader])
        n_samples = len(y)
        if apply_smote: X, y = self._smote(X, y)
        loader = DataLoader(TensorDataset(torch.FloatTensor(X), torch.LongTensor(y)), batch_size=64, shuffle=True)
        if not self.local_model: self.local_model = self.model_class(input_dim=X.shape[1])
        self.local_model.to(self.device).train()
        cw = torch.FloatTensor([len(y)/(2*np.sum(y==i)) if np.sum(y==i)>0 else 1.0 for i in range(2)]).to(self.device)
        crit, opt = nn.CrossEntropyLoss(weight=cw), optim.Adam(self.local_model.parameters(), lr=lr)
        sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)
        for _ in range(epochs):
            loss_sum = 0
            for inp, lab in loader:
                inp, lab = inp.to(self.device), lab.to(self.device)
                opt.zero_grad()
                loss = crit(self.local_model(inp), lab)
                loss.backward()
                opt.step()
                loss_sum += loss.item()
            sched.step(loss_sum/len(loader))
        return self.local_model.get_weights(), n_samples
