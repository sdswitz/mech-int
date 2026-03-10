import torch
import torch.nn as nn
import torch.nn.functional as F

def SAEloss(xhat, x, f, lam=1e-3):
    return F.mse_loss(xhat, x) + lam * f.abs().mean()

class SparseAutoEncoder(nn.Module):
    def __init__(self, d, m):
        super().__init__()
        self.b_pre = nn.Parameter(torch.zeros(d))
        self.b_enc = nn.Parameter(torch.zeros(m))
        self.W_enc = nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(d,m)))
        self.W_dec = nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(m,d)))
        self.W_dec = nn.Parameter(self.W_dec / self.W_dec.norm(dim=1, keepdim=True))
        
        self.act = nn.ReLU()
        
    def encoder(self, x):
        x = x - self.b_pre
        x = x @ self.W_enc
        x = x + self.b_enc
        x = self.act(x)
        return x
    
    def decoder(self, f):
        xhat = f @ self.W_dec
        xhat = xhat + self.b_pre
        return xhat
    
    def forward(self, x):
        f = self.encoder(x)
        xhat = self.decoder(f)
        return xhat, f