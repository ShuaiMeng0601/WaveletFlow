import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward, DWTInverse

class Mlp2d(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class WNO2D(nn.Module):
    def __init__(self, hidden_size, level=2, wave='db4', mlp_ratio=4.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.level = level
        
        self.dwt = DWTForward(J=level, wave=wave, mode='zero')
        self.idwt = DWTInverse(wave=wave, mode='zero')
        
        # d_j has shape: (B, C, 3, H', W')
        # We can process each of the 3 subbands or all together.
        # we will process (B, C*3, H', W') using an MLP, then reshape back
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        
        self.mlps = nn.ModuleList([
            Mlp2d(hidden_size * 3, mlp_hidden_dim, hidden_size * 3) for _ in range(level)
        ])
        
        # Trainable scalars: beta_j, alpha_j, eta
        # We use ParameterList for per-level scalars
        self.beta = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(level)])
        self.alpha = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(level)])
        self.eta = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(level)])

    def forward(self, x, spatial_size=None):
        bias = x

        dtype = x.dtype
        x = x.float()
        B, N, C = x.shape

        if spatial_size == None:
            H = W = int(N ** 0.5)
        else:
            H, W = spatial_size
            
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous() # (B, C, H, W)
        
        # Wavelet Transform
        yl, yh = self.dwt(x)
        
        # yh is a list of length `level`, with shapes like (B, C, 3, H_j, W_j)
        new_yh = []
        for j in range(self.level):
            # j=0 is finest scale. We can map j to scale directly. 1-indexed j_scale
            j_scale = j + 1
            
            d_j = yh[j] # (B, C, 3, H_j, W_j)
            _, _, _, hj, wj = d_j.shape
            
            # reshape to apply MLP over C*3
            d_j_reshaped = d_j.view(B, C * 3, hj, wj)
            
            # d_tilde_j = (beta + alpha * 2^(j * eta)) * MLP(d_j)
            mlp_out = self.mlps[j](d_j_reshaped)
            
            factor = self.beta[j] + self.alpha[j] * (2 ** (j_scale * self.eta[j]))
            d_tilde_j_reshaped = factor * mlp_out
            
            d_tilde_j = d_tilde_j_reshaped.view(B, C, 3, hj, wj)
            new_yh.append(d_tilde_j)
        
        x = self.idwt((yl, new_yh))
        
        # Reshape to token format
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.reshape(B, N, C)
        x = x.type(dtype)
        
        return x + bias

class WaveletBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, h=14, w=8, level=2, wave='db4', **kwargs):
        super().__init__()
        from timm.models.layers import DropPath
        self.norm1 = norm_layer(dim)

        self.filter = WNO2D(hidden_size=dim, level=level, wave=wave, mlp_ratio=mlp_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        
        from models.afno2d import Mlp
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.double_skip = True

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.filter(x)

        if self.double_skip:
            x = x + residual
            residual = x

        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + residual
        return x

if __name__ == '__main__':
    model = WNO2D(hidden_size=128, level=2)
    x = torch.randn(2, 32*32, 128)
    out = model(x)
    print("Output shape:", out.shape)
    assert out.shape == x.shape, "Shape mismatch after WNO2D"
