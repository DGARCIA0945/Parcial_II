import torch
import torch.nn as nn
import torch.nn.functional as F

class SelectiveSSM(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Ajustamos para que cada canal tenga su propio lambda o se expanda correctamente
        self.log_lambda = nn.Parameter(torch.rand(d_model, d_state))

        self.proj_delta = nn.Linear(d_model, d_model, bias=True)
        self.proj_B     = nn.Linear(d_model, d_state, bias=False)
        self.proj_C     = nn.Linear(d_model, d_state, bias=False)

        self.out_proj = nn.Linear(d_model * d_state, d_model)

    def forward(self, x):
        B, L, D = x.shape
        N = self.d_state

        lam   = F.softplus(self.log_lambda) # (D, N)
        delta = F.softplus(self.proj_delta(x)).unsqueeze(-1) # (B, L, D, 1)
        Bt    = self.proj_B(x).unsqueeze(2)  # (B, L, 1, N)
        Ct    = self.proj_C(x).unsqueeze(2)  # (B, L, 1, N)

        # Discretización simplificada
        A_d = torch.exp(-delta * lam) # (B, L, D, N)
        B_d = delta * Bt # (B, L, D, N) - Simplificado para evitar división por cero compleja

        # Inicializamos h con la dimensión de d_model también: (B, D, N)
        h = torch.zeros(B, D, N, device=x.device)

        ys = []
        for t in range(L):
            h  = A_d[:, t, :, :] * h + B_d[:, t, :, :]
            yt = h * Ct[:, t, :, :] # Element-wise product
            ys.append(yt.view(B, 1, -1)) # Aplanamos D*N para la proyección de salida

        out = torch.cat(ys, dim=1)
        out = self.out_proj(out)
        return out

class TemporalBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, kernel_size: int = 3):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.conv = nn.Conv1d(
            d_model, d_model, kernel_size,
            padding=kernel_size // 2,
            groups=d_model
        )
        self.ssm  = SelectiveSSM(d_model, d_state)
        self.gate = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        # Conv1d espera (B, C, L)
        xc = self.conv(x.transpose(1, 2)).transpose(1, 2)
        xs = self.ssm(xc)
        g  = torch.sigmoid(self.gate(x))
        x  = self.proj(xs * g)
        return x + residual

class SpectralBlock(nn.Module):
    def __init__(self, d_model: int, seq_len: int, K: int = 32):
        super().__init__()
        self.K = K
        self.seq_len = seq_len
        self.filter_real = nn.Parameter(torch.randn(d_model, K) * 0.02)
        self.filter_imag = nn.Parameter(torch.randn(d_model, K) * 0.02)

    def forward(self, x):
        B, L, D = x.shape
        X = torch.fft.rfft(x, dim=1)
        # Promedio de magnitud para encontrar frecuencias dominantes
        mag = X.abs().mean(dim=(0, 2))
        k_actual = min(self.K, X.shape[1])
        topk_idx = mag.topk(k_actual).indices
        
        X_k = X[:, topk_idx, :]
        filt = torch.complex(self.filter_real[:, :k_actual], self.filter_imag[:, :k_actual])
        
        # Aplicar filtro
        X_k = X_k * filt.T.unsqueeze(0)
        
        X_out = torch.zeros(B, X.shape[1], D, dtype=torch.cfloat, device=x.device)
        X_out[:, topk_idx, :] = X_k
        out = torch.fft.irfft(X_out, n=L, dim=1)
        return out

class Fusion(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(d_model) * 0.5)
        self.beta  = nn.Parameter(torch.ones(d_model) * 0.5)

    def forward(self, x_temp, x_spec):
        return self.alpha * x_temp + self.beta * x_spec

class AnomalyDetector(nn.Module):
    def __init__(self, input_dim, d_model=64, d_state=16,
                 seq_len=32, K=16, n_layers=2, n_classes=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.temporal_blocks = nn.ModuleList([TemporalBlock(d_model, d_state) for _ in range(n_layers)])
        self.spectral = SpectralBlock(d_model, seq_len, K)
        self.fusion   = Fusion(d_model)
        self.norm_out = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, n_classes)
        )

    def forward(self, x):
        x = self.input_proj(x)
        xt = x
        for block in self.temporal_blocks:
            xt = block(xt)
        xs = self.spectral(x)
        z  = self.fusion(xt, xs)
        z  = self.norm_out(z)
        # Clasificamos usando el último paso de tiempo
        return self.head(z[:, -1, :])
