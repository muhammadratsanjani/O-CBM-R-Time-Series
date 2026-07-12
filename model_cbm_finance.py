import torch
import torch.nn as nn
import torch.nn.functional as F

class OrthogonalCBMLOB(nn.Module):
    def __init__(self, input_dim=40, seq_length=100, num_concepts=104, residual_dim=32, num_classes=3, mode='O-CBM+R'):
        """
        Arsitektur hibrida untuk JMLR (Ablation Study Support):
        - input_dim=40: Data LOB
        - seq_length=100: Sliding window
        - num_concepts=104: Konsep manusia dari FI-2010
        - residual_dim=32: Ruang Penemuan Bebas
        - mode: 'ERM', 'CBM', atau 'O-CBM+R'
        """
        super().__init__()
        self.mode = mode
        
        # 1. Feature Extractor (LSTM untuk Time-Series)
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=128, num_layers=2, batch_first=True)
        
        # 2. Concept Bottleneck
        self.concept_extractor = nn.Linear(128, num_concepts)
        self.bn_concept = nn.BatchNorm1d(num_concepts)
        
        # 3. Residual Bottleneck
        self.residual_extractor = nn.Linear(128, residual_dim)
        self.bn_residual = nn.BatchNorm1d(residual_dim)
        
        # 4. Final Predictor
        if mode == 'ERM':
            self.predictor = nn.Linear(128, num_classes)
        elif mode == 'CBM':
            self.predictor = nn.Linear(num_concepts, num_classes)
        else: # O-CBM+R
            self.predictor = nn.Linear(num_concepts + residual_dim, num_classes)

    def forward(self, x):
        # x shape: (Batch, Seq_Length, Input_Dim)
        lstm_out, (hn, cn) = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        
        c_pred = self.bn_concept(self.concept_extractor(last_hidden))
        res_pred = self.bn_residual(self.residual_extractor(last_hidden))
        
        if self.mode == 'ERM':
            y_pred = self.predictor(last_hidden)
        elif self.mode == 'CBM':
            y_pred = self.predictor(c_pred)
        else: # O-CBM+R
            bottleneck = torch.cat([c_pred, res_pred], dim=1)
            y_pred = self.predictor(bottleneck)
            
        return c_pred, res_pred, y_pred

def orthogonality_loss(c_pred, res_pred):
    """
    KUNCI TEORETIS JMLR: 
    Menghitung Cross-Subspace Orthogonality Penalty (Frobenius Norm) agar 
    Residual tegak lurus secara geometris dari Konsep Manusia.
    Sesuai dengan Eq. 3 di manuscript: || C_norm^T R_norm ||_F
    """
    # Normalisasi fitur-wise (kolom) untuk konsep dan residual
    c_norm = F.normalize(c_pred, p=2, dim=0)
    res_norm = F.normalize(res_pred, p=2, dim=0)
    
    # Cross-correlation matrix
    cross_corr = torch.matmul(c_norm.T, res_norm)
    
    # Menggunakan Frobenius Norm sesuai dengan revisi Methodology di JMLR
    return torch.linalg.matrix_norm(cross_corr, ord='fro')
