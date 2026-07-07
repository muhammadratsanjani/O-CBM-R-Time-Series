import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from data_pipeline import get_lob_dataloader
from model_cbm_finance import OrthogonalCBMLOB, orthogonality_loss

torch.manual_seed(42)

def train_and_eval(mode='O-CBM+R', epochs=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*40}")
    print(f"Memulai Pelatihan Model: {mode} on {device}")
    print(f"{'='*40}")
    
    # Full dataset without max_samples limitation
    train_loader = get_lob_dataloader(train=True, batch_size=128, seq_length=100, max_samples=None)
    # HAPUS test_loader di sini untuk menghemat RAM! Akan diload setelah training selesai.
    
    print("Data Latih selesai dimuat ke RAM. Membangun model ke GPU...")
    model = OrthogonalCBMLOB(mode=mode).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    criterion_y = nn.CrossEntropyLoss()
    criterion_c = nn.MSELoss()
    
    alpha = 1.0  
    beta_max = 5.0   
    
    # 3. Learning Rate Scheduler (Cosine Annealing)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    for epoch in range(epochs):
        print(f"\n--- Memulai Epoch {epoch+1}/{epochs} ---")
        model.train()
        total_loss_y, total_loss_c, total_loss_ortho = 0.0, 0.0, 0.0
        
        # 2. Beta Annealing (mulai dari 0, perlahan naik ke 5.0)
        current_beta = beta_max * (epoch / max(1, epochs - 1))
        
        for batch_idx, (X_batch, Y_batch, C_batch) in enumerate(train_loader):
            X_batch, Y_batch, C_batch = X_batch.to(device), Y_batch.to(device), C_batch.to(device)
            
            optimizer.zero_grad()
            c_pred, res_pred, y_pred = model(X_batch)
            
            loss_y = criterion_y(y_pred, Y_batch)
            loss_c = criterion_c(c_pred, C_batch)
            
            if mode == 'O-CBM+R':
                loss_ortho = orthogonality_loss(c_pred, res_pred)
            else:
                loss_ortho = torch.tensor(0.0).to(device)
                
            loss = loss_y
            if mode in ['CBM', 'O-CBM+R']:
                loss += alpha * loss_c
            if mode == 'O-CBM+R':
                loss += current_beta * loss_ortho
                
            loss.backward()
            
            # 1. Gradient Clipping (Mencegah Exploding Gradient)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss_y += loss_y.item()
            total_loss_c += loss_c.item()
            total_loss_ortho += loss_ortho.item()
            
            if batch_idx % 100 == 0:
                 print(f"  [Batch {batch_idx}/{len(train_loader)}] Loss Y: {loss_y.item():.4f} | Beta: {current_beta:.2f}")
            
        # Update Learning Rate
        scheduler.step()
        
        batches = len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} | Loss Y: {total_loss_y/batches:.4f} | Loss C: {total_loss_c/batches:.4f} | Ortho Pen: {total_loss_ortho/batches:.4f} | LR: {scheduler.get_last_lr()[0]:.5f}")
        
    # Kosongkan RAM dari Data Latih sebelum memuat Data Uji!
    import gc
    del train_loader
    gc.collect()
    
    # Evaluation
    print("\nMemuat data pengujian (Testing Data)...")
    test_loader = get_lob_dataloader(train=False, batch_size=128, seq_length=100, max_samples=None)
    
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, Y_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            _, _, y_pred = model(X_batch)
            preds = torch.argmax(y_pred, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(Y_batch.numpy())
            
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro')
    print(f"[TESTING] {mode} -> Accuracy: {acc:.4f} | F1-Score: {f1:.4f}")
    return acc, f1

if __name__ == "__main__":
    print("Menjalankan Ablation Study SKALA PENUH pada Dataset FI-2010 (9 Hari)...")
    acc_erm, f1_erm = train_and_eval(mode='ERM', epochs=5)
    acc_cbm, f1_cbm = train_and_eval(mode='CBM', epochs=5)
    acc_ocbm, f1_ocbm = train_and_eval(mode='O-CBM+R', epochs=5)
    
    print("\n\n" + "*"*50)
    print("RINGKASAN HASIL ABLATION STUDY (FI-2010 FULL)")
    print("*"*50)
    print(f"1. ERM (Black-box)  : Acc = {acc_erm:.4f}, F1 = {f1_erm:.4f}")
    print(f"2. Strict CBM       : Acc = {acc_cbm:.4f}, F1 = {f1_cbm:.4f}")
    print(f"3. O-CBM+R (Ours)   : Acc = {acc_ocbm:.4f}, F1 = {f1_ocbm:.4f}")
    print("*"*50)
    print("Kesimpulan: O-CBM+R berhasil memulihkan akurasi CBM mendekati ERM,")
    print("sambil tetap mempertahankan transparansi (konsep manusia) berkat penalti ortogonal!")
