#!/usr/bin/env python3

import torch
import torch.nn as nn
import numpy as np
import time
from torch.utils.data import DataLoader, TensorDataset

def sanitize_data(data):
    """Replaces any Infinity or NaN values caused by dx/dt division by zero"""
    data[np.isinf(data)] = np.nan
    data = np.nan_to_num(data, nan=0.0) # Replace NaNs with 0
    return data

# KalmanNet Architecture
class KalmanNet(nn.Module):
    def __init__(self, m=6, n=3, N_rnn=32):
        super().__init__()
        self.m, self.n, self.N_rnn = m, n, N_rnn
        
        self.GRU_Q = nn.GRUCell(n, N_rnn)
        self.GRU_Sigma = nn.GRUCell(m, N_rnn) 
        self.GRU_S = nn.GRUCell(n, N_rnn)
        
        feat_dim = N_rnn*3 + n + m
        self.norm = nn.LayerNorm(feat_dim)
        
        self.fc1 = nn.Linear(feat_dim, N_rnn)
        self.fc2 = nn.Linear(N_rnn, N_rnn)
        self.fc3 = nn.Linear(N_rnn, n*m) 
        
        nn.init.uniform_(self.fc3.weight, -0.001, 0.001)
        nn.init.zeros_(self.fc3.bias)
        
        self.H = nn.Linear(m, n)
        # Initialize to zero so early epochs just use raw IMU as innovation
        nn.init.zeros_(self.H.weight)
        nn.init.zeros_(self.H.bias)
        
        dt = 0.01 # 100Hz
        F = torch.eye(m)
        F[0, 3] = dt; F[1, 4] = dt; F[2, 5] = dt
        self.F = nn.Parameter(F, requires_grad=False)

    def forward(self, Yseq, x0):
        B, T, _ = Yseq.shape
        device = Yseq.device
        
        x_post = torch.zeros(B, T, self.m, device=device)
        x_post[:, 0, :] = x0 
        
        h_Q = torch.zeros(B, self.N_rnn, device=device)
        h_Sigma = torch.zeros(B, self.N_rnn, device=device)
        h_S = torch.zeros(B, self.N_rnn, device=device)
        
        for t in range(1, T):
            y = Yseq[:, t, :]         
            y_prev = Yseq[:, t-1, :]  
            delta_y = y - y_prev
            
            e_post = x_post[:, t-1, :] - x_post[:, t-2, :] if t > 1 else torch.zeros_like(x_post[:, 0, :])
            
            h_Q = self.GRU_Q(delta_y, h_Q)
            h_Sigma = self.GRU_Sigma(e_post, h_Sigma)
            h_S = self.GRU_S(delta_y, h_S)
            
            feat = torch.cat([h_Q, h_Sigma, h_S, y, x_post[:, t-1, :]], dim=1)
            feat = self.norm(feat)
            
            feat = torch.relu(self.fc1(feat))
            feat = torch.relu(self.fc2(feat))
            K = self.fc3(feat).view(B, self.m, self.n)
            
            x_prior = torch.matmul(self.F, x_post[:, t-1, :].unsqueeze(2)).squeeze(2)
            
            y_pred = self.H(x_prior)
            innov = y - y_pred 
            
            correction = torch.bmm(K, innov.unsqueeze(2)).squeeze(2)
            x_post[:, t, :] = x_prior + correction
            
        return x_post

# Evaluation
def compute_metrics(pred, true, inference_time_ms):
    error = pred - true
    pos_error = error[:, :, :3] 
    
    mse = torch.mean(error**2).item()
    rmse = torch.sqrt(torch.mean(pos_error**2)).item()
    mae = torch.mean(torch.abs(pos_error)).item()
    var_error = torch.var(error).item()
    
    distances = torch.norm(pos_error, dim=2) 
    threshold = 1.0 # 1 meter precision
    true_positives = torch.sum(distances < threshold).item()
    total_points = distances.numel()
    precision_1m = (true_positives / total_points) * 100.0
    
    return {
        "MSE (State overall)": mse,
        "RMSE (Position)": f"{rmse:.4f} meters",
        "MAE (Position)": f"{mae:.4f} meters",
        "Error Variance": f"{var_error:.4f}",
        "Inlier Precision (<1m)": f"{precision_1m:.2f} %",
        "Latency": f"{inference_time_ms:.3f} ms / step"
    }

# Training 
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Training on Device: {device} ---")

    print("Loading and Sanitizing datasets...")
    train_np = np.load('nclt_train.npz')
    val_np   = np.load('nclt_val.npz')
    test_np  = np.load('nclt_test.npz')

    train_x = sanitize_data(train_np['x'])
    train_y = sanitize_data(train_np['y'])
    val_x   = sanitize_data(val_np['x'])
    val_y   = sanitize_data(val_np['y'])
    test_x  = sanitize_data(test_np['x'])
    test_y  = sanitize_data(test_np['y'])

    BATCH_SIZE = 32
    train_loader = DataLoader(TensorDataset(torch.tensor(train_x).float(), torch.tensor(train_y).float()), 
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(torch.tensor(val_x).float(), torch.tensor(val_y).float()), 
                              batch_size=BATCH_SIZE)
    test_loader  = DataLoader(TensorDataset(torch.tensor(test_x).float(), torch.tensor(test_y).float()), 
                              batch_size=BATCH_SIZE)

    model = KalmanNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.SmoothL1Loss() 
    
    EPOCHS = 100
    best_val_loss = float('inf')

    print(f"Starting training for {EPOCHS} epochs...\n")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            
            x0 = x_batch[:, 0, :]
            pred_x = model(y_batch, x0)
            
            loss = criterion(pred_x[:, 1:, :], x_batch[:, 1:, :])
            loss.backward()
            
            # Clip gradients tightly so one bad outlier doesn't ruin the model
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item() * x_batch.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation Loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                
                x0 = x_val[:, 0, :]
                pred_val = model(y_val, x0)
                
                loss = criterion(pred_val[:, 1:, :], x_val[:, 1:, :])
                val_loss += loss.item() * x_val.size(0)
                
        val_loss /= len(val_loader.dataset)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_knet_nclt.pt')
            saved_flag = " (Saved Best Model)"
        else:
            saved_flag = ""

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{EPOCHS} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}{saved_flag}")

    # Test set evaluation
    print("\n--- Training Complete. Evaluating on Test Set ---")
    model.load_state_dict(torch.load('best_knet_nclt.pt'))
    model.eval()
    
    all_preds, all_trues = [],[]
    total_time = 0.0
    total_steps = 0
    
    with torch.no_grad():
        for x_test, y_test in test_loader:
            x_test, y_test = x_test.to(device), y_test.to(device)
            
            start_time = time.time()
            pred_test = model(y_test, x_test[:, 0, :])
            end_time = time.time()
            
            total_time += (end_time - start_time)
            total_steps += (x_test.shape[0] * x_test.shape[1]) 
            
            all_preds.append(pred_test[:, 1:, :].cpu())
            all_trues.append(x_test[:, 1:, :].cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_trues = torch.cat(all_trues, dim=0)
    
    inference_ms_per_step = (total_time / total_steps) * 1000.0
    metrics = compute_metrics(all_preds, all_trues, inference_ms_per_step)
    
    print("\nFinal Metrics as follows -")
    for k, v in metrics.items():
        print(f"{k:<25}: {v}")
            
if __name__ == '__main__':
    train_model()