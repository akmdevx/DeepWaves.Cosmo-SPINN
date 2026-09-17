import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import math
import matplotlib.pyplot as plt

from ./utils import util.py
from ./Models import Evol_Net.py#, or Evol_Net2.py, or SR_Net.py ; Use according to the task


##################
# Training
##################
def train_epoch(G, loader, optimizer, device,labels,epoch):
    """Train for one epoch with L2 loss"""
    G.train()
    total_loss = 0
    mse_total_loss = 0    
    for x, y_real, s in loader:
        x = x.to(device)
        y_real = y_real.to(device)
        s = s.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        y_pred = G(x, s)

        index = torch.randint(
            low=0,
            high=len(labels),
            size=(len(s),),
            device=device,
        )
        
        s_ind = labels[index]

        #print(s_ind.shape)
        delta = 1e-5  # Same as 0.001
        
        a_min_l = -0.8273942837611394
        a_max_l = -0.8048255392272029# --- build scale factors a, a2 and a_dott the same way you did --# your snippet used: z = (labels * 0.5 + 0.5) * 7.0 then z = 7 - z % 6 ...
        a_ind = (s_ind * 0.5 + 0.5)
        a_ind = a_ind*(a_max_l - a_min_l)+a_min_l
        a_ind = 10**a_ind
        a_ind2 = a_ind+delta
        a_ind2 = torch.log10(a_ind2)
        a_ind2 = (a_ind2-a_min_l)/(a_max_l-a_min_l)
        s_ind2 = (a_ind2-0.5)/0.5
        #s_ind2 = s_ind + torch.log1p(delta / (10**s_ind))
        x_ind = x.clone()#[index].clone()
        gen = G(x_ind,s_ind)
        g2 = G(x_ind,s_ind2)
        phy_loss = compute_pres(gen,g2,s_ind,s_ind2,stats)
        print(phy_loss)
        # L2 loss
        loss_data = F.mse_loss(y_pred,y_real)
        loss_phy = phy_loss

        grads = torch.autograd.grad(loss_data,
                                    G.parameters(),
                                    retain_graph=True)
        
        g_data = torch.sqrt(sum(g.pow(2).sum() for g in grads if g is not None))
        
        grads = torch.autograd.grad(loss_phy,
                                    G.parameters(),
                                    retain_graph=True)
        
        g_phy = torch.sqrt(sum(g.pow(2).sum() for g in grads if g is not None))


        ramp_epochs = 1000#750
        alpha = 0.02*min(1.0, epoch / ramp_epochs)

        #loss = loss_data + alpha * lambda_phy * phy_loss
        
        if alpha > 0:

            lambda_phy = g_data.detach() / (g_phy.detach() + 1e-12)
        
            loss = loss_data + alpha * lambda_phy * loss_phy
        
        else:
            loss = loss_data

        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        mse_total_loss += loss_data.item()
    
    return total_loss / len(loader), mse_total_loss/len(loader)

##################
# Main
##################
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load data


    save_dir1 = "link to the path for EVOL_DATA_SR and EVOL_DATA_TRAIN_SR from Zenodo"
    fdm_train_np = np.load(os.path.join(save_dir1, "fdm_train.npy"))
    labels_train_np = np.load(os.path.join(save_dir1, "labels_train.npy"))
    ic_train_np = np.load(os.path.join(save_dir1, "ic_train.npy"))

    fdm_test_np = np.load(os.path.join(save_dir1, "fdm_test.npy"))
    labels_test_np = np.load(os.path.join(save_dir1, "labels_test.npy"))
    ic_test_np = np.load(os.path.join(save_dir1, "ic_test.npy"))

    stats = np.load(os.path.join(save_dir1, "stats.npy"),allow_pickle = True).item()
    # ---- convert to torch ----
    fdm_train = torch.tensor(fdm_train_np, dtype=torch.float32)
    labels_train = torch.tensor(labels_train_np, dtype=torch.float32)
    ic_train = torch.tensor(ic_train_np, dtype=torch.float32)
    #ic_train_real = torch.tensor(ic_train_np_real, dtype = torch.float32)
    max_val = np.load(os.path.join(save_dir1,"max_val.npy"))
    min_val = np.load(os.path.join(save_dir1,"min_val.npy"))

    fdm_test = torch.tensor(fdm_test_np, dtype=torch.float32)
    labels_test = torch.tensor(labels_test_np, dtype=torch.float32)
    ic_test = torch.tensor(ic_test_np, dtype=torch.float32)
    
    print(f"Training set size: {train_size}")
    print(f"Test set size: {test_size}")
    
    # Create datasets
    train_dataset = TensorDataset(ic_train, fdm_train, labels_train)
    test_dataset = TensorDataset(ic_test, fdm_test, labels_test)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    print(f"Train DataLoader created with {len(train_loader)} batches")
    print(f"Test DataLoader created with {len(test_loader)} batches")
    
    # Initialize model
    print("Initializing generator...")
    G = ConditionalGenerator(cond_dim=64).to(device)
    # Load the generator checkpoint of the model just trained with MSE Loss for 250 epochs
    # Optimizer
    optimizer = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.9, 0.999))
    
    # Load checkpoint if exists
    checkpoint_dir = "..."
    g_checkpoint = os.path.join(checkpoint_dir, "generator_l2.pt")
    
    start_epoch = 0
    if os.path.exists(g_checkpoint):
        print("Loading checkpoint...")
        G.load_state_dict(torch.load(g_checkpoint))
        print("Checkpoint loaded")
    
    # Training loop
    epochs = 2750
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(start_epoch, epochs):
        # Train on training set
        train_loss, mse_loss = train_epoch(G, train_loader, optimizer, device,labels,epoch)
        
        # Evaluate on test set
        G.eval()
        test_loss = 0
        with torch.no_grad():
            for x, y, s in test_loader:
                x, y, s = x.to(device), y.to(device), s.to(device)
                y_pred = G(x, s)
                test_loss += F.mse_loss(y_pred, y).item()
        test_loss /= len(test_loader)
        G.train()
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {mse_loss:.6f} | Test Loss: {test_loss:.6f}")
        
        # Save images every 10 epochs (using training data for visualization)
        if (epoch + 1) % 10 == 0:
            print(f"Saving images for epoch {epoch+1}...")
            sample_images(epoch+1, G, fdm_train, labels_train, ic_train, stats)
        
        # Save checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            print(f"Saving checkpoint at epoch {epoch+1}...")
            torch.save(G.state_dict(), g_checkpoint)
            
            # Also save versioned checkpoint
            torch.save(G.state_dict(), 
                      os.path.join(checkpoint_dir, f"generator_l2_epoch_{epoch+1}_10_2.pt"))
    
    # Final save
    print("Training complete! Saving final model...")
    torch.save(G.state_dict(), g_checkpoint)
