import torch
import os
from tqdm import tqdm
from training.eval import evaluate_model  # assumes you have eval_model() defined
import matplotlib.pyplot as plt


def train_model(model, train_loader, val_loader, criterion, optimizer, cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Training settings
    epochs = cfg['training']['epochs']
    patience = cfg['training'].get('patience', 10)
    checkpoint_path = cfg['training'].get('checkpoint_path', 'checkpoints/best_model.pth')
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    best_val_auc = 0.0
    early_stop_counter = 0
    train_loss_log = []
    train_auc_log = []

    print("🧠 Starting training...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_loss_log.append(avg_loss)
        print(f"🔁 Epoch {epoch+1} - Train Loss: {avg_loss:.4f}")

        # Validation
        val_auc = evaluate_model(model, val_loader, device=device, verbose=True)
        train_auc_log.append(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            early_stop_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✅ Validation AUC improved to {val_auc:.4f}. Model saved.")
        else:
            early_stop_counter += 1
            print(f"⏳ No improvement. Early stop counter: {early_stop_counter}/{patience}")

        if early_stop_counter >= patience:
            print("⛔ Early stopping triggered.")
            break

    # Plot training loss
    plt.plot(train_loss_log)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("results/training_loss.png")
    plt.show()
    # Plot training AUC
    plt.plot(train_auc_log)
    plt.title("Training AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.grid(True)
    plt.savefig("results/training_auc.png")
    plt.show()

    print("🏁 Training complete. Best AUC:", best_val_auc)
