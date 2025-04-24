from tqdm import tqdm
from training.eval import evaluate_model


def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device):
    model.to(device)

    if val_loader is None:
        print("No validation set provided. Skipping evaluation.")

    # Training settings
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

    if val_loader is None:
        return None
    
    model.eval()
    val_macro_f1 = evaluate_model(model, val_loader, device)

    return val_macro_f1