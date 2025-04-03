import torch
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm


def evaluate_model(model, dataloader, device=None, threshold=0.5, verbose=False):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    model.to(device)

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in tqdm(dataloader, desc="Evaluating", leave=False):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(batch_y.cpu().numpy())

    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)

    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs >= threshold).astype(int)

    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = 0.0  # AUC not defined if only one class present

    macro_f1 = f1_score(labels, preds, average='macro')

    if verbose:
        print(f"🎯 Validation AUC: {auc:.4f}")
        print(f"🎯 Validation Macro F1: {macro_f1:.4f}")

    return macro_f1
