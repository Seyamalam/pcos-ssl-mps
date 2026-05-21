from __future__ import annotations

import torch
from tqdm import tqdm

from pcos_ssl.training.metrics import binary_classification_metrics


def train_one_epoch(model, loader, criterion, optimizer, device, max_batches: int | None) -> float:
    model.train()
    total_loss = 0.0
    seen = 0
    progress = tqdm(loader, desc="train", leave=False)
    for batch_idx, (images, labels) in enumerate(progress, start=1):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        seen += batch_size
        progress.set_postfix(loss=f"{total_loss / seen:.4f}")
        if max_batches is not None and batch_idx >= max_batches:
            break
    return total_loss / max(seen, 1)


@torch.inference_mode()
def evaluate(model, loader, criterion, device, max_batches: int | None) -> tuple[float, dict]:
    model.eval()
    total_loss = 0.0
    seen = 0
    y_true: list[int] = []
    y_prob: list[float] = []

    progress = tqdm(loader, desc="eval", leave=False)
    for batch_idx, (images, labels) in enumerate(progress, start=1):
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)[:, 1]

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        seen += batch_size
        y_true.extend(labels.cpu().tolist())
        y_prob.extend(probs.cpu().tolist())
        if max_batches is not None and batch_idx >= max_batches:
            break

    metrics = binary_classification_metrics(y_true, y_prob)
    return total_loss / max(seen, 1), metrics

