"""Training loop for one epoch, shared by every experiment."""
import torch


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    """Train `model` for one pass over `loader` and return mean loss and accuracy.

    criterion: loss(logits, labels). The long-tail experiments pass a class-weighted
    or logit-adjusted loss here (see scripts/train.py).
    scaler: a torch.amp.GradScaler to train in float16 mixed precision
    (Micikevicius et al. 2018), or None for plain float32.
    """
    model.train()  # enables dropout / batch-norm statistics updates
    amp = scaler is not None
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        if amp:
            # channels_last memory layout makes float16 convolutions faster on tensor cores.
            images = images.contiguous(memory_format=torch.channels_last)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        # autocast runs the forward pass in float16 where it is safe, float32 elsewhere.
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            logits = model(images)
            loss = criterion(logits, labels)

        if amp:
            # Scale the loss so small float16 gradients do not underflow to zero,
            # then unscale before the optimizer step (skipped if gradients overflowed).
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Accumulate sums weighted by batch size so the last, smaller batch counts correctly.
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return {"loss": total_loss / total, "accuracy": correct / total}
