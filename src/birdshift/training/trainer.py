import torch


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    """One epoch. Pass a torch.amp.GradScaler to train in float16 mixed precision."""
    model.train()
    amp = scaler is not None
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        if amp:
            images = images.contiguous(memory_format=torch.channels_last)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            logits = model(images)
            loss = criterion(logits, labels)

        if amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return {
        "loss": total_loss / total,
        "accuracy": correct / total
    }
