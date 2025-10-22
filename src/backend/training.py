"""Training utilities for BloodMNIST classifiers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch
from torch import nn
from tqdm.auto import tqdm


@dataclass
class EpochMetrics:
    train_loss: float
    train_accuracy: float
    val_loss: float | None
    val_accuracy: float | None


def train_one_epoch(
    model: nn.Module,
    data_loader: Iterable,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch_index: int,
    total_epochs: int,
) -> Tuple[float, float]:
    """Train ``model`` for one epoch and return loss and accuracy."""

    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0

    progress = tqdm(
        data_loader,
        desc=f"Epoch {epoch_index}/{total_epochs}",
        leave=False,
    )

    # iterate through every batch
    for inputs, targets in progress:
        inputs = inputs.to(device)
        targets = targets.squeeze().to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * targets.size(0)
        preds = logits.argmax(dim=1)
        running_correct += (preds == targets).sum().item()
        total += targets.size(0)

        average_loss = running_loss / max(1, total)
        average_acc = running_correct / max(1, total)
        progress.set_postfix({"loss": f"{average_loss:.4f}", "acc": f"{average_acc:.4f}"})

    progress.close()

    epoch_loss = running_loss / max(1, total)
    epoch_accuracy = running_correct / max(1, total)
    return epoch_loss, epoch_accuracy


def evaluate_model(
    model: nn.Module,
    data_loader: Iterable,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate ``model`` returning loss and accuracy."""

    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0

    with torch.no_grad():
        for batch in data_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.squeeze().to(device)

            logits = model(inputs)
            loss = criterion(logits, targets)

            running_loss += loss.item() * targets.size(0)
            preds = logits.argmax(dim=1)
            running_correct += (preds == targets).sum().item()
            total += targets.size(0)

    epoch_loss = running_loss / max(1, total)
    epoch_accuracy = running_correct / max(1, total)
    return epoch_loss, epoch_accuracy


def train_model(
    model: nn.Module,
    train_loader: Iterable,
    val_loader: Iterable,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 5,
) -> Tuple[List[EpochMetrics], Dict[str, torch.Tensor]]:
    """Train the model for ``epochs`` and return metrics and best weights."""

    history: List[EpochMetrics] = [] # this will store metrics for every epoch
    
    best_state = deepcopy(model.state_dict()) # this will store the best model weights
    best_val_acc = -float("inf") # initialize best validation accuracy

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch_index=epoch,
            total_epochs=epochs,
        )

        val_loss = None
        val_acc = None
        
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = deepcopy(model.state_dict())

        history.append(
            EpochMetrics(
                train_loss=train_loss,
                train_accuracy=train_acc,
                val_loss=val_loss,
                val_accuracy=val_acc,
            )
        )

    return history, best_state


def predict_labels(
    model: nn.Module,
    data_loader: Iterable,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run inference and return (predictions, targets)."""

    model.eval()
    preds: List[torch.Tensor] = []
    targets_all: List[torch.Tensor] = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.squeeze().to(device)

            logits = model(inputs)
            batch_preds = logits.argmax(dim=1)

            preds.append(batch_preds.cpu())
            targets_all.append(targets.cpu())

    return torch.cat(preds), torch.cat(targets_all)


def compute_confusion_matrix(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Compute a confusion matrix given predicted and true labels."""

    matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for pred, target in zip(predictions, targets):
        matrix[target.long(), pred.long()] += 1
    return matrix