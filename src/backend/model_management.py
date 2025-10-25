from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
from torch import nn
from torch.optim import Optimizer

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RESULTS_ROOT = _PROJECT_ROOT / "results"

_LEADERBOARD_FILENAMES = {
    "global": "top5_global.json",
    "MLP": "top5_MLP.json",
    "CNN": "top5_CNN.json",
}


def _ensure_results_root() -> Path:
    _RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    return _RESULTS_ROOT


def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in {"_", "-"} else "_" for c in name)


def create_results_dir(model_name: str, timestamp: str | None = None) -> tuple[Path, str]:
    root = _ensure_results_root()
    timestamp = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    base_name = f"{_safe_name(model_name)}_{timestamp}"
    run_dir = root / base_name
    suffix = 1
    while run_dir.exists():
        run_dir = root / f"{base_name}_{suffix:02d}"
        suffix += 1

    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir, timestamp


def record_experiment(
    model: nn.Module,
    model_name: str,
    architecture: Dict[str, Any],
    criterion: nn.Module,
    optimizer: Optimizer,
    metrics: Dict[str, Any],
    confusion_matrix: torch.Tensor,
    extra_metadata: Dict[str, Any] | None = None,
    timestamp: str | None = None,
    model_type: str | None = None,
    epochs: int | None = None,
) -> Dict[str, Any]:
    run_dir, used_timestamp = create_results_dir(model_name, timestamp)
    confusion_cpu = confusion_matrix.detach().cpu().to(torch.int64)

    used_model_type = _resolve_model_type(model, model_type)
    trainable_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))

    loss_metadata: Dict[str, Any] = {
        "name": criterion.__class__.__name__,
        "repr": repr(criterion),
    }

    weight_tensor = getattr(criterion, "weight", None)
    if isinstance(weight_tensor, torch.Tensor) and weight_tensor.numel() > 0:
        loss_metadata["weighted"] = True
        loss_metadata["class_weights"] = weight_tensor.detach().cpu().tolist()
    else:
        loss_metadata["weighted"] = False

    metadata: Dict[str, Any] = {
        "model_name": model_name,
        "timestamp": used_timestamp,
        "architecture": dict(architecture),
        "trainable_parameters": trainable_params,
        "loss": loss_metadata,
        "optimizer": {
            "name": optimizer.__class__.__name__,
            "repr": repr(optimizer),
            "defaults": {k: v for k, v in optimizer.defaults.items()},
        },
        "metrics": metrics,
        "confusion_matrix": {
            "matrix": confusion_cpu.tolist(),
        },
        "model_type": used_model_type,
    }

    if extra_metadata:
        metadata["extra"] = extra_metadata

    if epochs is not None:
        metadata["epochs"] = int(epochs)

    with open(run_dir / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    _update_leaderboards(
        leaderboard_root=_ensure_results_root(),
        entry={
            "model_name": model_name,
            "model_type": used_model_type,
            "results_dir": str(run_dir),
            "timestamp": used_timestamp,
            "metrics": metrics,
        },
    )

    return {"path": run_dir, "metadata": metadata}


def list_result_runs(model_name: str | None = None) -> list[Path]:
    root = _ensure_results_root()
    if not root.exists():
        return []

    if model_name:
        pattern = f"{_safe_name(model_name)}_*"
    else:
        pattern = "*"

    return sorted(path for path in root.glob(pattern) if path.is_dir())


def load_metadata(results_dir: str | Path) -> Dict[str, Any]:
    directory = Path(results_dir)
    with open(directory / "metadata.json", "r", encoding="utf-8") as fh:
        return json.load(fh)


def _resolve_model_type(model: nn.Module, explicit_type: str | None) -> str:
    if explicit_type:
        return explicit_type

    model_name = model.__class__.__name__.lower()
    if "mlp" in model_name:
        return "MLP"
    if "cnn" in model_name:
        return "CNN"
    return "OTHER"


def _update_leaderboards(leaderboard_root: Path, entry: Dict[str, Any]) -> None:
    leaderboard_root.mkdir(parents=True, exist_ok=True)

    model_type = entry.get("model_type", "OTHER")
    files = _leaderboard_targets_for(model_type, leaderboard_root)

    for _, file_path in files:
        current = _load_leaderboard(file_path)
        updated = _merge_leaderboard(current, entry)
        _save_leaderboard(file_path, updated)


def _leaderboard_targets_for(model_type: str, root: Path) -> List[Tuple[str, Path]]:
    boards: List[Tuple[str, Path]] = []
    for key, filename in _LEADERBOARD_FILENAMES.items():
        if key == "global" or key == model_type:
            boards.append((key, root / filename))
    return boards


def _load_leaderboard(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as fh:
        loaded = json.load(fh)
    if isinstance(loaded, list):
        return loaded
    return []


def _save_leaderboard(path: Path, entries: Iterable[Dict[str, Any]]) -> None:
    ordered = list(entries)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(ordered, fh, indent=2)


def _merge_leaderboard(
    entries: List[Dict[str, Any]],
    new_entry: Dict[str, Any],
    max_entries: int = 5,
) -> List[Dict[str, Any]]:
    filtered = [e for e in entries if e.get("results_dir") != new_entry.get("results_dir")]
    filtered.append(new_entry)
    filtered.sort(key=_leaderboard_sort_key, reverse=True)
    return filtered[:max_entries]


def _leaderboard_sort_key(entry: Dict[str, Any]) -> Tuple[float, float, str]:
    metrics = entry.get("metrics", {})
    accuracy = float(metrics.get("accuracy", float("nan")))
    f1_macro = float(metrics.get("f1_macro", float("nan")))
    # In case of missing metrics we fall back to very low values to keep ordering predictable.
    if accuracy != accuracy:
        accuracy = float("-inf")
    if f1_macro != f1_macro:
        f1_macro = float("-inf")
    return accuracy, f1_macro, entry.get("timestamp", "")
