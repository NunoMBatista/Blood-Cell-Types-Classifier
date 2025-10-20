"""Experiment management helpers for BloodMNIST models."""

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
    """Create and return a unique results directory for ``model_name``."""

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
) -> Dict[str, Any]:
    """Persist model artifacts and metadata for reproducibility."""

    run_dir, used_timestamp = create_results_dir(model_name, timestamp)

    model_state_file = "model_state.pt"
    optimizer_state_file = "optimizer_state.pt"
    loss_state_file = "loss_state.pt"
    confusion_file = "confusion_matrix.pt"

    torch.save(model.state_dict(), run_dir / model_state_file)
    torch.save(optimizer.state_dict(), run_dir / optimizer_state_file)
    torch.save(criterion.state_dict(), run_dir / loss_state_file)
    confusion_cpu = confusion_matrix.detach().cpu().to(torch.int64)
    torch.save(confusion_cpu, run_dir / confusion_file)

    used_model_type = _resolve_model_type(model, model_type)

    metadata: Dict[str, Any] = {
        "model_name": model_name,
        "timestamp": used_timestamp,
        "architecture": dict(architecture),
        "loss": {
            "name": criterion.__class__.__name__,
            "repr": repr(criterion),
            "state_dict_file": loss_state_file,
        },
        "optimizer": {
            "name": optimizer.__class__.__name__,
            "repr": repr(optimizer),
            "state_dict_file": optimizer_state_file,
            "defaults": {k: v for k, v in optimizer.defaults.items()},
        },
        "metrics": metrics,
        "confusion_matrix": {
            "file": confusion_file,
            "matrix": confusion_cpu.tolist(),
        },
        "artifacts": {
            "model_state": model_state_file,
            "optimizer_state": optimizer_state_file,
            "loss_state": loss_state_file,
        },
        "model_type": used_model_type,
    }

    if extra_metadata:
        metadata["extra"] = extra_metadata

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
    """List saved result directories, optionally filtered by ``model_name``."""

    root = _ensure_results_root()
    if not root.exists():
        return []

    if model_name:
        pattern = f"{_safe_name(model_name)}_*"
    else:
        pattern = "*"

    return sorted(path for path in root.glob(pattern) if path.is_dir())


def _load_metadata(results_dir: Path) -> Dict[str, Any]:
    with open(results_dir / "metadata.json", "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_experiment(
    model: nn.Module,
    results_dir: str | Path,
    optimizer: Optimizer | None = None,
    criterion: nn.Module | None = None,
    map_location: Any | None = None,
) -> Dict[str, Any]:
    """Load model (and optionally optimizer) state from ``results_dir``."""

    directory = Path(results_dir)
    metadata = _load_metadata(directory)

    model_state_path = directory / metadata["artifacts"]["model_state"]
    model.load_state_dict(torch.load(model_state_path, map_location=map_location))

    if optimizer is not None:
        optimizer_state_path = directory / metadata["artifacts"]["optimizer_state"]
        optimizer.load_state_dict(torch.load(optimizer_state_path, map_location=map_location))

    if criterion is not None:
        loss_state_path = directory / metadata["artifacts"]["loss_state"]
        if loss_state_path.exists():
            criterion.load_state_dict(torch.load(loss_state_path, map_location=map_location))

    return metadata


def load_confusion_matrix(results_dir: str | Path, map_location: Any | None = None) -> torch.Tensor:
    """Load a previously saved confusion matrix tensor."""

    directory = Path(results_dir)
    metadata = _load_metadata(directory)
    confusion_path = directory / metadata["confusion_matrix"]["file"]
    return torch.load(confusion_path, map_location=map_location)


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
    """Update per-model-type and global leaderboards with ``entry``."""

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
    """Insert ``new_entry`` into ``entries`` if it qualifies for top ``max_entries``."""

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
