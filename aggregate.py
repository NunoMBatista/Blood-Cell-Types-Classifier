import csv
import json
from pathlib import Path
from typing import Any, Dict, List


RESULTS_ROOT = Path(__file__).resolve().parent / "useful_results"
OUTPUT_DIR = RESULTS_ROOT / "aggregated_metrics"


def load_metadata(run_dir: Path) -> Dict:
	metadata_path = run_dir / "metadata.json"
	if not metadata_path.is_file():
		return {}
	with metadata_path.open("r", encoding="utf-8") as handle:
		return json.load(handle)


def gather_child_dirs(parent: Path) -> List[Path]:
	if not parent.is_dir():
		return []
	return sorted([entry for entry in parent.iterdir() if entry.is_dir()])


def format_sequence(values) -> str:
	if isinstance(values, (list, tuple)):
		return "-".join(str(item) for item in values)
	return "" if values is None else str(values)


def build_mlp_architecture_rows() -> List[Dict[str, Any]]:
	rows: List[Dict[str, Any]] = []
	run_dirs = gather_child_dirs(RESULTS_ROOT / "MLP" / "architecture_comparison")
	baseline_dir = RESULTS_ROOT / "MLP" / "baseline_MLP"
	if (baseline_dir / "metadata.json").is_file():
		run_dirs.append(baseline_dir)

	for run_dir in run_dirs:
		metadata = load_metadata(run_dir)
		if not metadata:
			continue
		architecture = metadata.get("architecture", {})
		optimizer = metadata.get("optimizer", {})
		loss = metadata.get("loss", {})
		metrics = metadata.get("metrics", {})
		rows.append(
			{
				"run": run_dir.name,
				"hidden_dims": format_sequence(architecture.get("hidden_dims")),
				"dropout": architecture.get("dropout", ""),
				"optimizer": optimizer.get("name", ""),
				"loss": loss.get("name", ""),
				"accuracy": metrics.get("accuracy", ""),
				"f1_macro": metrics.get("f1_macro", ""),
			}
		)
	return rows


def build_mlp_optimizer_rows() -> List[Dict[str, Any]]:
	rows: List[Dict[str, Any]] = []
	run_dirs = gather_child_dirs(RESULTS_ROOT / "MLP" / "optimizer_comparison")
	for run_dir in run_dirs:
		metadata = load_metadata(run_dir)
		if not metadata:
			continue
		architecture = metadata.get("architecture", {})
		optimizer = metadata.get("optimizer", {})
		defaults = optimizer.get("defaults", {})
		loss = metadata.get("loss", {})
		metrics = metadata.get("metrics", {})
		rows.append(
			{
				"run": run_dir.name,
				"optimizer": optimizer.get("name", ""),
				"learning_rate": defaults.get("lr", ""),
				"weight_decay": defaults.get("weight_decay", ""),
				"hidden_dims": format_sequence(architecture.get("hidden_dims")),
				"loss": loss.get("name", ""),
				"accuracy": metrics.get("accuracy", ""),
				"f1_macro": metrics.get("f1_macro", ""),
			}
		)
	return rows


def build_mlp_loss_rows() -> List[Dict[str, Any]]:
	rows: List[Dict[str, Any]] = []
	run_dirs = gather_child_dirs(RESULTS_ROOT / "MLP" / "loss_comparison")
	for run_dir in run_dirs:
		metadata = load_metadata(run_dir)
		if not metadata:
			continue
		architecture = metadata.get("architecture", {})
		optimizer = metadata.get("optimizer", {})
		loss = metadata.get("loss", {})
		metrics = metadata.get("metrics", {})
		rows.append(
			{
				"run": run_dir.name,
				"loss": loss.get("name", ""),
				"weighted": loss.get("weighted", ""),
				"hidden_dims": format_sequence(architecture.get("hidden_dims")),
				"optimizer": optimizer.get("name", ""),
				"accuracy": metrics.get("accuracy", ""),
				"f1_macro": metrics.get("f1_macro", ""),
			}
		)
	return rows


def build_cnn_architecture_rows() -> List[Dict[str, Any]]:
	rows: List[Dict[str, Any]] = []
	run_dirs = gather_child_dirs(RESULTS_ROOT / "CNN" / "architecture_comparison")
	baseline_dir = RESULTS_ROOT / "CNN" / "baseline_CNN"
	if (baseline_dir / "metadata.json").is_file():
		run_dirs.append(baseline_dir)

	for run_dir in run_dirs:
		metadata = load_metadata(run_dir)
		if not metadata:
			continue
		architecture = metadata.get("architecture", {})
		optimizer = metadata.get("optimizer", {})
		loss = metadata.get("loss", {})
		metrics = metadata.get("metrics", {})
		rows.append(
			{
				"run": run_dir.name,
				"conv_channels": format_sequence(architecture.get("conv_channels")),
				"linear_dims": format_sequence(architecture.get("linear_dims")),
				"dropout": architecture.get("dropout", ""),
				"optimizer": optimizer.get("name", ""),
				"loss": loss.get("name", ""),
				"accuracy": metrics.get("accuracy", ""),
				"f1_macro": metrics.get("f1_macro", ""),
			}
		)
	return rows


def build_cnn_optimizer_rows() -> List[Dict[str, Any]]:
	rows: List[Dict[str, Any]] = []
	run_dirs = gather_child_dirs(RESULTS_ROOT / "CNN" / "optimizer_comparison")
	for run_dir in run_dirs:
		metadata = load_metadata(run_dir)
		if not metadata:
			continue
		architecture = metadata.get("architecture", {})
		optimizer = metadata.get("optimizer", {})
		defaults = optimizer.get("defaults", {})
		loss = metadata.get("loss", {})
		metrics = metadata.get("metrics", {})
		rows.append(
			{
				"run": run_dir.name,
				"optimizer": optimizer.get("name", ""),
				"learning_rate": defaults.get("lr", ""),
				"weight_decay": defaults.get("weight_decay", ""),
				"conv_channels": format_sequence(architecture.get("conv_channels")),
				"linear_dims": format_sequence(architecture.get("linear_dims")),
				"loss": loss.get("name", ""),
				"accuracy": metrics.get("accuracy", ""),
				"f1_macro": metrics.get("f1_macro", ""),
			}
		)
	return rows


def build_cnn_loss_rows() -> List[Dict[str, Any]]:
	rows: List[Dict[str, Any]] = []
	run_dirs = gather_child_dirs(RESULTS_ROOT / "CNN" / "loss_comparison")
	for run_dir in run_dirs:
		metadata = load_metadata(run_dir)
		if not metadata:
			continue
		architecture = metadata.get("architecture", {})
		optimizer = metadata.get("optimizer", {})
		loss = metadata.get("loss", {})
		metrics = metadata.get("metrics", {})
		rows.append(
			{
				"run": run_dir.name,
				"loss": loss.get("name", ""),
				"weighted": loss.get("weighted", ""),
				"conv_channels": format_sequence(architecture.get("conv_channels")),
				"linear_dims": format_sequence(architecture.get("linear_dims")),
				"optimizer": optimizer.get("name", ""),
				"accuracy": metrics.get("accuracy", ""),
				"f1_macro": metrics.get("f1_macro", ""),
			}
		)
	return rows


def write_csv(rows: List[Dict[str, Any]], headers: List[str], file_path: Path) -> None:
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	with file_path.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=headers)
		writer.writeheader()
		for row in rows:
			writer.writerow({header: row.get(header, "") for header in headers})


def aggregate() -> None:
	write_csv(
		build_mlp_architecture_rows(),
		["run", "hidden_dims", "dropout", "optimizer", "loss", "accuracy", "f1_macro"],
		OUTPUT_DIR / "mlp_architecture.csv",
	)
	write_csv(
		build_mlp_optimizer_rows(),
		[
			"run",
			"optimizer",
			"learning_rate",
			"weight_decay",
			"hidden_dims",
			"loss",
			"accuracy",
			"f1_macro",
		],
		OUTPUT_DIR / "mlp_optimizer.csv",
	)
	write_csv(
		build_mlp_loss_rows(),
		["run", "loss", "weighted", "hidden_dims", "optimizer", "accuracy", "f1_macro"],
		OUTPUT_DIR / "mlp_loss.csv",
	)
	write_csv(
		build_cnn_architecture_rows(),
		[
			"run",
			"conv_channels",
			"linear_dims",
			"dropout",
			"optimizer",
			"loss",
			"accuracy",
			"f1_macro",
		],
		OUTPUT_DIR / "cnn_architecture.csv",
	)
	write_csv(
		build_cnn_optimizer_rows(),
		[
			"run",
			"optimizer",
			"learning_rate",
			"weight_decay",
			"conv_channels",
			"linear_dims",
			"loss",
			"accuracy",
			"f1_macro",
		],
		OUTPUT_DIR / "cnn_optimizer.csv",
	)
	write_csv(
		build_cnn_loss_rows(),
		[
			"run",
			"loss",
			"weighted",
			"conv_channels",
			"linear_dims",
			"optimizer",
			"accuracy",
			"f1_macro",
		],
		OUTPUT_DIR / "cnn_loss.csv",
	)


if __name__ == "__main__":
	aggregate()
