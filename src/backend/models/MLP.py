from typing import Any, Iterable, Sequence

import torch
from torch import nn


class BaselineMLP(nn.Module):
	def __init__(
		self,
		input_shape: Sequence[int] = (3, 28, 28),
		hidden_dims: Iterable[int] | None = None,
		num_classes: int = 8,
		dropout: float = 0.2,
	) -> None:
		super().__init__()

		if hidden_dims is None:
			hidden_dims = (512, 256)

		self.input_shape = tuple(input_shape)
		self.hidden_dims = tuple(hidden_dims)
		self.num_classes = int(num_classes)
		self.dropout = float(dropout)

		input_features = int(torch.tensor(self.input_shape).prod().item())

		layers: list[nn.Module] = [nn.Flatten()]
		in_features = input_features
		for hidden_dim in self.hidden_dims:
			layers.extend(
				[
					nn.Linear(in_features, hidden_dim),
					nn.BatchNorm1d(hidden_dim),
					nn.ReLU(inplace=True),
					nn.Dropout(p=dropout),
				]
			)
			in_features = hidden_dim

		layers.append(nn.Linear(in_features, num_classes))

		self.mlp = nn.Sequential(*layers)
  
		self.model_config = {
			"input_shape": self.input_shape,
			"hidden_dims": list(self.hidden_dims),
			"num_classes": self.num_classes,
			"dropout": self.dropout,
			"batch_norm": True,
		}

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.mlp(x)

	def get_config(self) -> dict[str, Any]:
		"""Return a shallow copy of the architecture configuration."""
		return dict(self.model_config)

