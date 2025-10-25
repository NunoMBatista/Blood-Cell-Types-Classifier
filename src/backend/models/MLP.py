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

		"""
		Hidden dims (512, 256): 
	
			Architecture:
	
				Input: (3, 28, 28) -> Flatten() -> (3*28*28=2352) -> Linear(2352, 512) -> ReLU -> Dropout(0.2) -> (512)
				Input: (512) -> Linear(512, 256) -> ReLU -> Dropout(0.2) -> (256)
				Input: (256) -> Linear(256, 8) -> (8) 
    
			Parameter count:
				Number of parameters in a Linear layer = in_features * out_features + out_features (for bias)
				1st Linear Layer: 2352*512 + 512 = 1,204,864
				2nd Linear Layer: 512*256 + 256 = 131,328
				3rd Linear Layer: 256*8 + 8 = 2,056
				Total Linear Layer Parameters = 1,204,864 + 131,328 + 2,056 = 1,338,248
    
		"""


		layers: list[nn.Module] = [nn.Flatten()]
		in_features = input_features
		for hidden_dim in self.hidden_dims:
			layers.extend(
				[
					nn.Linear(in_features, hidden_dim, bias=True), # fully connected layer, maps input features to hidden_dim features
					nn.BatchNorm1d(hidden_dim), # normalizes the output of the linear layer to improve training stability and performance
					nn.ReLU(inplace=True), # applies the ReLU activation function to introduce non-linearity
					nn.Dropout(p=dropout), # randomly sets a fraction of input units to 0 during training to prevent overfitting
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

