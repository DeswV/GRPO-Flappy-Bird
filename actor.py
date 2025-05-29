import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FlappyBirdActor(nn.Module):
    def __init__(self, input_dim: int = 12, hidden_dims: tuple[int] = (128, 128, 64), output_dim: int = 2):
        super().__init__()
        self.config = {
            "input_dim": input_dim,
            "hidden_dims": hidden_dims,
            "output_dim": output_dim
        }

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)

    def forward(self, x):
        x = self.backbone(x)
        logits = self.output_layer(x)
        return logits

    def get_action(self, observation: list | np.ndarray | torch.Tensor):
        """
        :param observation: shape: [12]
        :return: action: int, probs: list[float]
        """
        if isinstance(observation, (list, np.ndarray)):
            observation = torch.tensor(observation, dtype=torch.float)
        assert observation.ndim == 1, "Observation must be a 1D tensor."
        observation.to(self.output_layer.weight.device)

        logits = self.forward(observation)
        probs = F.softmax(logits, dim=-1)
        action = torch.multinomial(probs, 1).item()
        return action, probs.tolist()

    def save_checkpoint(self, file_path: str):
        torch.save({
            "model_state_dict": self.state_dict(),
            "config": self.config
        }, file_path)

    @classmethod
    def load_checkpoint(cls, file_path: str):
        checkpoint = torch.load(file_path, map_location='cpu')
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["model_state_dict"])
        return model
