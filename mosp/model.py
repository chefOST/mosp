"""MOSCATO's MS-TCN++ branches and action-conditioned state refinement.

MS-TCN++ blocks are adapted from sj-li/MS-TCN2, commit
f423a9e65f4ccb1cd7322eb9f94946a19e787993 (MIT; see
third_party/MS-TCN2-LICENSE.txt). The upstream source's stray ``MS_TCB`` token
is not retained. MOSCATO specifies ten layers in all four stages; its state
branch uses sigmoid rather than the upstream action branch's softmax.

The authors have not released the MOSCATO model implementation. SAI's exact
fusion channels, confidence aggregation and boundary handling are unspecified
in the paper. The explicit choices below are recorded in MOSPConfig and
docs/reproduction-decisions.md; they are not claimed to be the authors' code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class MOSPConfig:
    num_state_classes: int
    num_actions: int
    beta: float
    feature_dim: int = 2048
    num_layers: int = 10
    num_refinements: int = 3
    num_f_maps: int = 64
    dropout: float = 0.5
    use_sai: bool = True
    sai_fusion: Literal["classwise", "dense"] = "classwise"

    def __post_init__(self) -> None:
        for name in ("num_state_classes", "num_actions", "feature_dim",
                     "num_layers", "num_f_maps"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.num_refinements, int) or self.num_refinements < 0:
            raise ValueError("num_refinements must be a nonnegative integer")
        if not 0.0 <= self.beta <= 1.0:
            raise ValueError("beta must be explicitly chosen in [0, 1]")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not self.use_sai and self.beta != 0:
            raise ValueError("use_sai=False requires beta=0")
        if self.sai_fusion not in ("classwise", "dense"):
            raise ValueError("sai_fusion must be 'classwise' or 'dense'")


class PredictionGeneration(nn.Module):
    """Dual-dilation prediction-generation stage from MS-TCN++."""

    def __init__(self, feature_dim: int, num_classes: int, num_layers: int,
                 num_f_maps: int, dropout: float):
        super().__init__()
        self.conv_1x1_in = nn.Conv1d(feature_dim, num_f_maps, 1)
        self.conv_dilated_1 = nn.ModuleList([
            nn.Conv1d(num_f_maps, num_f_maps, 3, padding=2 ** (num_layers - 1 - i),
                      dilation=2 ** (num_layers - 1 - i))
            for i in range(num_layers)
        ])
        self.conv_dilated_2 = nn.ModuleList([
            nn.Conv1d(num_f_maps, num_f_maps, 3, padding=2 ** i, dilation=2 ** i)
            for i in range(num_layers)
        ])
        self.conv_fusion = nn.ModuleList([
            nn.Conv1d(2 * num_f_maps, num_f_maps, 1) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, features: Tensor) -> Tensor:
        hidden = self.conv_1x1_in(features)
        for dilated_1, dilated_2, fusion in zip(
            self.conv_dilated_1, self.conv_dilated_2, self.conv_fusion
        ):
            residual = hidden
            hidden = fusion(torch.cat((dilated_1(hidden), dilated_2(hidden)), dim=1))
            hidden = residual + self.dropout(F.relu(hidden))
        return self.conv_out(hidden)


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation: int, num_f_maps: int, dropout: float):
        super().__init__()
        self.conv_dilated = nn.Conv1d(num_f_maps, num_f_maps, 3,
                                      padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(num_f_maps, num_f_maps, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: Tensor) -> Tensor:
        return hidden + self.dropout(self.conv_1x1(F.relu(self.conv_dilated(hidden))))


class Refinement(nn.Module):
    def __init__(self, num_classes: int, num_layers: int, num_f_maps: int,
                 dropout: float):
        super().__init__()
        self.conv_1x1 = nn.Conv1d(num_classes, num_f_maps, 1)
        self.layers = nn.ModuleList([
            DilatedResidualLayer(2 ** i, num_f_maps, dropout) for i in range(num_layers)
        ])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, probabilities: Tensor) -> Tensor:
        hidden = self.conv_1x1(probabilities)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.conv_out(hidden)


class MSTCNPlusPlus(nn.Module):
    def __init__(self, config: MOSPConfig, num_classes: int, multilabel: bool):
        super().__init__()
        self.multilabel = multilabel
        self.PG = PredictionGeneration(config.feature_dim, num_classes,
                                       config.num_layers, config.num_f_maps,
                                       config.dropout)
        self.Rs = nn.ModuleList([
            Refinement(num_classes, config.num_layers, config.num_f_maps,
                       config.dropout) for _ in range(config.num_refinements)
        ])

    def forward(self, features: Tensor) -> Tensor:
        logits = self.PG(features)
        stages = [logits]
        for refinement in self.Rs:
            probabilities = torch.sigmoid(logits) if self.multilabel else logits.softmax(dim=1)
            logits = refinement(probabilities)
            stages.append(logits)
        return torch.stack(stages)


def _correlation_tensor(values, name: str, config: MOSPConfig) -> Tensor:
    if values is None:
        raise ValueError(f"{name} training correlations are required when SAI is enabled")
    tensor = torch.as_tensor(values, dtype=torch.float32).detach().clone()
    if tensor.shape != (config.num_actions, config.num_state_classes):
        raise ValueError(f"{name} must have shape [num_actions, num_state_classes]")
    if not torch.isfinite(tensor).all() or (tensor < 0).any():
        raise ValueError(f"{name} correlations must be finite and nonnegative")
    row_sums = tensor.sum(dim=1)
    normalized = torch.isclose(row_sums, torch.ones_like(row_sums), atol=1e-5, rtol=1e-5)
    empty = row_sums == 0
    if not (normalized | empty).all():
        raise ValueError(f"{name} rows must sum to 1, or 0 for unseen actions")
    return tensor


class StateActionInteraction(nn.Module):
    """Equation 5 using predicted action segments; no annotation IDs accepted.

    An odd-length action's center frame belongs to both confidence windows,
    matching Algorithm 1's inclusive halves. State intervals use integer floor
    midpoints and form a disjoint partition. Missing endpoint evidence is zero.
    The classwise fusion interprets f as independently combining each pair's
    two inputs; dense fusion is selectable because the paper does not specify it.
    """

    def __init__(self, config: MOSPConfig, before_correlations, after_correlations):
        super().__init__()
        self.register_buffer("before_correlations",
                             _correlation_tensor(before_correlations, "before", config))
        self.register_buffer("after_correlations",
                             _correlation_tensor(after_correlations, "after", config))
        classes = config.num_state_classes
        groups = classes if config.sai_fusion == "classwise" else 1
        self.fusion = nn.Conv1d(2 * classes, classes, kernel_size=1, groups=groups)

    def action_evidence(self, probabilities: Tensor) -> Tensor:
        """Return [B, 2*C, T], interleaving after/before evidence per class.

        Discrete boundaries and selected action classes are detached; confidence
        averages remain differentiable with respect to TAS probabilities.
        """
        if probabilities.ndim != 3 or probabilities.shape[1] != self.before_correlations.shape[0]:
            raise ValueError("action probabilities must have shape [B, num_actions, T]")
        if probabilities.shape[0] == 0 or probabilities.shape[2] == 0:
            raise ValueError("action probabilities need at least one video and frame")
        if not torch.isfinite(probabilities).all() or (probabilities < 0).any():
            raise ValueError("action probabilities must be finite and nonnegative")
        if not torch.allclose(probabilities.sum(dim=1),
                              torch.ones_like(probabilities[:, 0]), atol=1e-5, rtol=1e-5):
            raise ValueError("action probabilities must sum to one at each frame")
        length = probabilities.shape[-1]
        classes = self.before_correlations.shape[1]
        batch_evidence = []
        for video_probs in probabilities:
            labels = video_probs.detach().argmax(dim=0)
            change_points = (labels[1:] != labels[:-1]).nonzero().flatten().add(1).tolist()
            boundaries = [0, *change_points, length]
            segments = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                action = int(labels[start])
                midpoint = (start + end) // 2
                before_end = (start + end + 1) // 2
                confidence_before = video_probs[action, start:before_end].mean()
                confidence_after = video_probs[action, midpoint:end].mean()
                segments.append((midpoint,
                                 confidence_before * self.before_correlations[action],
                                 confidence_after * self.after_correlations[action]))
            # Each interval gets evidence from the adjacent action segments.
            # expand/cat preserves confidence gradients, including a T=1 video.
            interval_bounds = [0, *(segment[0] for segment in segments), length]
            intervals = []
            zeros = video_probs.new_zeros(classes)
            for i, (start, end) in enumerate(zip(interval_bounds[:-1], interval_bounds[1:])):
                if start == end:
                    continue
                after = segments[i - 1][2] if i > 0 else zeros
                before = segments[i][1] if i < len(segments) else zeros
                paired = torch.stack((after, before), dim=1).reshape(2 * classes, 1)
                intervals.append(paired.expand(-1, end - start))
            batch_evidence.append(torch.cat(intervals, dim=1))
        return torch.stack(batch_evidence)

    def forward(self, action_logits: Tensor) -> Tensor:
        return self.fusion(self.action_evidence(action_logits.softmax(dim=1)))


class MOSPSystem(nn.Module):
    """Independent state/TAS MS-TCN++ branches, SAI, and Equation 8 fusion."""

    def __init__(self, config: MOSPConfig, before_correlations=None, after_correlations=None):
        super().__init__()
        self.config = config
        self.state_model = MSTCNPlusPlus(config, config.num_state_classes, multilabel=True)
        self.action_model = MSTCNPlusPlus(config, config.num_actions, multilabel=False)
        self.sai = StateActionInteraction(config, before_correlations, after_correlations) if config.use_sai else None

    def forward(self, features: Tensor) -> dict[str, Tensor | None]:
        if features.ndim != 3 or features.shape[1] != self.config.feature_dim:
            raise ValueError("features must have shape [B, feature_dim, T]")
        if features.shape[0] == 0 or features.shape[2] == 0:
            raise ValueError("features need at least one video and frame")
        if not features.is_floating_point() or not torch.isfinite(features).all():
            raise ValueError("features must be finite floating-point values")
        state_logits = self.state_model(features)
        action_logits = self.action_model(features)
        state_probabilities = state_logits[-1].sigmoid()
        sai_logits = self.sai(action_logits[-1]) if self.sai is not None else None
        probabilities = state_probabilities if sai_logits is None else (
            (1 - self.config.beta) * state_probabilities + self.config.beta * sai_logits.sigmoid()
        )
        return {"state_logits": state_logits, "action_logits": action_logits,
                "sai_logits": sai_logits, "probabilities": probabilities}
