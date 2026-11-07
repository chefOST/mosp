"""Joint MOSP, SAI, and TAS objectives (MOSCATO Equations 4 and 7).

State BCE is applied at every MS-TCN++ stage, following upstream multistage
supervision; SAI has one output. TAS smoothing follows pinned MS-TCN2: squared
log-probability changes clipped to 16, detached preceding frame, weight 0.15.
These explicit conventions are recorded in docs/reproduction-decisions.md.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor
from torch.nn import functional as F


def _masked_mean(values: Tensor, valid: Tensor) -> Tensor:
    valid = valid.expand_as(values)
    # Retain the graph for empty padded batches rather than returning NaN.
    return (values * valid).sum() / valid.sum().clamp_min(1)


def joint_loss(outputs: Mapping[str, Tensor | None], state_targets: Tensor,
               action_targets: Tensor, pos_weight: float = 5.0,
               smoothing_weight: float = 0.15, valid_mask: Tensor | None = None,
               ignore_index: int = -100) -> dict[str, Tensor]:
    """Return total, mosp, sai, tas, tas_ce and tas_smoothing scalar tensors.

    Inputs have [K,B,C,T] state logits, [K,B,A,T] action logits, [B,C,T]
    binary/multilabel targets, and [B,T] action IDs. Optional valid_mask [B,T]
    excludes padded frames from every objective; ignore_index excludes unknown
    action labels from TAS only. No target labels are passed into model.forward.
    """
    states = outputs["state_logits"]
    actions = outputs["action_logits"]
    sai_logits = outputs.get("sai_logits")
    if not isinstance(states, Tensor) or not isinstance(actions, Tensor):
        raise ValueError("state_logits and action_logits must be tensors")
    if states.ndim != 4 or actions.ndim != 4 or states.shape[0] < 1 or actions.shape[0] < 1:
        raise ValueError("stage logits must have shape [K,B,C,T] with K>=1")
    if state_targets.shape != states.shape[1:]:
        raise ValueError("state_targets must have shape [B,C,T] matching state logits")
    batch, classes, length = states.shape[1:]
    if batch == 0 or classes == 0 or length == 0:
        raise ValueError("state logits need nonempty batch, classes and time")
    if action_targets.shape != (batch, length) or actions.shape[1] != batch or actions.shape[-1] != length:
        raise ValueError("action targets/logits must match the state batch and time")
    if action_targets.dtype != torch.long:
        raise ValueError("action_targets must be torch.long")
    if not math.isfinite(pos_weight) or pos_weight <= 0:
        raise ValueError("pos_weight must be positive and finite")
    if not math.isfinite(smoothing_weight) or smoothing_weight < 0:
        raise ValueError("smoothing_weight must be nonnegative and finite")
    if not torch.isfinite(state_targets).all() or ((state_targets < 0) | (state_targets > 1)).any():
        raise ValueError("state_targets must be finite values in [0,1]")
    if valid_mask is None:
        valid_mask = torch.ones((batch, length), dtype=torch.bool, device=states.device)
    elif valid_mask.shape != (batch, length):
        raise ValueError("valid_mask must have shape [B,T]")
    else:
        valid_mask = valid_mask.to(device=states.device, dtype=torch.bool)
    action_valid = valid_mask & (action_targets != ignore_index)
    if (((action_targets < 0) | (action_targets >= actions.shape[2])) & action_valid).any():
        raise ValueError("action target ID is outside the configured action vocabulary")
    safe_actions = action_targets.masked_fill(~action_valid, ignore_index)
    state_valid = valid_mask[:, None, :].to(states.dtype)
    targets = state_targets.to(device=states.device, dtype=states.dtype)
    positive_weight = states.new_full((classes, 1), pos_weight)
    mosp_loss = states.sum() * 0
    for logits in states:
        values = F.binary_cross_entropy_with_logits(logits, targets,
                                                    pos_weight=positive_weight, reduction="none")
        mosp_loss = mosp_loss + _masked_mean(values, state_valid)
    sai_loss = states.sum() * 0
    if sai_logits is not None:
        if not isinstance(sai_logits, Tensor) or sai_logits.shape != targets.shape:
            raise ValueError("sai_logits must match state_targets [B,C,T]")
        values = F.binary_cross_entropy_with_logits(sai_logits, targets,
                                                    pos_weight=positive_weight, reduction="none")
        sai_loss = _masked_mean(values, state_valid)
    ce_loss = actions.sum() * 0
    smoothing_loss = actions.sum() * 0
    for logits in actions:
        ce = F.cross_entropy(logits, safe_actions, ignore_index=ignore_index, reduction="none")
        ce_loss = ce_loss + _masked_mean(ce, action_valid.to(ce.dtype))
        if length > 1:
            log_probs = logits.log_softmax(dim=1)
            delta = log_probs[:, :, 1:] - log_probs[:, :, :-1].detach()
            smooth = delta.square().clamp(max=16)
            pairs = (action_valid[:, 1:] & action_valid[:, :-1])[:, None, :]
            smoothing_loss = smoothing_loss + smoothing_weight * _masked_mean(smooth, pairs.to(smooth.dtype))
    tas_loss = ce_loss + smoothing_loss
    return {"total": mosp_loss + sai_loss + tas_loss, "mosp": mosp_loss,
            "sai": sai_loss, "tas": tas_loss, "tas_ce": ce_loss,
            "tas_smoothing": smoothing_loss}
