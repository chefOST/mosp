import math

import pytest
import torch

from mosp.losses import joint_loss


def zero_outputs(length=1, stages=2, sai=True):
    return {
        "state_logits": torch.zeros(stages, 1, 2, length, requires_grad=True),
        "action_logits": torch.zeros(stages, 1, 3, length, requires_grad=True),
        "sai_logits": torch.zeros(1, 2, length, requires_grad=True) if sai else None,
    }


def test_joint_weighted_bce_and_action_ce_match_hand_calculation():
    outputs = zero_outputs()
    targets = torch.tensor([[[1.0], [0.0]]])
    losses = joint_loss(outputs, targets, torch.zeros(1, 1, dtype=torch.long))
    assert losses["mosp"].item() == pytest.approx(6 * math.log(2))
    assert losses["sai"].item() == pytest.approx(3 * math.log(2))
    assert losses["tas_ce"].item() == pytest.approx(2 * math.log(3))
    assert losses["tas_smoothing"].item() == 0
    assert losses["total"].item() == pytest.approx(9 * math.log(2) + 2 * math.log(3))
    losses["total"].backward()
    assert all(torch.isfinite(logits.grad).all() for logits in outputs.values())


def test_smoothing_clips_large_changes_at_upstream_threshold():
    outputs = {
        "state_logits": torch.zeros(1, 1, 1, 2),
        "action_logits": torch.tensor([[[[100.0, -100.0], [-100.0, 100.0]]]], requires_grad=True),
        "sai_logits": None,
    }
    losses = joint_loss(outputs, torch.zeros(1, 1, 2), torch.tensor([[0, 1]]))
    assert losses["tas_smoothing"].item() == pytest.approx(0.15 * 16)
    assert torch.isfinite(losses["total"])


def test_smoothing_detaches_previous_frame():
    actions = torch.tensor([[[[0.5, 0.1, -0.3], [-0.5, -0.1, 0.3]]]], requires_grad=True)
    outputs = {"state_logits": torch.zeros(1, 1, 1, 3),
               "action_logits": actions, "sai_logits": None}
    losses = joint_loss(outputs, torch.zeros(1, 1, 3), torch.zeros(1, 3, dtype=torch.long))
    losses["tas_smoothing"].backward()
    torch.testing.assert_close(actions.grad[:, :, :, 0], torch.zeros_like(actions.grad[:, :, :, 0]))
    assert actions.grad[:, :, :, 1:].abs().sum() > 0


def test_padding_does_not_change_loss_or_gradients():
    outputs = zero_outputs(length=2)
    state_targets = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    action_targets = torch.tensor([[0, 123456]])
    valid = torch.tensor([[True, False]])
    losses = joint_loss(outputs, state_targets, action_targets, valid_mask=valid)
    reference = joint_loss(zero_outputs(), state_targets[:, :, :1], action_targets[:, :1])
    for name in losses:
        torch.testing.assert_close(losses[name], reference[name])
    losses["total"].backward()
    for logits in outputs.values():
        torch.testing.assert_close(logits.grad[..., 1], torch.zeros_like(logits.grad[..., 1]))


def test_all_masked_batch_retains_zero_loss_graph():
    outputs = zero_outputs(length=1)
    losses = joint_loss(outputs, torch.zeros(1, 2, 1), torch.tensor([[-100]]),
                        valid_mask=torch.zeros(1, 1, dtype=torch.bool))
    assert all(value.item() == 0 for value in losses.values())
    losses["total"].backward()
    assert all(torch.isfinite(logits.grad).all() for logits in outputs.values())


def test_unknown_action_labels_do_not_drop_state_supervision():
    losses = joint_loss(zero_outputs(), torch.ones(1, 2, 1), torch.tensor([[-100]]))
    assert losses["tas"].item() == 0
    assert losses["mosp"].item() > 0
    assert losses["sai"].item() > 0


def test_without_sai_omits_only_sai_term():
    losses = joint_loss(zero_outputs(sai=False), torch.zeros(1, 2, 1), torch.tensor([[0]]))
    assert losses["sai"].item() == 0
    torch.testing.assert_close(losses["total"], losses["mosp"] + losses["tas"])


@pytest.mark.parametrize("pos_weight,smoothing_weight", [(0, 0.15), (float("nan"), 0.15), (5, -1)])
def test_invalid_loss_weights_rejected(pos_weight, smoothing_weight):
    with pytest.raises(ValueError):
        joint_loss(zero_outputs(), torch.zeros(1, 2, 1), torch.tensor([[0]]),
                   pos_weight=pos_weight, smoothing_weight=smoothing_weight)


def test_out_of_range_action_and_malformed_state_targets_rejected():
    with pytest.raises(ValueError, match="vocabulary"):
        joint_loss(zero_outputs(), torch.zeros(1, 2, 1), torch.tensor([[3]]))
    with pytest.raises(ValueError, match="state_targets"):
        joint_loss(zero_outputs(), torch.full((1, 2, 1), 2.0), torch.tensor([[0]]))
