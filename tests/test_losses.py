import math

import pytest
import torch
from torch.nn import functional as F

from mosp.losses import joint_loss


def zero_outputs(length=2, stages=2, sai=True):
    return {
        "state_logits": torch.zeros(stages, 1, 2, length, requires_grad=True),
        "action_logits": torch.zeros(stages, 1, 3, length, requires_grad=True),
        "sai_logits": torch.zeros(1, 2, length, requires_grad=True) if sai else None,
    }


def test_joint_weighted_bce_and_action_ce_match_hand_calculation():
    outputs = zero_outputs()
    targets = torch.tensor([[[1.0, 1.0], [0.0, 0.0]]])
    losses = joint_loss(outputs, targets, torch.zeros(1, 2, dtype=torch.long))
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


def test_padding_excludes_supervised_targets_and_padded_output_gradients():
    outputs = zero_outputs(length=2)
    state_targets = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    action_targets = torch.tensor([[0, 123456]])
    valid = torch.tensor([[True, False]])
    losses = joint_loss(outputs, state_targets, action_targets, valid_mask=valid)
    assert losses["mosp"].item() == pytest.approx(6 * math.log(2))
    assert losses["sai"].item() == pytest.approx(3 * math.log(2))
    assert losses["tas_ce"].item() == pytest.approx(2 * math.log(3))
    assert losses["tas_smoothing"].item() == 0
    losses["total"].backward()
    for logits in outputs.values():
        torch.testing.assert_close(logits.grad[..., 1], torch.zeros_like(logits.grad[..., 1]))


def test_all_masked_batch_retains_zero_loss_graph():
    outputs = zero_outputs()
    losses = joint_loss(outputs, torch.zeros(1, 2, 2), torch.tensor([[-100, -100]]),
                        valid_mask=torch.zeros(1, 2, dtype=torch.bool))
    assert all(value.item() == 0 for value in losses.values())
    losses["total"].backward()
    assert all(torch.isfinite(logits.grad).all() for logits in outputs.values())


def test_unknown_action_labels_do_not_drop_state_supervision():
    losses = joint_loss(zero_outputs(), torch.ones(1, 2, 2), torch.tensor([[-100, -100]]))
    assert losses["tas"].item() == 0
    assert losses["mosp"].item() > 0
    assert losses["sai"].item() > 0


def test_without_sai_omits_only_sai_term():
    losses = joint_loss(zero_outputs(sai=False), torch.zeros(1, 2, 2), torch.tensor([[0, 0]]))
    assert losses["sai"].item() == 0
    torch.testing.assert_close(losses["total"], losses["mosp"] + losses["tas"])


@pytest.mark.parametrize("pos_weight,smoothing_weight", [(0, 0.15), (float("nan"), 0.15), (5, -1)])
def test_invalid_loss_weights_rejected(pos_weight, smoothing_weight):
    with pytest.raises(ValueError):
        joint_loss(zero_outputs(), torch.zeros(1, 2, 2), torch.tensor([[0, 0]]),
                   pos_weight=pos_weight, smoothing_weight=smoothing_weight)


def test_out_of_range_action_and_malformed_state_targets_rejected():
    with pytest.raises(ValueError, match="vocabulary"):
        joint_loss(zero_outputs(), torch.zeros(1, 2, 2), torch.tensor([[3, 0]]))
    with pytest.raises(ValueError, match="state_targets"):
        joint_loss(zero_outputs(), torch.full((1, 2, 2), 2.0), torch.tensor([[0, 0]]))


def upstream_smoothing(stage_logits, mask):
    # Reference expression from sj-li/MS-TCN2 model.py, pinned commit
    # f423a9e65f4ccb1cd7322eb9f94946a19e787993. mask is [B,A,T].
    terms = []
    for p in stage_logits:
        terms.append(0.15 * torch.mean(torch.clamp(
            F.mse_loss(F.log_softmax(p[:, :, 1:], dim=1),
                       F.log_softmax(p.detach()[:, :, :-1], dim=1), reduction="none"),
            min=0, max=16) * mask[:, :, 1:]))
    return torch.stack(terms).sum()


@pytest.mark.parametrize("valid", [
    [[True, True, True, True], [True, True, True, True]],
    [[True, True, False, False], [True, True, True, False]],
    [[False, True, False, True], [True, False, True, True]],
])
def test_smoothing_values_and_gradients_match_pinned_upstream(valid):
    generator = torch.Generator().manual_seed(41)
    actions = torch.randn(3, 2, 3, 4, generator=generator, dtype=torch.float64, requires_grad=True)
    validity = torch.tensor(valid)
    outputs = {"state_logits": torch.zeros(3, 2, 1, 4, dtype=torch.float64),
               "action_logits": actions, "sai_logits": None}
    labels = torch.tensor([[0, -100, 2, -100], [-100, 1, -100, 0]])
    actual = joint_loss(outputs, torch.zeros(2, 1, 4), labels, valid_mask=validity)["tas_smoothing"]
    reference = upstream_smoothing(actions, validity[:, None, :].expand(-1, 3, -1))
    torch.testing.assert_close(actual, reference)
    actual_gradient = torch.autograd.grad(actual, actions, retain_graph=True)[0]
    reference_gradient = torch.autograd.grad(reference, actions)[0]
    torch.testing.assert_close(actual_gradient, reference_gradient)


def test_unknown_labels_do_not_remove_real_frame_smoothing():
    outputs = {"state_logits": torch.zeros(1, 1, 1, 3),
               "action_logits": torch.tensor([[[[0.5, -0.5, 0.5], [-0.5, 0.5, -0.5]]]]),
               "sai_logits": None}
    labelled = joint_loss(outputs, torch.zeros(1, 1, 3), torch.tensor([[0, 1, 0]]))
    unknown = joint_loss(outputs, torch.zeros(1, 1, 3), torch.full((1, 3), -100))
    assert unknown["tas_ce"].item() == 0
    assert unknown["tas_smoothing"].item() > 0
    torch.testing.assert_close(unknown["tas_smoothing"], labelled["tas_smoothing"])


def test_smoothing_padding_zeros_remain_in_global_denominator():
    outputs = {"state_logits": torch.zeros(1, 1, 1, 3),
               "action_logits": torch.tensor([[[[0.0, 0.5, 100.0], [0.0, -0.5, -100.0]]]]),
               "sai_logits": None}
    actual = joint_loss(outputs, torch.zeros(1, 1, 3), torch.tensor([[0, 1, -100]]),
                        valid_mask=torch.tensor([[True, True, False]]))["tas_smoothing"]
    cropped = {name: value[..., :2] if value is not None else None for name, value in outputs.items()}
    short = joint_loss(cropped, torch.zeros(1, 1, 2), torch.tensor([[0, 1]]))["tas_smoothing"]
    torch.testing.assert_close(actual, short / 2)


def test_single_frame_preserves_upstream_empty_mean_behavior():
    outputs = zero_outputs(length=1)
    actual = joint_loss(outputs, torch.zeros(1, 2, 1), torch.zeros(1, 1, dtype=torch.long))["tas_smoothing"]
    reference = upstream_smoothing(outputs["action_logits"], torch.ones(1, 3, 1))
    assert torch.isnan(actual) and torch.isnan(reference)
