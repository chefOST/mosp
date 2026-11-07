import inspect

import pytest
import torch

from mosp.model import MOSPConfig, MOSPSystem, StateActionInteraction


torch.set_num_threads(1)


def small_config(**overrides):
    values = dict(num_state_classes=2, num_actions=2, beta=0.3,
                  feature_dim=3, num_layers=2, num_refinements=1,
                  num_f_maps=4, dropout=0.0)
    values.update(overrides)
    return MOSPConfig(**values)


def make_model(**overrides):
    return MOSPSystem(small_config(**overrides), torch.eye(2), torch.eye(2))


def test_published_architecture_defaults_and_explicit_beta():
    with pytest.raises(TypeError):
        MOSPConfig(num_state_classes=2, num_actions=2)
    config = MOSPConfig(num_state_classes=2, num_actions=2, beta=0.5)
    model = MOSPSystem(config, torch.eye(2), torch.eye(2))
    assert config.feature_dim == 2048
    assert len(model.state_model.Rs) == len(model.action_model.Rs) == 3
    assert len(model.state_model.PG.conv_dilated_1) == 10
    assert model.state_model.PG.conv_1x1_in.out_channels == 64
    assert [layer.dilation[0] for layer in model.state_model.PG.conv_dilated_1] == [2 ** i for i in reversed(range(10))]
    assert [layer.dilation[0] for layer in model.state_model.PG.conv_dilated_2] == [2 ** i for i in range(10)]
    assert all(len(stage.layers) == 10 for stage in model.state_model.Rs)
    assert model.state_model.Rs[0].conv_out.weight.data_ptr() != model.state_model.Rs[1].conv_out.weight.data_ptr()


@pytest.mark.parametrize("length", [1, 7])
def test_outputs_and_inference_do_not_require_annotation_actions(length):
    model = make_model().eval()
    assert tuple(inspect.signature(model.forward).parameters) == ("features",)
    outputs = model(torch.randn(2, 3, length))
    assert outputs["state_logits"].shape == (2, 2, 2, length)
    assert outputs["action_logits"].shape == (2, 2, 2, length)
    assert outputs["sai_logits"].shape == (2, 2, length)
    expected = 0.7 * outputs["state_logits"][-1].sigmoid() + 0.3 * outputs["sai_logits"].sigmoid()
    torch.testing.assert_close(outputs["probabilities"], expected)
    assert torch.isfinite(outputs["probabilities"]).all()
    assert ((outputs["probabilities"] >= 0) & (outputs["probabilities"] <= 1)).all()


def test_state_refinement_sigmoid_and_action_refinement_softmax():
    model = make_model().eval()
    inputs = {}
    state_hook = model.state_model.Rs[0].register_forward_pre_hook(
        lambda _module, args: inputs.__setitem__("state", args[0].detach().clone())
    )
    action_hook = model.action_model.Rs[0].register_forward_pre_hook(
        lambda _module, args: inputs.__setitem__("action", args[0].detach().clone())
    )
    outputs = model(torch.randn(1, 3, 5))
    state_hook.remove()
    action_hook.remove()
    torch.testing.assert_close(inputs["state"], outputs["state_logits"][0].sigmoid())
    torch.testing.assert_close(inputs["action"], outputs["action_logits"][0].softmax(dim=1))


def test_midpoint_intervals_confidence_and_endpoints():
    sai = StateActionInteraction(small_config(), torch.eye(2), torch.eye(2))
    probs = torch.tensor([[[0.9, 0.8, 0.7, 0.4, 0.3, 0.2],
                           [0.1, 0.2, 0.3, 0.6, 0.7, 0.8]]], requires_grad=True)
    evidence = sai.action_evidence(probs)
    # Segments [0,3) and [3,6), midpoint boundaries 1 and 4.
    # Class-major channels: after0,before0,after1,before1.
    expected = torch.tensor([[[0, 0.75, 0.75, 0.75, 0, 0],
                              [0.85, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0.75, 0.75],
                              [0, 0.65, 0.65, 0.65, 0, 0]]])
    torch.testing.assert_close(evidence, expected)
    evidence.sum().backward()
    assert probs.grad is not None and probs.grad.abs().sum() > 0


def test_no_action_background_and_single_frame_are_finite():
    # An explicit background action may have no observed state correlations.
    correlations = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    sai = StateActionInteraction(small_config(), correlations, correlations)
    with torch.no_grad():
        sai.fusion.bias.zero_()
    logits = torch.tensor([[[-10.0], [10.0]]], requires_grad=True)
    output = sai(logits)
    torch.testing.assert_close(output, torch.zeros_like(output))
    output.sum().backward()
    assert torch.isfinite(logits.grad).all()


def test_sai_loss_can_train_action_branch_through_confidences():
    model = make_model().eval()
    with torch.no_grad():
        model.sai.fusion.weight.fill_(1.0)
    outputs = model(torch.randn(1, 3, 9))
    outputs["sai_logits"].sum().backward()
    gradients = [parameter.grad for parameter in model.action_model.parameters()]
    assert any(gradient is not None and gradient.abs().sum() > 0 for gradient in gradients)
    assert all(parameter.grad is None for parameter in model.state_model.parameters())


@pytest.mark.parametrize("invalid", [None, torch.ones(2, 2),
                                    torch.tensor([[float("nan"), 0], [0, 1]]),
                                    torch.tensor([[-1.0, 2], [0, 1]])])
def test_requires_valid_empirical_correlations(invalid):
    with pytest.raises(ValueError):
        MOSPSystem(small_config(), invalid, torch.eye(2))


def test_empty_time_rejected_and_without_sai_is_explicit():
    with pytest.raises(ValueError, match="at least one"):
        make_model()(torch.empty(1, 3, 0))
    with pytest.raises(ValueError, match="beta=0"):
        small_config(use_sai=False)
    model = MOSPSystem(small_config(use_sai=False, beta=0))
    outputs = model(torch.randn(1, 3, 1))
    assert outputs["sai_logits"] is None
    torch.testing.assert_close(outputs["probabilities"], outputs["state_logits"][-1].sigmoid())


def test_classwise_and_dense_fusion_are_recorded_and_constructed():
    assert make_model().sai.fusion.groups == 2
    assert make_model(sai_fusion="dense").sai.fusion.groups == 1
    with pytest.raises(ValueError):
        small_config(sai_fusion="unknown")


def test_correlations_survive_checkpoint_state_dict():
    original = make_model().eval()
    restored = make_model().eval()
    restored.load_state_dict(original.state_dict())
    features = torch.randn(1, 3, 6)
    torch.testing.assert_close(original(features)["probabilities"], restored(features)["probabilities"])


@pytest.mark.parametrize("beta", [0.0, 1.0])
def test_beta_extremes_select_the_requested_branch(beta):
    output = make_model(beta=beta).eval()(torch.randn(1, 3, 5))
    expected = output["state_logits"][-1].sigmoid() if beta == 0 else output["sai_logits"].sigmoid()
    torch.testing.assert_close(output["probabilities"], expected)
