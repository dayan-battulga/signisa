"""kaggle1st port: inference reproduces hoyso48's published TFLite (features AND logits),
padding never reaches real frames, and the full recipe trains with AWP restoring weights."""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from signisa.kaggle1st import (  # noqa: E402
    LHAND, POINT_LANDMARKS, RHAND, Kaggle1stNet, MaskedBatchNorm, PointsDataset, RecipeConfig,
    features_of, flipped, padded, points_cache, train_kaggle1st)
from signisa.preprocess.kaggle import load_holistic  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TFLITE = ROOT / "data/models/kaggle1st/model.tflite"
SAMPLES = sorted((ROOT / "data/samples").glob("*.parquet"))


def published_model(interpreter, index: dict) -> Kaggle1stNet:
    """Ensemble member 'model/' (first of three) of the published TFLite, in our port.
    TFLite folds each BN into x * mul + sub: weight=mul, bias=sub, eps=0 reproduces it."""
    def w(name):
        return torch.from_numpy(interpreter.get_tensor(index[f"model/{name}"]).astype(np.float32))

    state = {"stem.weight": w("stem_conv/Tensordot/MatMul"),
             "top.weight": w("top_conv/Tensordot/MatMul"),
             "top.bias": w("top_conv/BiasAdd/ReadVariableOp"),
             "classifier.weight": w("classifier/MatMul"),
             "classifier.bias": w("classifier/BiasAdd/ReadVariableOp")}

    def put_bn(key, name):
        state[f"{key}.weight"], state[f"{key}.bias"] = (
            w(f"{name}/batchnorm/mul"), w(f"{name}/batchnorm/sub"))

    put_bn("stem_bn", "stem_bn")
    for i, b in enumerate([0, 1, 2, 4, 5, 6], start=1):  # their blocks 1-6 around the attention
        for ours, theirs in (("expand", "expand_conv"), ("project", "project_conv")):
            state[f"blocks.{b}.{ours}.weight"] = w(f"{i}_{theirs}/Tensordot/MatMul")
            state[f"blocks.{b}.{ours}.bias"] = w(f"{i}_{theirs}/BiasAdd/ReadVariableOp")
        depthwise = w(f"{i}_dwconv/{i}_dwconv_dwconv/depthwise")  # (1, 1, 17, 384)
        state[f"blocks.{b}.dwconv.weight"] = depthwise[0, 0].T.unsqueeze(1)
        put_bn(f"blocks.{b}.bn", f"{i}_bn")
        eca = "eca/conv1d" if i == 1 else f"eca_{i - 1}/conv1d_{i - 1}"
        state[f"blocks.{b}.eca.conv.weight"] = w(f"{eca}/Conv1D").reshape(1, 1, 5)
    for j, b in enumerate([3, 7]):
        put_bn(f"blocks.{b}.norm1", "batch_normalization" + (f"_{2 * j}" if j else ""))
        put_bn(f"blocks.{b}.norm2", f"batch_normalization_{2 * j + 1}")
        mhsa = "multi_head_self_attention" + ("_1" if j else "")
        state[f"blocks.{b}.qkv.weight"] = w(f"{mhsa}/dense{'_4' if j else ''}/Tensordot/MatMul")
        state[f"blocks.{b}.proj.weight"] = w(f"{mhsa}/dense_{1 + 4 * j}/Tensordot/MatMul")
        state[f"blocks.{b}.up.weight"] = w(f"dense_{2 + 4 * j}/Tensordot/MatMul")
        state[f"blocks.{b}.down.weight"] = w(f"dense_{3 + 4 * j}/Tensordot/MatMul")

    net = Kaggle1stNet().eval()
    assert set(state) == {k for k, _ in net.named_parameters()}  # every weight is theirs
    net.load_state_dict(state, strict=False)  # shape mismatches still raise
    for m in net.modules():
        if isinstance(m, MaskedBatchNorm):
            m.eps = 0.0
    return net


@pytest.mark.skipif(not TFLITE.exists() or not SAMPLES, reason="published TFLite or samples absent")
def test_port_reproduces_published_model():
    litert = pytest.importorskip("ai_edge_litert.interpreter")
    interpreter = litert.Interpreter(model_path=str(TFLITE), experimental_preserve_all_tensors=True)
    interpreter.allocate_tensors()
    index = {t["name"]: t["index"] for t in interpreter.get_tensor_details()}
    net = published_model(interpreter, index)
    for path in SAMPLES[:5]:
        raw = load_holistic(path)
        interpreter.resize_tensor_input(index["serving_default_inputs:0"], raw.shape)
        interpreter.allocate_tensors()
        interpreter.set_tensor(index["serving_default_inputs:0"], raw)
        interpreter.invoke()
        # the TFLite skips the training-time empty-frame filter, so compare unfiltered
        ours = features_of(raw[:, POINT_LANDMARKS, :2])
        theirs = interpreter.get_tensor(index["preprocess/SelectV2_5"])[0]
        np.testing.assert_allclose(ours, theirs, atol=1e-4, err_msg=path.name)
        x = torch.from_numpy(ours)[None]
        with torch.no_grad():
            logits = net(x, torch.ones(x.shape[:2], dtype=torch.bool))[0].numpy()
        theirs = interpreter.get_tensor(index["model/classifier/MatMul;model/classifier/BiasAdd"])[0]
        np.testing.assert_allclose(logits, theirs, rtol=1e-3, atol=1e-3, err_msg=path.name)
        assert logits.argmax() == theirs.argmax()


def test_flip_swaps_hands_and_undoes_itself():
    x = np.random.default_rng(0).random((5, len(POINT_LANDMARKS), 2)).astype(np.float32)
    np.testing.assert_allclose(flipped(flipped(x)), x, atol=1e-6)
    left = [POINT_LANDMARKS.index(i) for i in LHAND]
    right = [POINT_LANDMARKS.index(i) for i in RHAND]
    np.testing.assert_allclose(flipped(x)[:, left], np.stack([1 - x[:, right, 0], x[:, right, 1]], -1))


def test_padding_never_reaches_real_frames():
    torch.manual_seed(0)
    net = Kaggle1stNet().eval()
    assert 1.8e6 < sum(p.numel() for p in net.parameters()) < 1.9e6  # writeup: ~1.85M
    short, long = torch.randn(7, 708), torch.randn(30, 708)
    x, mask, _ = padded([(short, 0), (long, 1)])
    with torch.no_grad():
        alone = net(short[None], torch.ones(1, 7, dtype=torch.bool))[0]
        torch.testing.assert_close(net(x, mask)[0], alone, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(len(SAMPLES) < 30, reason="needs the 30 local samples")
def test_recipe_trains_and_awp_restores_weights():
    cache = points_cache(SAMPLES, np.arange(len(SAMPLES)) % 250, workers=2)
    train_ds = PointsDataset(cache, np.arange(20), augment=True)
    val_ds = PointsDataset(cache, np.arange(20, 30), augment=False)
    # lr 0 isolates AWP: any perturbation left un-subtracted would show up as weight drift
    cfg = RecipeConfig(epochs=2, batch_size=8, micro_batch=4, lr=0.0, awp_start_epoch=0,
                       late_dropout_start_epoch=0, num_workers=0)
    model, history = train_kaggle1st(cfg, train_ds, val_ds, "cpu")
    assert len(history["val_loss"]) == 2 and np.isfinite(history["train_loss"]).all()
    torch.manual_seed(cfg.seed)
    initial = Kaggle1stNet()
    drift = max((a - b).abs().max().item()
                for a, b in zip(model.parameters(), initial.parameters()))
    assert drift < 1e-5
