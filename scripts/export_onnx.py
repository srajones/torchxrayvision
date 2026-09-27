#!/usr/bin/env python3
"""Export TorchXRayVision classifiers to ONNX for the browser bench.

Each graph returns raw logits. The page applies sigmoid and ``op_norm`` so a
score of 0.5 still means "at this head's operating point", matching
``DenseNet.forward`` / ``ResNet.forward``.

Not for clinical use.

Examples:
    python scripts/export_onnx.py
    python scripts/export_onnx.py --all --out-dir demo/browser/models
"""

import argparse
import json
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torchxrayvision as xrv


CLASSIFIERS = [
    ("densenet121-res224-all", "All cohorts", "DenseNet-121", "primary"),
    ("densenet121-res224-nih", "NIH", "DenseNet-121", "cohort"),
    ("densenet121-res224-pc", "PadChest", "DenseNet-121", "cohort"),
    ("densenet121-res224-chex", "CheXpert", "DenseNet-121", "cohort"),
    ("densenet121-res224-mimic_nb", "MIMIC-NB", "DenseNet-121", "cohort"),
    ("densenet121-res224-mimic_ch", "MIMIC-CH", "DenseNet-121", "cohort"),
    ("densenet121-res224-rsna", "RSNA", "DenseNet-121", "cohort"),
    ("resnet50-res512-all", "ResNet 512", "ResNet-50", "cohort"),
]


class DenseNetLogits(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.features = model.features
        self.classifier = model.classifier

    def forward(self, x):
        features = self.features(x)
        features = F.relu(features)
        pooled = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
        return self.classifier(pooled)


class ResNetLogits(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.model

    def forward(self, x):
        return self.model(x)


def metadata_for(weights, title, family, role):
    spec = xrv.models.model_urls[weights]
    labels = list(spec["labels"])
    thresholds = []
    for value in spec["op_threshs"]:
        if value != value:
            thresholds.append(None)
        else:
            thresholds.append(float(value))
    resolution = int(spec["input_resolution"])
    return {
        "id": weights,
        "file": weights + ".onnx",
        "title": title,
        "family": family,
        "role": role,
        "resolution": resolution,
        "labels": labels,
        "op_threshs": thresholds,
        "description": spec.get("description", ""),
    }


def wrap(weights):
    if weights.startswith("densenet"):
        model = xrv.models.DenseNet(weights=weights)
        model.eval()
        return model, DenseNetLogits(model).eval()
    if weights.startswith("resnet"):
        model = xrv.models.ResNet(weights=weights)
        model.eval()
        return model, ResNetLogits(model).eval()
    raise SystemExit("Unsupported weights: " + weights)


def check(model, wrapper, onnx_path, resolution):
    import numpy as np
    import onnxruntime as ort

    torch.manual_seed(0)
    sample = torch.randn(1, 1, resolution, resolution)
    sample = sample / sample.abs().max() * 800

    with torch.no_grad():
        expected = wrapper(sample).numpy()

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    actual = session.run(None, {"input": sample.numpy()})[0]
    if not np.allclose(expected, actual, rtol=1e-3, atol=1e-3):
        raise SystemExit(
            "ONNX logits do not match PyTorch for %s. max abs diff %s"
            % (onnx_path, np.max(np.abs(expected - actual)))
        )

    with torch.no_grad():
        full = model(sample)
        renorm = xrv.models.op_norm(torch.sigmoid(wrapper(sample)), model.op_threshs)
    if not torch.allclose(full, renorm, rtol=1e-4, atol=1e-4, equal_nan=True):
        raise SystemExit("op_norm contract drifted for " + onnx_path)
    print("parity ok", os.path.basename(onnx_path))


def export_one(weights, title, family, role, out_dir):
    model, wrapper = wrap(weights)
    spec = metadata_for(weights, title, family, role)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, spec["file"])
    resolution = spec["resolution"]
    dummy = torch.zeros(1, 1, resolution, resolution)
    torch.onnx.export(
        wrapper,
        dummy,
        out_path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        dynamo=False,
    )
    check(model, wrapper, out_path, resolution)
    return spec


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default_dir = os.path.join(here, "..", "demo", "browser", "models")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="densenet121-res224-all")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--out-dir", default=default_dir)
    args = parser.parse_args()
    out_dir = os.path.abspath(args.out_dir)

    chosen = CLASSIFIERS if args.all else [row for row in CLASSIFIERS if row[0] == args.weights]
    if not chosen:
        raise SystemExit("Unknown weights. Use --all or one of: " + ", ".join(r[0] for r in CLASSIFIERS))

    registry = []
    for weights, title, family, role in chosen:
        print("exporting", weights)
        registry.append(export_one(weights, title, family, role, out_dir))

    if args.all or len(chosen) > 1:
        path = os.path.join(out_dir, "registry.json")
        payload = {
            "not_for_clinical_use": True,
            "score_note": (
                "Each network emits logits. The browser applies sigmoid, then "
                "op_norm. A score of 0.5 means the sigmoid matched that head's "
                "operating threshold. It is not a probability of disease. "
                "Heads with a null threshold were not trained and are ignored."
            ),
            "citation": "https://arxiv.org/abs/2111.00595",
            "models": registry,
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        print("wrote", path)


if __name__ == "__main__":
    main()
