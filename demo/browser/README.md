# Browser bench

Educational only. Not a medical device.

From the repo root:

```bash
./start.sh
```

`curl` and `python3` are enough. The script does not install PyTorch. It downloads the eight ONNX files published with the `edu-bench-v1` release when they are not already in `models/`, then downloads ONNX Runtime Web (`1.30.0`, WASM only) and `dicom-parser` into `vendor/` (gitignored). It serves this folder on `127.0.0.1:8080` and opens it. `PORT` overrides the port. Ctrl+C stops the server.

The page keeps the image in the tab. Drop a PNG, JPG, or uncompressed little-endian DICOM. Compressed DICOM is refused.

## What the page computes

Each graph returns **logits**. The page applies sigmoid, then `op_norm`, so 0.5 is that head's operating point and not a probability. Heads with an empty label or a null threshold were not trained and are ignored. The list is the mean across trained heads, plus how many sit at or above 0.5. Click a finding to paint where the All datasets DenseNet looked on the center crop it actually scored. That bright area is a class activation map, not a traced lesion. The other seven readers vote but do not draw.

Input is `1x1xHxW`, already scaled to about [-1024, 1024], center-cropped, then resized with PyTorch bilinear (`align_corners=False`). That matches `xrv.utils.load_image`: 8-bit images use the first channel and maxval 255; uncompressed DICOM uses `2**BitsStored-1`, and MONOCHROME1 is inverted. No VOI LUT and no rescale slope.

Networks:

- DenseNet-121 at 224: `all` (also writes a class activation map as `densenet121-res224-all-map.onnx`), NIH, PadChest, CheXpert, MIMIC-NB, MIMIC-CH, RSNA
- ResNet-50 at 512: `resnet50-res512-all`

Race, age, and sex models are not in this bench. WebAssembly runs single-threaded so the page does not need cross-origin isolation.

## Export instead of downloading

```bash
pip install torch torchvision onnx onnxruntime
python scripts/export_onnx.py --all --out-dir demo/browser/models
```

The ONNX files are not committed. `./start.sh` will use whatever is already in `models/`.
