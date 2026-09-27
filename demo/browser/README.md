# Browser export

Educational only. Not a medical device.

`scripts/export_onnx.py --all` writes one ONNX file per core classifier into this folder, plus `registry.json`:

- DenseNet-121 at 224: `all`, NIH, PadChest, CheXpert, MIMIC-NB, MIMIC-CH, RSNA
- ResNet-50 at 512: `resnet50-res512-all`

Each graph returns **logits**. The browser applies sigmoid, then `op_norm`, so 0.5 is that head's operating point and not a probability. Heads with a null threshold were not trained and must be ignored.

Input is `1x1xHxW`, already scaled to about [-1024, 1024], center-cropped, then resized. Match `xrv.utils.load_image` (8-bit images use the first channel and maxval 255; uncompressed DICOM uses `2**BitsStored-1`, and MONOCHROME1 is inverted).

Race, age, and sex models are not exported. They are not part of a read.

```bash
pip install torch torchvision onnx onnxruntime
python scripts/export_onnx.py --all --out-dir demo/browser/models
```

The ONNX files are generated locally and are not committed.
