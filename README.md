# SPFD: synthetic mechanism demo

Executable companion demo for **Structure-Protected Frequency Debiasing for Pipeline Detection in Geographically Held-Out Side-Scan Sonar Sections**.

**Synthetic data only.** This CPU example demonstrates selected SPFD mechanisms. It is separate from the laboratory implementation, trained weights, real sonar experiments, and reported paper results. Demo outputs are not publication evidence.

## Quick start

Use Python 3.12. No pretrained weights, real sonar data, or GPU are required.

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python -B -X utf8 run_demo.py --output outputs/my_synthetic_run
python -B -X utf8 -m unittest discover -s tests -v
```

The tested dependency versions are recorded in `requirements-tested.txt`. The demo uses CPU execution; a CPU-only PyTorch installation can also be used. The default run performs 12 warm-start and 24 joint-training steps. Outputs include synthetic diagnostics, predictions, and a small locally generated synthetic-model checkpoint. No laboratory checkpoint is distributed.

## Data and mechanisms

`data/SYNTHETIC_sample.npz` contains 24 artificial 24-by-24 images. It is generated deterministically with seed 1729 and regenerated automatically if missing. To generate it explicitly, run `python data/generate.py`. See `data/DATA_CARD.md` for fields and split definitions.

The source implements detector-guided spectral selection, complementary amplitude mixing, target-neighborhood restoration, GT-identity-associated two-view losses, and shared regional class posteriors. `METHOD_MAP.md` documents the correspondence and simplifications. The tiny three-query detector, miniature training schedule, single-level features and simplified geometry do not reproduce the full DEIM-D-FINE-S pipeline. The demo does not compute paper AP, cross-frequency generalization, spatial confidence intervals, or measured GPU costs.

## Real study data and restricted implementation

The study uses [SubPipe v3.0.1](https://zenodo.org/records/12666132), distributed by its original providers under CC BY 4.0. This repository contains generated synthetic data rather than a redistribution of the real sonar dataset or study-specific partition archive.

The full experimental implementation and trained weights remain subject to laboratory-controlled access. After publication, requests may be sent to zhaoboyu@tyut.edu.cn, subject to laboratory approval. Public availability of this demonstration does not imply public availability of those materials.

## Validation

The publication copy is checked with the 13 included unit tests and a complete CPU demo run. See `VALIDATION.md` for the verified commands and scope.
