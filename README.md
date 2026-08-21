**On the Transferability of Reliability-Guided Logit Purification in Training-Free CLIP Semantic Segmentation**

RGLP studies reliability-guided logit purification and its transferability for training-free
CLIP-based semantic segmentation. The refinement runs entirely at test time: it estimates
per-token reliability from the frozen backbone, rewrites the logits of unreliable regions using
high-confidence local evidence, and then applies a domain-transform structural refinement. No
additional trainable parameters are introduced, and one source-selected profile is applied to
every dataset without per-dataset retuning.

## Installation

```bash
conda create -n rglp python=3.10 -y
conda activate rglp
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

## Dataset Layout

```
data/
├── VOC2012/
│   ├── JPEGImages/
│   ├── SegmentationClass/
│   └── ImageSets/
│
├── ADEChallengeData2016/
│   ├── images/
│   └── annotations/
│
└── COCOStuff27/
    ├── images/
    └── annotations/
```

`COCOStuff27/` uses the 27-class COCO-Stuff protocol (the merged stuff/thing categories),
not the 171-class protocol.

To use a different location, edit `DATASET.DATAROOT` in the corresponding config file.

## Checkpoints

Pretrained checkpoints will be released separately. **Coming soon.**

Expected layout once downloaded:

```
weights/
├── voc/best_weight.pth
├── ade20k/best_weight.pth
└── coco_stuff27/best_weight.pth
```

The path is read from `LOAD_PATH` in each config, or can be overridden with
`--load_path` when using `tools/test_tta.py`.

## Evaluation

Baseline (no refinement):

```bash
python tools/test.py --cfg config/voc_baseline.yaml           --model_module model.model
python tools/test.py --cfg config/ade20k_baseline.yaml        --model_module model.model
python tools/test.py --cfg config/coco_stuff27_baseline.yaml  --model_module model.model
```

RGLP refinement:

```bash
python tools/test.py --cfg config/voc_rglp.yaml           --model_module model.model_sfp_dtlr
python tools/test.py --cfg config/ade20k_rglp.yaml        --model_module model.model_sfp_dtlr
python tools/test.py --cfg config/coco_stuff27_rglp.yaml  --model_module model.model_sfp_dtlr
```

`tools/test_tta.py` is the evaluation entry point used for the reported numbers. It accepts the
same arguments plus multi-view and ablation options:

```bash
python tools/test_tta.py --cfg config/voc_rglp.yaml --model_module model.model_sfp_dtlr --scales 1.0
python tools/test_tta.py --cfg config/voc_rglp.yaml --model_module model.model_sfp_dtlr --scales 1.0 --flip
```

Useful flags: `--limit N` (evaluate the first N images only), `--save_dir DIR` (write predictions
to a fresh directory), `--load_path PATH` (override the checkpoint), `--sfp_disable dtlr,proxy,cpsfp`
(component ablation). Both scripts print the per-class IoU and the mIoU at the end of the run.

## Method

```
RGLP
├── reliability estimation      (per-token reliability from the frozen backbone)
├── selective logit purification (unreliable tokens rewritten from high-confidence local evidence)
└── DTLR refinement             (domain-transform structural refinement of the logits)
```

The refinement is training-free and adds no parameters to the segmentation model. The operating
point in `config/*_rglp.yaml` is identical across all datasets in this repository.

## Results

Results will be updated upon publication.

## Repository Structure

```
RGLP/
├── config/           # dataset configs (baseline and RGLP) + config loader
├── model/            # baseline model and the RGLP refinement module
├── text/             # CLIP text embeddings used at inference
├── tools/            # evaluation entry points
├── utils/            # preprocessing, metric, runtime assets
├── README.md
├── requirements.txt
└── .gitignore
```



## Acknowledgements

This repository builds on the ReCLIP / ReCLIP++ codebase for CLIP bias rectification
(*Learn to Rectify the Bias of CLIP for Unsupervised Semantic Segmentation*, CVPR 2024) and on
the OpenAI CLIP reference implementation. We thank the authors of both for releasing their code.
