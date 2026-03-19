# LoRA Vision-Language Model Adaptation

Parameter-efficient fine-tuning of vision-language detectors using LoRA on the first 10,000 images from BDD100K training set.

## Performance Results

Evaluation metric: **mAP@300 (IoU 0.50:0.95)** - mean Average Precision with 300 detection proposals, averaged across IoU thresholds from 0.50 to 0.95 (COCO-style evaluation).

Evaluated on **BDD100K validation set**.

| Model | Baseline | After LoRA | Improvement |
|-------|----------|------------|-------------|
| OWLv2-base-patch16 | 20.63% | **27.81%** | **+7.18%** |
| OmDet-Turbo-Swin | 17.14% | **21.64%** | **+4.50%** |
| Grounding DINO-tiny | 20.89% | **23.60%** | **+2.71%** |

**Key findings:**
- OWLv2 demonstrates strongest improvement (+7.18% mAP)
- All models show measurable performance gains on BDD100K
- Training losses decreased consistently across all models (see loss curves)

## Training Configuration

- **Training data:** First 10,000 images from BDD100K training set
- **Evaluation data:** BDD100K validation set
- **Epochs:** 50
- **Method:** LoRA with frozen text encoders
- **Loss:** CIoU for bounding box regression + classification loss
- **Evaluation metric:** mAP@300, IoU thresholds 0.50:0.95

## Training Curves

**OWLv2** ([google/owlv2-base-patch16-ensemble](https://huggingface.co/google/owlv2-base-patch16-ensemble))
![OWLv2 Training](OWLv2_LoRA_checkpoints_text_frozen_b16_ciou_training_losses.png)

**OmDet-Turbo** ([omlab/omdet-turbo-swin-tiny-hf](https://huggingface.co/omlab/omdet-turbo-swin-tiny-hf))
![OmDet Training](Omdet_LoRA_checkpoints_ciou_50epochs.png)

**Grounding DINO** ([IDEA-Research/grounding-dino-tiny](https://huggingface.co/IDEA-Research/grounding-dino-tiny))
![GDINO Training](GroundingDINO_checkpoints_ciou_textfreeze_50epochs.png)
