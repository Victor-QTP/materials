# Recent Project: Zero-Shot Auto-Labeling & Edge Deployment for Autonomous Driving Perception

**A training-free data engine that labels driving datasets with zero human annotation.**

- **~96% of human-label quality:** the engine's labels nearly match those of human annotators (BDD100K, evidence in [Section 1](#1-the-data-engine-training-free-auto-labeling))
- **814.7 FPS on an RTX 3090:** a real-time detector trained purely on the engine's labels, no human labels anywhere in the loop ([Section 2](#2-edge-deployment-distilled-yolov8-real-time-inference))
- **Works beyond driving, unchanged:** the same engine labeled biomedical microscopy imagery with zero tuning ([Section 3](#3-domain-transfer-zero-shot-generalization-to-biomedical-imagery))

**Quoc-Thang Phan (Victor)**, Computer Vision & Multimodal AI Engineer
[quocthang.phan.0430@gmail.com](mailto:quocthang.phan.0430@gmail.com) | [LinkedIn](https://www.linkedin.com/in/victor-quoc-thang-phan-60b932300)

---

## 1. The Data Engine: Training-Free Auto-Labeling

**Raw footage in, training-ready annotations out.** A training-free **ensemble learning** system fuses four open-vocabulary detectors (OWLv2, OmDet-Turbo, Grounding DINO, SAM3) into boxes, labels, instance masks, and captions. No training, no per-dataset tuning.

[![System Overview](https://cdn.jsdelivr.net/gh/Victor-QTP/materials@main/Data_engine/System_overview.png)](https://cdn.jsdelivr.net/gh/Victor-QTP/materials@main/Data_engine/System_overview.png)

**Does the output actually work?**

- **~96% of human-label quality, zero human labels:** the engine's annotations score within ~4% of a fully supervised [Cascade R-CNN (ConvNeXt-B)](https://github.com/SysCV/bdd100k-models/blob/main/det/README.md) trained on human-annotated ground truth (BDD100K-val, 34.31 mAP).
- **Beats the standard fusion method:** 34.31 vs. 32.70 mAP for Weighted Boxes Fusion on BDD100K-val.
- **Strong where labels are hardest to get:** on CODA2022 (rare road-hazard corner cases), the engine recovers far more of the hard objects than the strongest single model (62.77 vs. 44.55 mAR for SAM3) and improves precision (21.71 mAP, +2.10 over SAM3).

**How the pipeline works:**

1. **Detect:** the training-free ensemble fuses box proposals and open-vocabulary labels from all four detectors. This fused detection output alone is what the label-quality numbers above measure.
2. **Extend to masks:** [SAM3](SAM3/README.md), now as a promptable segmenter rather than a detector, turns the fused boxes into pixel-level instance masks
3. **Extend to captions:** [QwenVL](Object_captioning_Qwen2VL/README.md) adds a semantic caption to each detected object

**The whole pipeline runs on a single 24 GB GPU** through a [Gradio auto-labeling UI](https://cdn.jsdelivr.net/gh/Victor-QTP/materials@main/UI/UI_demo.png): 1-click annotation, ontology testing, and captioning. Full architecture: [Data Engine README](Data_engine/README.md) ([PDF overview](Data_engine/System_overview.pdf)).

| Detections (ensemble) & captions | Instance masks (SAM3) |
| :---: | :---: |
| [![Input (detection)](https://cdn.jsdelivr.net/gh/Victor-QTP/materials@main/SAM3/BDD100K_val_c415a08c-50060410.png)](https://cdn.jsdelivr.net/gh/Victor-QTP/materials@main/SAM3/BDD100K_val_c415a08c-50060410.png) | [![Output (SAM3 masks)](https://cdn.jsdelivr.net/gh/Victor-QTP/materials@main/SAM3/BDD100K_val_c415a08c-50060410_SAM3.png)](https://cdn.jsdelivr.net/gh/Victor-QTP/materials@main/SAM3/BDD100K_val_c415a08c-50060410_SAM3.png) |

---

## 2. Edge Deployment: Distilled YOLOv8 Real-Time Inference

A **YOLOv8-Nano** trained *entirely* on the engine's annotations, then optimized PyTorch → ONNX → TensorRT for edge hardware.

| Runtime | Precision | Hardware | Avg. Inference Time | Avg. FPS |
| :--- | :--- | :--- | :--- | :--- |
| **PyTorch (.pt)** | FP32 | RTX 3090 | 4.9 ms | **205.6 FPS** |
| **ONNX Runtime** | FP32 | RTX 3090 | 2.9 ms | **345.3 FPS** |
| **TensorRT** | FP16 | RTX 3090 | 1.2 ms | **814.7 FPS** |
| **Edge Target** | FP16 | Jetson Orin Nano 8GB | ~22-26 ms | **~38-45 FPS** (projected) |

**Real-time inference demos:** [TensorRT](https://www.youtube.com/watch?v=MUm59Mw1z0E) | [ONNX Runtime](https://www.youtube.com/watch?v=e6DNVmk0I_g) | [PyTorch](https://www.youtube.com/watch?v=Om5kYzBqwuw)

[![YOLOv8 Nano Distillation Demo](https://img.youtube.com/vi/MUm59Mw1z0E/0.jpg)](https://www.youtube.com/watch?v=MUm59Mw1z0E)

*Demo footage credit: [source video](https://www.youtube.com/watch?v=EXFlYUM5FgI)*

---

## 3. Domain Transfer: Zero-Shot Generalization to Biomedical Imagery

**The same engine, unchanged,** labeled noisy **phase-contrast microscopy** (stem cell cultures) with no manual annotation and no domain-specific tuning. A YOLOv8-Nano distilled from those labels runs at **820.8 FPS** (TensorRT FP16, RTX 3090, same optimization pipeline and Jetson target as above).

**Cell detection demos:** [TensorRT](https://www.youtube.com/watch?v=wP3FX8Q1Qn0) | [ONNX Runtime](https://www.youtube.com/watch?v=E8D2qqdhG7I) | [PyTorch](https://www.youtube.com/watch?v=O2x_RBgCVEg)

[![YOLOv8n Cell Detector](https://img.youtube.com/vi/wP3FX8Q1Qn0/0.jpg)](https://www.youtube.com/watch?v=wP3FX8Q1Qn0)

*Footage credit: [Dr. Signal tech-bio](https://www.drsignal.com.tw/zh-hant-tw/tech-bio)*

---

## 4. Optional Specialization: LoRA Adaptation & VLM Label Verification

The engine is training-free. LoRA is an **optional** layer for squeezing out extra domain performance.

Results are per detector. Creating and fusing the specialists into the engine is **future work, since a fused system with specialists would no longer be training-free.**

- **Detector specialists:** LoRA-adapted three open-vocabulary detector architectures (OWLv2, OmDet-Turbo, Grounding DINO), freezing text encoders to preserve open-vocabulary capability. **+7.18, +4.50, and +2.71 mAP** respectively on BDD100K-val.

| Model | Baseline | After LoRA | Δ |
|-------|----------|------------|---|
| OWLv2-base-patch16 | 20.63% | **27.81%** | +7.18% |
| OmDet-Turbo-Swin | 17.14% | **21.64%** | +4.50% |
| GroundingDINO-tiny | 20.89% | **23.60%** | +2.71% |

*mAP@[0.50:0.95], evaluated on BDD100K validation set.*

- [README](LoRA-vision-adaptation/README.md)

### VLM Captioner Distillation: Generalist → Driving Specialist
- **Label verification:** a lightweight module that verifies detector-predicted labels by describing each detected object. I distill the 32B QwenVL3 teacher's captions of the objects detected by our ensemble on KITTI, then use these caption-crop pairs to LoRA-finetune the 8B student. **Tripled BLEU-4 (0.11 → 0.34) and nearly tripled CIDEr (1.07 → 3.01)**, with ROUGE-L up from 0.32 to 0.52.

**VLM Captioner Distillation (32B teacher → 8B student) on KITTI**

| Metric | Before | After | Δ |
|--------|--------|-------|---|
| BLEU-4 | 0.11 | **0.34** | +0.23 |
| ROUGE-L | 0.32 | **0.52** | +0.20 |
| CIDEr | 1.07 | **3.01** | +1.94 |

*BLEU-4: exact 1-to-4 word GT match. ROUGE-L: longest in-order GT word sequence. CIDEr: similarity to descriptive GT phrases.*

#### Training curves

**OWLv2** ([google/owlv2-base-patch16-ensemble](https://huggingface.co/google/owlv2-base-patch16-ensemble))

[![OWLv2 Training Curves](https://cdn.jsdelivr.net/gh/Victor-QTP/materials@main/LoRA-vision-adaptation/OWLv2_LoRA_checkpoints_text_frozen_b16_ciou_training_losses.png)](https://cdn.jsdelivr.net/gh/Victor-QTP/materials@main/LoRA-vision-adaptation/OWLv2_LoRA_checkpoints_text_frozen_b16_ciou_training_losses.png)

**OmDet-Turbo** ([omlab/omdet-turbo-swin-tiny-hf](https://huggingface.co/omlab/omdet-turbo-swin-tiny-hf))

[![OmDet-Turbo Training Curves](https://cdn.jsdelivr.net/gh/Victor-QTP/materials@main/LoRA-vision-adaptation/Omdet_LoRA_checkpoints_ciou_50epochs.png)](https://cdn.jsdelivr.net/gh/Victor-QTP/materials@main/LoRA-vision-adaptation/Omdet_LoRA_checkpoints_ciou_50epochs.png)

**Grounding DINO** ([IDEA-Research/grounding-dino-tiny](https://huggingface.co/IDEA-Research/grounding-dino-tiny))

[![GroundingDINO Training Curves](https://cdn.jsdelivr.net/gh/Victor-QTP/materials@main/LoRA-vision-adaptation/GroundingDINO_checkpoints_ciou_textfreeze_50epochs.png)](https://cdn.jsdelivr.net/gh/Victor-QTP/materials@main/LoRA-vision-adaptation/GroundingDINO_checkpoints_ciou_textfreeze_50epochs.png)

---

## 5. Multimodal RAG: Rare Scenario & Object Finder

**Natural-language search over unannotated driving footage: no fine-tuning, no manual browsing of 100K+ frames.** Two parallel Qdrant indexes (CLIP ViT-g-14 image embeddings + Nomic text embeddings over QwenVL captions) fused into ranked results, with grounded Q&A over the retrieved frames.

[![RAG Architecture](https://cdn.jsdelivr.net/gh/Victor-QTP/materials@main/RAG_tutorial/ExtensionB_Pipeline.PNG)](https://cdn.jsdelivr.net/gh/Victor-QTP/materials@main/RAG_tutorial/ExtensionB_Pipeline.PNG)

*Three-stage architecture as run in the demo below: CLIP retrieval finds the rare scene, QwenVL captions and answers over it, and retrieved frames feed the auto-labeling engine (Section 1), focusing labeling effort on rare objects.*

**Query:** *"What vehicles are in the image with the yellow taxi?"*

[![RAG Demo - Retrieved Scene](https://cdn.jsdelivr.net/gh/Victor-QTP/materials@main/RAG_tutorial/rag_demo_yellow_taxi.png)](https://cdn.jsdelivr.net/gh/Victor-QTP/materials@main/RAG_tutorial/rag_demo_yellow_taxi.png)

**Top-1 retrieved scene** (frame_3800): a bustling urban street with a yellow taxi, several cars, a motorcycle, and various storefronts and signs.
**Answer:** *In the image with the yellow taxi, there are several cars and a motorcycle.*

**Uses:** rare-scenario mining, edge-case curation, pre-filtering data for the auto-labeling pipeline, grounded Q&A that reduces VLM hallucination.

- [Notebook](RAG_tutorial/vehicle_search_VLM_tutorial.ipynb) | [Requirements](RAG_tutorial/requirements.txt)

---

## Awards & Certificates

[**Best Paper Award, ICICT 2026 (Honolulu, HI, USA)**](https://dl.acm.org/doi/10.1145/3803291.3803339)
[Paper](https://dl.acm.org/doi/10.1145/3803291.3803339) | [Certificate](Awards_and_certificates/ICICT_26_BestPaperAward_r.pdf)

[![Best Paper Award - ICICT 2026](https://cdn.jsdelivr.net/gh/Victor-QTP/materials@main/Awards_and_certificates/ICICT_26_bestpaper.jpg)](https://cdn.jsdelivr.net/gh/Victor-QTP/materials@main/Awards_and_certificates/ICICT_26_bestpaper.jpg)

Supporting certificates:

- [IELTS 7.5 / CEFR C1](Awards_and_certificates/IELTS-Thang0001.pdf)
- [Yanmar Agri R&D Internship](Awards_and_certificates/Certificate-Yanmar.pdf)
- [KIT Bio Tech & IT Spring School](Awards_and_certificates/Certificate_KIT_BioTech_IT_Spring_School%20.pdf)
- [KIT Global Human Resource Development](Awards_and_certificates/KIT_Certificate%20of%20Completion_PHAN%20QUOC%20THANG.pdf)
- [Folder README](Awards_and_certificates/README.md)

---

## License

[MIT](LICENSE)