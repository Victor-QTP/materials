### Zero-Shot Detection & Segmentation Pipeline

**1. Semantic Localization (Left):** VLM-based detectors identify and generate open-vocabulary descriptive captions for rare or unannotated objects in the wild.

**2. Pixel-Level Refinement (Right):** Bounding boxes and open-vocabulary descriptions are passed into SAM3 as prompts, and then converted into pixel-level instance masks. This extracts the exact geometric boundaries of objects, bridging the gap between high-level semantic understanding and spatial precision.

<div style="display: flex; flex-direction: row;">
  <img src="BDD100K_val_c415a08c-50060410.png" width="49%">
  <img src="BDD100K_val_c415a08c-50060410_SAM3.png" width="49%">
</div>