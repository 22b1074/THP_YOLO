# Multiple Object Tracking with Re-ID

This repository contains `tracking.py`, which implements the BoxSort tracker. I chose BoxSort over DeepSort because it includes a Re-ID model that helps maintain consistent IDs for the same object, even if the object is missing or occluded for several frames. Specifically, the osnet_x0_25_market50 Re-ID model is used for faster inference without significant loss in accuracy.


# YOLOv11 Tweaks

This repo contains experiments modifying a YOLOv11 model in `yolov11.ipynb`, including the training loop and evaluation metrics for predictions.

- Current change:
  - Added RepConv in detection head

- Planned changes:
  - Change stride = 1 in initial layers
  - Try other backbone and neck modifications
