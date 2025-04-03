One Shot Object Detection with GroundingDINO and Segment Anything Model (SAM):
This project is an implementation from the paper "Adapting Pre-Trained Vision Models for Novel Instance Detection and Segmentation" (https://arxiv.org/abs/2405.17859).

This project provides a pipeline for one shot object detection using the GroundingDINO object detector and the Segment Anything Model (SAM) for segmentation. The goal is to detect objects from a reference image (template), segment them, extract features, and compute similarity for object detection.

Key Points:

GroundingDINO for Object Detection: Detects objects in images based on textual prompts (e.g., "cycle").

Segment Anything Model (SAM) for Segmentation: Segments the detected objects and generates masks for each object.

Object Proposals: Extracts object proposals based on bounding boxes and segmentation masks.

Feature Extraction using DINOv2: Uses DINOv2 for feature extraction of the segmented objects.

Object Detection : Computes feature similarity between reference and test images to identify the target instance.



