# 每日从arXiv中获取最新YOLO相关论文


## Towards Object Segmentation Mask Selection Using Specular Reflections / 

发布日期：2026-02-25

作者：Katja Kossira

摘要：Specular reflections pose a significant challenge for object segmentation, as their sharp intensity transitions often mislead both conventional algorithms and deep learning based methods. However, as the specular reflection must lie on the surface of the object, this fact can be exploited to improve the segmentation masks. By identifying the largest region containing the reflection as the object, we derive a more accurate object mask without requiring specialized training data or model adaption. We evaluate our method on both synthetic and real world images and compare it against established and state\-of\-the\-art techniques including Otsu thresholding, YOLO, and SAM2. Compared to the best performing baseline SAM2, our approach achieves up to 26.7% improvement in IoU, 22.3% in DSC, and 9.7% in pixel accuracy. Qualitative evaluations on real world images further confirm the robustness and generalizability of the proposed approach.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2602.21777v1)

---


## DAGS\-SLAM: Dynamic\-Aware 3DGS SLAM via Spatiotemporal Motion Probability and Uncertainty\-Aware Scheduling / 

发布日期：2026-02-25

作者：Li Zhang

摘要：Mobile robots and IoT devices demand real\-time localization and dense reconstruction under tight compute and energy budgets. While 3D Gaussian Splatting \(3DGS\) enables efficient dense SLAM, dynamic objects and occlusions still degrade tracking and mapping. Existing dynamic 3DGS\-SLAM often relies on heavy optical flow and per\-frame segmentation, which is costly for mobile deployment and brittle under challenging illumination. We present DAGS\-SLAM, a dynamic\-aware 3DGS\-SLAM system that maintains a spatiotemporal motion probability \(MP\) state per Gaussian and triggers semantics on demand via an uncertainty\-aware scheduler. DAGS\-SLAM fuses lightweight YOLO instance priors with geometric cues to estimate and temporally update MP, propagates MP to the front\-end for dynamic\-aware correspondence selection, and suppresses dynamic artifacts in the back\-end via MP\-guided optimization. Experiments on public dynamic RGB\-D benchmarks show improved reconstruction and robust tracking while sustaining real\-time throughput on a commodity GPU, demonstrating a practical speed\-accuracy tradeoff with reduced semantic invocations toward mobile deployment.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2602.21644v1)

---


## EKF\-Based Depth Camera and Deep Learning Fusion for UAV\-Person Distance Estimation and Following in SAR Operations / 

发布日期：2026-02-24

作者：Luka Šiktar

摘要：Search and rescue \(SAR\) operations require rapid responses to save lives or property. Unmanned Aerial Vehicles \(UAVs\) equipped with vision\-based systems support these missions through prior terrain investigation or real\-time assistance during the mission itself. Vision\-based UAV frameworks aid human search tasks by detecting and recognizing specific individuals, then tracking and following them while maintaining a safe distance. A key safety requirement for UAV following is the accurate estimation of the distance between camera and target object under real\-world conditions, achieved by fusing multiple image modalities. UAVs with deep learning\-based vision systems offer a new approach to the planning and execution of SAR operations. As part of the system for automatic people detection and face recognition using deep learning, in this paper we present the fusion of depth camera measurements and monocular camera\-to\-body distance estimation for robust tracking and following. Deep learning\-based filtering of depth camera data and estimation of camera\-to\-body distance from a monocular camera are achieved with YOLO\-pose, enabling real\-time fusion of depth information using the Extended Kalman Filter \(EKF\) algorithm. The proposed subsystem, designed for use in drones, estimates and measures the distance between the depth camera and the human body keypoints, to maintain the safe distance between the drone and the human target. Our system provides an accurate estimated distance, which has been validated against motion capture ground truth data. The system has been tested in real time indoors, where it reduces the average errors, root mean square error \(RMSE\) and standard deviations of distance estimation up to 15,3% in three tested scenarios.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2602.20958v1)

---


## A Text\-Guided Vision Model for Enhanced Recognition of Small Instances / 

发布日期：2026-02-23

作者：Hyun\-Ki Jung

摘要：As drone\-based object detection technology continues to evolve, the demand is shifting from merely detecting objects to enabling users to accurately identify specific targets. For example, users can input particular targets as prompts to precisely detect desired objects. To address this need, an efficient text\-guided object detection model has been developed to enhance the detection of small objects. Specifically, an improved version of the existing YOLO\-World model is introduced. The proposed method replaces the C2f layer in the YOLOv8 backbone with a C3k2 layer, enabling more precise representation of local features, particularly for small objects or those with clearly defined boundaries. Additionally, the proposed architecture improves processing speed and efficiency through parallel processing optimization, while also contributing to a more lightweight model design. Comparative experiments on the VisDrone dataset show that the proposed model outperforms the original YOLO\-World model, with precision increasing from 40.6% to 41.6%, recall from 30.8% to 31%, F1 score from 35% to 35.5%, and mAP@0.5 from 30.4% to 30.7%, confirming its enhanced accuracy. Furthermore, the model demonstrates superior lightweight performance, with the parameter count reduced from 4 million to 3.8 million and FLOPs decreasing from 15.7 billion to 15.2 billion. These results indicate that the proposed approach provides a practical and effective solution for precise object detection in drone\-based applications.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2602.19503v1)

---


## TactEx: An Explainable Multimodal Robotic Interaction Framework for Human\-Like Touch and Hardness Estimation / 

发布日期：2026-02-21

作者：Felix Verstraete

摘要：Accurate perception of object hardness is essential for safe and dexterous contact\-rich robotic manipulation. Here, we present TactEx, an explainable multimodal robotic interaction framework that unifies vision, touch, and language for human\-like hardness estimation and interactive guidance. We evaluate TactEx on fruit\-ripeness assessment, a representative task that requires both tactile sensing and contextual understanding. The system fuses GelSight\-Mini tactile streams with RGB observations and language prompts. A ResNet50\+LSTM model estimates hardness from sequential tactile data, while a cross\-modal alignment module combines visual cues with guidance from a large language model \(LLM\). This explainable multimodal interface allows users to distinguish ripeness levels with statistically significant class separation \(p < 0.01 for all fruit pairs\). For touch placement, we compare YOLO with Grounded\-SAM \(GSAM\) and find GSAM to be more robust for fine\-grained segmentation and contact\-site selection. A lightweight LLM parses user instructions and produces grounded natural\-language explanations linked to the tactile outputs. In end\-to\-end evaluations, TactEx attains 90% task success on simple user queries and generalises to novel tasks without large\-scale tuning. These results highlight the promise of combining pretrained visual and tactile models with language grounding to advance explainable, human\-like touch perception and decision\-making in robotics.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2602.18967v1)

---

