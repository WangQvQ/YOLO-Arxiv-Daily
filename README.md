# 每日从arXiv中获取最新YOLO相关论文


## SPMamba\-YOLO: An Underwater Object Detection Network Based on Multi\-Scale Feature Enhancement and Global Context Modeling / 

发布日期：2026-02-26

作者：Guanghao Liao

摘要：Underwater object detection is a critical yet challenging research problem owing to severe light attenuation, color distortion, background clutter, and the small scale of underwater targets. To address these challenges, we propose SPMamba\-YOLO, a novel underwater object detection network that integrates multi\-scale feature enhancement with global context modeling. Specifically, a Spatial Pyramid Pooling Enhanced Layer Aggregation Network \(SPPELAN\) module is introduced to strengthen multi\-scale feature aggregation and expand the receptive field, while a Pyramid Split Attention \(PSA\) mechanism enhances feature discrimination by emphasizing informative regions and suppressing background interference. In addition, a Mamba\-based state space modeling module is incorporated to efficiently capture long\-range dependencies and global contextual information, thereby improving detection robustness in complex underwater environments. Extensive experiments on the URPC2022 dataset demonstrate that SPMamba\-YOLO outperforms the YOLOv8n baseline by more than 4.9% in mAP@0.5, particularly for small and densely distributed underwater objects, while maintaining a favorable balance between detection accuracy and computational cost.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2602.22674v1)

---


## Don't let the information slip away / 

发布日期：2026-02-26

作者：Taozhe Li

摘要：Real\-time object detection has advanced rapidly in recent years. The YOLO series of detectors is among the most well\-known CNN\-based object detection models and cannot be overlooked. The latest version, YOLOv26, was recently released, while YOLOv12 achieved state\-of\-the\-art \(SOTA\) performance with 55.2 mAP on the COCO val2017 dataset. Meanwhile, transformer\-based object detection models, also known as DEtection TRansformer \(DETR\), have demonstrated impressive performance. RT\-DETR is an outstanding model that outperformed the YOLO series in both speed and accuracy when it was released. Its successor, RT\-DETRv2, achieved 53.4 mAP on the COCO val2017 dataset. However, despite their remarkable performance, all these models let information to slip away. They primarily focus on the features of foreground objects while neglecting the contextual information provided by the background. We believe that background information can significantly aid object detection tasks. For example, cars are more likely to appear on roads rather than in offices, while wild animals are more likely to be found in forests or remote areas rather than on busy streets. To address this gap, we propose an object detection model called Association DETR, which achieves state\-of\-the\-art results compared to other object detection models on the COCO val2017 dataset.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2602.22595v1)

---


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

