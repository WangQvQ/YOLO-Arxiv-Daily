# 每日从arXiv中获取最新YOLO相关论文


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


## Depth\-Enhanced YOLO\-SAM2 Detection for Reliable Ballast Insufficiency Identification / 

发布日期：2026-02-21

作者：Shiyu Liu

摘要：This paper presents a depth\-enhanced YOLO\-SAM2 framework for detecting ballast insufficiency in railway tracks using RGB\-D data. Although YOLOv8 provides reliable localization, the RGB\-only model shows limited safety performance, achieving high precision \(0.99\) but low recall \(0.49\) due to insufficient ballast, as it tends to over\-predict the sufficient class. To improve reliability, we incorporate depth\-based geometric analysis enabled by a sleeper\-aligned depth\-correction pipeline that compensates for RealSense spatial distortion using polynomial modeling, RANSAC, and temporal smoothing. SAM2 segmentation further refines region\-of\-interest masks, enabling accurate extraction of sleeper and ballast profiles for geometric classification.   Experiments on field\-collected top\-down RGB\-D data show that depth\-enhanced configurations substantially improve the detection of insufficient ballast. Depending on bounding\-box sampling \(AABB or RBB\) and geometric criteria, recall increases from 0.49 to as high as 0.80, and F1\-score improves from 0.66 to over 0.80. These results demonstrate that integrating depth correction with YOLO\-SAM2 yields a more robust and reliable approach for automated railway ballast inspection, particularly in visually ambiguous or safety\-critical scenarios.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2602.18961v1)

---


## BloomNet: Exploring Single vs. Multiple Object Annotation for Flower Recognition Using YOLO Variants / 

发布日期：2026-02-20

作者：Safwat Nusrat

摘要：Precise localization and recognition of flowers are crucial for advancing automated agriculture, particularly in plant phenotyping, crop estimation, and yield monitoring. This paper benchmarks several YOLO architectures such as YOLOv5s, YOLOv8n/s/m, and YOLOv12n for flower object detection under two annotation regimes: single\-image single\-bounding box \(SISBB\) and single\-image multiple\-bounding box \(SIMBB\). The FloralSix dataset, comprising 2,816 high\-resolution photos of six different flower species, is also introduced. It is annotated for both dense \(clustered\) and sparse \(isolated\) scenarios. The models were evaluated using Precision, Recall, and Mean Average Precision \(mAP\) at IoU thresholds of 0.5 \(mAP@0.5\) and 0.5\-0.95 \(mAP@0.5:0.95\). In SISBB, YOLOv8m \(SGD\) achieved the best results with Precision 0.956, Recall 0.951, mAP@0.5 0.978, and mAP@0.5:0.95 0.865, illustrating strong accuracy in detecting isolated flowers. With mAP@0.5 0.934 and mAP@0.5:0.95 0.752, YOLOv12n \(SGD\) outperformed the more complicated SIMBB scenario, proving robustness in dense, multi\-object detection. Results show how annotation density, IoU thresholds, and model size interact: recall\-optimized models perform better in crowded environments, whereas precision\-oriented models perform best in sparse scenarios. In both cases, the Stochastic Gradient Descent \(SGD\) optimizer consistently performed better than alternatives. These density\-sensitive sensors are helpful for non\-destructive crop analysis, growth tracking, robotic pollination, and stress evaluation.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2602.18585v1)

---

