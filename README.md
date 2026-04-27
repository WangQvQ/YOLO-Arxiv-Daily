# 每日从arXiv中获取最新YOLO相关论文


## EgoMAGIC\- An Egocentric Video Field Medicine Dataset for Training Perception Algorithms / 

发布日期：2026-04-23

作者：Brian VanVoorst

摘要：This paper introduces EgoMAGIC \(Medical Assistance, Guidance, Instruction, and Correction\), an egocentric medical activity dataset collected as part of DARPA's Perceptually\-enabled Task Guidance \(PTG\) program. This dataset comprises 3,355 videos of 50 medical tasks, with at least 50 labeled videos per task. The primary objective of the PTG program was to develop virtual assistants integrated into augmented reality headsets to assist users in performing complex tasks.   To encourage exploration and research using this dataset, the medical training data has been released along with an action detection challenge focused on eight medical tasks. The majority of the videos were recorded using a head\-mounted stereo camera with integrated audio. From this dataset, 40 YOLO models were trained using 1.95 million labels to detect 124 medical objects, providing a robust starting point for developers working on medical AI applications.   In addition to introducing the dataset, this paper presents baseline results on action detection for the eight selected medical tasks across three models, with the best\-performing method achieving average mAP 0.526. Although this paper primarily addresses action detection as the benchmark, the EgoMAGIC dataset is equally suitable for action recognition, object identification and detection, error detection, and other challenging computer vision tasks.   The dataset is accessible via zenodo.org \(DOI: 10.5281/zenodo.19239154\).

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2604.22036v1)

---


## Proactive Detection of GUI Defects in Multi\-Window Scenarios via Multimodal Reasoning / 

发布日期：2026-04-21

作者：Xinyao Zhang

摘要：Multi\-window mobile scenarios, such as split\-screen and foldable modes, make GUI display defects more likely by forcing applications to adapt to changing window sizes and dynamic layout reflow. Existing detection techniques are limited in two ways: they are largely passive, analyzing screenshots only after problematic states have been reached, and they are mainly designed for conventional full\-screen interfaces, making them less effective in multi\-window settings.We propose an end\-to\-end framework for GUI display defect detection in multi\-window mobile scenarios. The framework proactively triggers split\-screen, foldable, and window\-transition states during app exploration, uses Set\-of\-Mark \(SoM\) to align screenshots with widget\-level interface elements, and leverages multimodal large language models with chain\-of\-thought prompting to detect, localize, and explain display defects. We also construct a benchmark of GUI display defects using 50 real\-world Android applications.Experimental results show that multi\-window settings substantially increase the exposure of layout\-related defects, with text truncation increasing by 184% compared with conventional full\-screen settings. At the application level, our method detects 40 defect\-prone apps with a false positive rate of 10.00% and a false negative rate of 11.11%, outperforming OwlEye and YOLO\-based baselines. At the fine\-grained level, it achieves the best F1 score of 87.2% for widget occlusion detection.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2604.19081v1)

---


## Domain\-Specialized Object Detection via Model\-Level Mixtures of Experts / 

发布日期：2026-04-20

作者：Svetlana Pavlitska

摘要：Mixture\-of\-Experts \(MoE\) models provide a structured approach to combining specialized neural networks and offer greater interpretability than conventional ensembles. While MoEs have been successfully applied to image classification and semantic segmentation, their use in object detection remains limited due to challenges in merging dense and structured predictions. In this work, we investigate model\-level mixtures of object detectors and analyze their suitability for improving performance and interpretability in object detection. We propose an MoE architecture that combines YOLO\-based detectors trained on semantically disjoint data subsets, with a learned gating network that dynamically weights expert contributions. We study different strategies for fusing detection outputs and for training the gating mechanism, including balancing losses to prevent expert collapse. Experiments on the BDD100K dataset demonstrate that the proposed MoE consistently outperforms standard ensemble approaches and provides insights into expert specialization across domains, highlighting model\-level MoEs as a viable alternative to traditional ensembling for object detection. Our code is available at https://github.com/KASTEL\-MobilityLab/mixtures\-of\-experts/.

中文摘要：


代码链接：https://github.com/KASTEL-MobilityLab/mixtures-of-experts/.

论文链接：[阅读更多](http://arxiv.org/abs/2604.18256v1)

---


## Autonomous Unmanned Aircraft Systems for Enhanced Search and Rescue of Drowning Swimmers: Image\-Based Localization and Mission Simulation / 

发布日期：2026-04-20

作者：Sascha Emanuel Zell

摘要：Drowning is an omnipresent risk associated with any activity on or in the water, and rescuing a drowning person is particularly challenging because of the time pressure, making a short response time important. Further complicating water rescue are unsupervised and extensive swimming areas, precise localization of the target, and the transport of rescue personnel. Technical innovations can provide a remedy: We propose an Unmanned Aircraft System \(UAS\), also known as a drone\-in\-a\-box system, consisting of a fleet of Unmanned Aerial Vehicles \(UAVs\) allocated to purpose\-built hangars near swimming areas. In an emergency, the UAS can be deployed in addition to Standard Rescue Operation \(SRO\) equipment to locate the distressed person early by performing a fully automated Search and Rescue \(S&R\) operation and dropping a flotation device. In this paper, we address automatically locating distressed swimmers using the image\-based object detection architecture You Only Look Once \(YOLO\). We present a dataset created for this application and outline the training process. We evaluate the performance of YOLO versions 3, 5, and 8 and architecture sizes \(nano, extra\-large\) using Mean Average Precision \(mAP\) metrics mAP@.5 and mAP@.5:.95. Furthermore, we present two Discrete\-Event Simulation \(DES\) approaches to simulate response times of SRO and UAS\-based water rescue. This enables estimation of time savings relative to SRO when selecting the UAS configuration \(type, number, and location of UAVs and hangars\). Computational experiments for a test area in the Lusatian Lake District, Germany, show that UAS assistance shortens response time. Even a small UAS with two hangars, each containing one UAV, reduces response time by a factor of five compared to SRO.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2604.18088v1)

---


## Interpretable Human Activity Recognition for Subtle Robbery Detection in Surveillance Videos / 

发布日期：2026-04-15

作者：Bryan Jhoan Cazáres Leyva

摘要：Non\-violent street robberies \(snatch\-and\-run\) are difficult to detect automatically because they are brief, subtle, and often indistinguishable from benign human interactions in unconstrained surveillance footage. This paper presents a hybrid, pose\-driven approach for detecting snatch\-and\-run events that combines real\-time perception with an interpretable classification stage suitable for edge deployment. The system uses a YOLO\-based pose estimator to extract body keypoints for each tracked person and computes kinematic and interaction features describing hand speed, arm extension, proximity, and relative motion between an aggressor\-victim pair. A Random Forest classifier is trained on these descriptors, and a temporal hysteresis filter is applied to stabilize frame\-level predictions and reduce spurious alarms. We evaluate the method on a staged dataset and on a disjoint test set collected from internet videos, demonstrating promising generalization across different scenes and camera viewpoints. Finally, we implement the complete pipeline on an NVIDIA Jetson Nano and report real\-time performance, supporting the feasibility of proactive, on\-device robbery detection.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2604.14329v1)

---

