<div align="center">

# YOLO ArXiv Daily

[![Daily Papers](https://img.shields.io/badge/📅-每日更新-blue)]()
[![arXiv](https://img.shields.io/badge/arXiv-最新论文-red)](https://arxiv.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)

*自动追踪 YOLO 相关最新论文，提供中英文双语摘要*

</div>

---

## 📑 论文列表

> ### 1. Hardware\-Accelerated Instance Segmentation for Resource\-Constrained Space Robotics with Criticality Analysis
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-09-02 |
> | 👤 作者 | Siddhant Shete |
>
> **📄 英文摘要：**
> Autonomous lunar missions require real\-time per\- ception under three coupled constraints: extreme low\-light conditions, limited onboard compute, and radiation\-induced hardware faults that can silently corrupt inference. We present a deployment\-oriented instance segmentation framework for resource\-constrained lunar robotics that jointly addresses quan\- tization calibration and system\-level fault exposure under strict compute constraints. First, we introduce Activation Variance Informative Sampling \(AVIS\), a label\-free calibration strategy that deterministically selects calibration samples based on activation variance statistics. Second, we deploy a YOLO\-based segmentation model on a Deep Learning Processor Unit \(DPU\) with architectural modifications that reduce CPU fallback paths and enable statically compiled execution with bounded latency in low\-lighting conditions. We further introduce a software\-level criticality analysis to estimate fault exposure and guide mitigation under radiation\-constrained operation. On a lunar micro\-rover platform, AVIS with bias correction recovers 69.8% of quantization\-induced accuracy loss while achieving 309 ms inference latency and 5.7 W power consumption. Targeted mitigation reduces global criticality by 31.7%. The results demonstrate an integrated approach and a blueprint for a reliable and safe AI perception framework under space deployment constraints.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2609.02219v1)

---

> ### 2. DESA\-TTA: Dynamic EMA and Source Anchoring for Test\-Time Adaptation
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-09-01 |
> | 👤 作者 | Atif Belal |
>
> **📄 英文摘要：**
> Vision\-language object detectors \(VLODs\) achieve strong zero\-shot performance but remain vulnerable to distribution shifts during deployment. Mean\-teacher methods for test\-time adaptation \(TTA\) can improve robustness by updating a student model using teacher\-generated pseudo\-labels. However, mean\-teacher TTA is highly sensitive to the choice of a fixed exponential moving average \(EMA\) coefficient for teacher updates, and repeated optimization with noisy pseudo\-labels can cause cumulative student drift. We propose Dynamic EMA and Source Anchoring for TTA \(DESA\-TTA\), a low\-overhead method that jointly regulates teacher updates and student drift through dynamic temporal averaging and source anchoring. Dynamic temporal averaging estimates teacher uncertainty from pseudo\-label confidence and box density and uses it to select a sample\-wise EMA coefficient within bounds determined by teacher parameter drift. Source anchoring partially restores the updated student parameters toward their pretrained values, with the anchoring strength increasing according to student drift. Experiments across diverse distribution shifts and two VLOD architectures show consistent improvements over existing TTA methods. On VOC\-C, DESA\-TTA improves AP$\_\{50\}$ by 14.5 points over zero\-shot inference while achieving 55% higher inference throughput than the previous state\-of\-the\-art TTA method for YOLO\-World. Our code: https://github.com/imatif17/DESA\-TTA
>
> **💻 代码链接：** https://github.com/imatif17/DESA-TTA
>
> 🔗 [阅读论文](http://arxiv.org/abs/2609.01795v1)

---

> ### 3. Vision\-Based Leader\-Follower Formation Control for Cooperative UAVs in GPS\-Degraded Environments
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-09-01 |
> | 👤 作者 | Deekshitha Angadi |
>
> **📄 英文摘要：**
> Cooperation in multi\-UAV systems requires reliable relative perception so that follower vehicles can maintain formation and continue their mission safely even when absolute positioning sensors degrade or fail. This paper presents a vision\-based cooperative formation framework running on a follower UAV that uses a front\-facing RGB\-D camera to detect, track, and localize a leader UAV in real\-time. A lightweight YOLO\-based detector is trained on a dedicated drone dataset and deployed onboard to predict leader bounding boxes, which are then fused with depth information via a pinhole camera model to estimate the leader's relative pose. These estimates provide a leader\-follower position controller and can also be used as a backup when GPS or external localization is unavailable. This framework is implemented as a set of ROS nodes and evaluated in a physics\-based multi\-UAV simulation built on XTDrone, with sensor noise and communication dropouts. We evaluate detection accuracy, runtime, and formation\-keeping error under nominal conditions and under simulated failures of the positioning sensors. The results show that the proposed framework maintains stable leader\-follower formations with reasonable computational cost and provides a practical basis for extending vision\-based cooperative formation control to real\-world multi\-UAV systems.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2609.01420v1)

---

> ### 4. Real\-Time Video Anomaly Detection Using YOLO Pose Estimation and CLIP\-Based Semantic Scoring
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-08-31 |
> | 👤 作者 | Vanodhya G. Warnasooriya |
>
> **📄 英文摘要：**
> We propose a lightweight two\-stage framework for real\-time video anomaly detection. The first stage employs YOLO v11n\-pose to detect persons and extract seventeen skeletal keypoints in a single forward pass. The second stage encodes each cropped person region through CLIP ViT\-B/32 and computes cosine similarity against predefined textual descriptions of anomalous behaviors. This architecture eliminates the need for optical flow, standalone pose estimators, and density\-based scoring modules. Experiments on CUHK Avenue, ShanghaiTech Campus, and a custom indoor dataset collected at Chulalongkorn University demonstrate an end\-to\-end throughput of approximately 51 FPS on an NVIDIA Titan XP GPU, a 3.36x speedup over the multi\-feature baseline, while maintaining frame\-level AUROC values of 89.26%, 70.26%, and 84.13%, respectively.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2608.31074v1)

---

> ### 5. SynCrash: A Multi\-Stage Pipeline for Zero\-Shot Accident Detection and Localization in Traffic Surveillance Video
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-08-30 |
> | 👤 作者 | Arkya Jyoti Bagchi |
>
> **📄 英文摘要：**
> We present SynCrash, a multi\-stage pipeline for zero\-shot accident detection, spatial localization, and collision\-type classification in fixed\-view CCTV surveillance video. Our approach addresses the ACCIDENT at CVPR 2026 Challenge, which requires predicting when an accident occurs, where in the frame the impact happens, and what type of collision it is, all without access to labeled real\-world training data. The pipeline operates in three decoupled stages: \(1\) Temporal localization via a VideoMAEv2\-giant backbone fine\-tuned on CARLA\-based synthetic clips with metadata\-aware embeddings and dense sliding\-window inference; \(2\) Spatial localization using YOLO for object detection combined with a physics\-informed hybrid heuristic that leverages bounding\-box overlap and trajectory\-based reasoning to predict the impact point; and \(3\) Collision\-type classification using a lightweight rule\-based strategy derived from the number and configuration of detected vehicles. The key insight is that temporal understanding benefits from supervised fine\-tuning on synthetic data, whereas spatial understanding is better served by pretrained object detectors and physics priors that transfer naturally across domains.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2608.29759v1)

---

<div align="center">

*由 [YOLO-Arxiv-Daily](https://github.com/WangQvQ/YOLO-Arxiv-Daily) 自动生成*

</div>