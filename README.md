# 每日从arXiv中获取最新YOLO相关论文


## ER\-Pose: Rethinking Keypoint\-Driven Representation Learning for Real\-Time Human Pose Estimation / 

发布日期：2026-03-09

作者：Nanjun Li

摘要：Single\-stage multi\-person pose estimation aims to jointly perform human localization and keypoint prediction within a unified framework, offering advantages in inference efficiency and architectural simplicity. Consequently, multi\-scale real\-time detection architectures, such as YOLO\-like models, are widely adopted for real\-time pose estimation. However, these approaches typically inherit a box\-driven modeling paradigm from object detection, in which pose estimation is implicitly constrained by bounding\-box supervision during training. This formulation introduces biases in sample assignment and feature representation, resulting in task misalignment and ultimately limiting pose estimation accuracy. In this work, we revisit box\-driven single\-stage pose estimation from a keypoint\-driven perspective and identify semantic conflicts among parallel objectives as a key source of performance degradation. To address this issue, we propose a keypoint\-driven learning paradigm that elevates pose estimation to a primary prediction objective. Specifically, we remove bounding\-box prediction and redesign the prediction head to better accommodate the high\-dimensional structured representations for pose estimation. We further introduce a keypoint\-driven dynamic sample assignment strategy to align training objectives with pose evaluation metrics, enabling dense supervision during training and efficient NMS\-free inference. In addition, we propose a smooth OKS\-based loss function to stabilize optimization in regression\-based pose estimation. Based on these designs, we develop a single\-stage multi\-person pose estimation framework, termed ER\-Pose. On MS COCO and CrowdPose, ER\-Pose\-n achieves AP improvements of 3.2/6.7 without pre\-training and 7.4/4.9 with pre\-training respectively compared with the baseline YOLO\-Pose. These improvements are achieved with fewer parameters and higher inference efficiency.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2603.08681v1)

---


## Real\-Time Drone Detection in Event Cameras via Per\-Pixel Frequency Analysis / 

发布日期：2026-03-09

作者：Michael Bezick

摘要：Detecting fast\-moving objects, such as unmanned aerial vehicle \(UAV\), from event camera data is challenging due to the sparse, asynchronous nature of the input. Traditional Discrete Fourier Transforms \(DFT\) are effective at identifying periodic signals, such as spinning rotors, but they assume uniformly sampled data, which event cameras do not provide. We propose a novel per\-pixel temporal analysis framework using the Non\-uniform Discrete Fourier Transform \(NDFT\), which we call Drone Detection via Harmonic Fingerprinting \(DDHF\). Our method uses purely analytical techniques that identify the frequency signature of drone rotors, as characterized by frequency combs in their power spectra, enabling a tunable and generalizable algorithm that achieves accurate real\-time localization of UAV. We compare against a YOLO detector under equivalent conditions, demonstrating improvement in accuracy and latency across a difficult array of drone speeds, distances, and scenarios. DDHF achieves an average localization F1 score of 90.89% and average latency of 2.39ms per frame, while YOLO achieves an F1 score of 66.74% and requires 12.40ms per frame. Through utilization of purely analytic techniques, DDHF is quickly tuned on small data, easily interpretable, and achieves competitive accuracies and latencies to deep learning alternatives.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2603.08386v1)

---


## Evaluating Synthetic Data for Baggage Trolley Detection in Airport Logistics / 

发布日期：2026-03-08

作者：Abdeldjalil Taibi

摘要：Efficient luggage trolley management is critical for reducing congestion and ensuring asset availability in modern airports. Automated detection systems face two main challenges. First, strict security and privacy regulations limit large\-scale data collection. Second, existing public datasets lack the diversity, scale, and annotation quality needed to handle dense, overlapping trolley arrangements typical of real\-world operations.   To address these limitations, we introduce a synthetic data generation pipeline based on a high\-fidelity Digital Twin of Algiers International Airport using NVIDIA Omniverse. The pipeline produces richly annotated data with oriented bounding boxes, capturing complex trolley formations, including tightly nested chains. We evaluate YOLO\-OBB using five training strategies: real\-only, synthetic\-only, linear probing, full fine\-tuning, and mixed training. This allows us to assess how synthetic data can complement limited real\-world annotations.   Our results show that mixed training with synthetic data and only 40 percent of real annotations matches or exceeds the full real\-data baseline, achieving 0.94 mAP@50 and 0.77 mAP@50\-95, while reducing annotation effort by 25 to 35 percent. Multi\-seed experiments confirm strong reproducibility with a standard deviation below 0.01 on mAP@50, demonstrating the practical effectiveness of synthetic data for automated trolley detection.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2603.07645v1)

---


## A Lightweight Digital\-Twin\-Based Framework for Edge\-Assisted Vehicle Tracking and Collision Prediction / 

发布日期：2026-03-07

作者：Murat Arda Onsu

摘要：Vehicle tracking, motion estimation, and collision prediction are fundamental components of traffic safety and management in Intelligent Transportation Systems \(ITS\). Many recent approaches rely on computationally intensive prediction models, which limits their practical deployment on resource\-constrained edge devices. This paper presents a lightweight digital\-twin\-based framework for vehicle tracking and spatiotemporal collision prediction that relies solely on object detection, without requiring complex trajectory prediction networks. The framework is implemented and evaluated in Quanser Interactive Labs \(QLabs\), a high\-fidelity digital twin of an urban traffic environment that enables controlled and repeatable scenario generation. A YOLO\-based detector is deployed on simulated edge cameras to localize vehicles and extract frame\-level centroid trajectories. Offline path maps are constructed from multiple traversals and indexed using K\-D trees to support efficient online association between detected vehicles and road segments. During runtime, consistent vehicle identifiers are maintained, vehicle speed and direction are estimated from the temporal evolution of path indices, and future positions are predicted accordingly. Potential collisions are identified by analyzing both spatial proximity and temporal overlap of predicted future trajectories. Our experimental results across diverse simulated urban scenarios show that the proposed framework predicts approximately 88% of collision events prior to occurrence while maintaining low computational overhead suitable for edge deployment. Rather than introducing a computationally intensive prediction model, this work introduces a lightweight digital\-twin\-based solution for vehicle tracking and collision prediction, tailored for real\-time edge deployment in ITS.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2603.07338v1)

---


## OV\-DEIM: Real\-time DETR\-Style Open\-Vocabulary Object Detection with GridSynthetic Augmentation / 

发布日期：2026-03-07

作者：Leilei Wang

摘要：Real\-time open\-vocabulary object detection \(OVOD\) is essential for practical deployment in dynamic environments, where models must recognize a large and evolving set of categories under strict latency constraints. Current real\-time OVOD methods are predominantly built upon YOLO\-style models. In contrast, real\-time DETR\-based methods still lag behind in terms of inference latency, model lightweightness, and overall performance. In this work, we present OV\-DEIM, an end\-to\-end DETR\-style open\-vocabulary detector built upon the recent DEIMv2 framework with integrated vision\-language modeling for efficient open\-vocabulary inference. We further introduce a simple query supplement strategy that improves Fixed AP without compromising inference speed. Beyond architectural improvements, we introduce GridSynthetic, a simple yet effective data augmentation strategy that composes multiple training samples into structured image grids. By exposing the model to richer object co\-occurrence patterns and spatial layouts within a single forward pass, GridSynthetic mitigates the negative impact of noisy localization signals on the classification loss and improves semantic discrimination, particularly for rare categories. Extensive experiments demonstrate that OV\-DEIM achieves state\-of\-the\-art performance on open\-vocabulary detection benchmarks, delivering superior efficiency and notable improvements on challenging rare categories. Code and pretrained models are available at https://github.com/wleilei/OV\-DEIM.

中文摘要：


代码链接：https://github.com/wleilei/OV-DEIM.

论文链接：[阅读更多](http://arxiv.org/abs/2603.07022v1)

---

