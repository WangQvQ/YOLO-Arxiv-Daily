<div align="center">

# YOLO ArXiv Daily

[![Daily Papers](https://img.shields.io/badge/📅-每日更新-blue)]()
[![arXiv](https://img.shields.io/badge/arXiv-最新论文-red)](https://arxiv.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)

*自动追踪 YOLO 相关最新论文，提供中英文双语摘要*

</div>

---

## 📑 论文列表

> ### 1. Image\-Domain Tilt Constrained Distributed Fusion for Maneuvering UAV Tracking with Multi\-Camera Electro\-Optical Observations
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-07-01 |
> | 👤 作者 | Minxing Sun |
>
> **📄 英文摘要：**
> Short\-horizon prediction is essential for electro\-optical UAV tracking, especially when the target is small, maneuvering, or intermittently observed. Image center, line\-of\-sight, and range measurements provide direct constraints on target position, but their constraints on acceleration are weak. As a result, prediction can lag during aggressive maneuvers.   This paper proposes an image\-domain tilt constrained distributed fusion method for maneuvering UAV tracking. The method uses the apparent roll and pitch of a rotorcraft target in the image as low\-level maneuver cues. A weak\-prior auto\-labeling pipeline first generates oriented bounding box and image\-domain tilt labels from synchronized video, gimbal IMU, and UAV IMU data. A YOLO\-OBB detector is then trained to provide online target position and tilt measurements. The front\-end Python implementation is publicly available at github.com/ShineMinxing/PythonYOLO.   In the fusion stage, the UAV state is modeled by position, velocity, and acceleration. Image\-domain roll and pitch are introduced as acceleration\-related pseudo\-observations. For distributed tracking, one mobile gimbal camera and two fixed ground cameras are fused asynchronously. Camera attitude error states are augmented into the filter to absorb extrinsic drift and cross\-camera systematic inconsistency. A Mahalanobis gate with time\-since\-last\-valid covariance widening is used to reject false detections and handle dropouts.   In simulation, adding roll/pitch observations reduces the prediction RMSE from 1.991 m to 0.821 m and decreases the cumulative prediction error by 60.75%. In real distributed experiments, a self\-consistency evaluation shows an 18.10% reduction in cumulative prediction error. The results show that image\-domain tilt can provide useful acceleration constraints for robust short\-horizon UAV prediction.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2607.01008v1)

---

> ### 2. Semantic\-Guided Reading Order Reconstruction in Historical Armenian Newspapers with LLMs
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-07-01 |
> | 👤 作者 | Chahan Vidal\-Gorène |
>
> **📄 英文摘要：**
> This paper addresses reading order reconstruction in historical Armenian newspapers, which combine complex layouts with limited language resources. We introduce a new annotated dataset of 66 pages and compare geometric heuristics, YOLO\-based layout parsing, an end\-to\-end document model ECLAIR, and a hybrid method combining semantic zone detection with a generative LLM. Our hybrid method achieves the lowest error rates of all evaluated approaches, reducing ordering errors by up to 76% over the strongest geometric baseline, and remains robust in multi\-page settings and under noisy OCR. Rather than targeting production the method is designed as a data bootstrapping strategy enabling rapid annotation in highly under\-resourced scenarios. Alongside the dataset, we release a specialized Tesseract OCR model for historical Armenian print.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2607.00596v1)

---

> ### 3. Real\-Time Source\-Free Object Detection
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-06-30 |
> | 👤 作者 | Sairam VCR |
>
> **📄 英文摘要：**
> Real\-world detectors for autonomous driving, surveillance, and robotics must handle domain\-shifts under strict latency and memory constraints, yet existing source\-free object detection \(SFOD\) methods rely on heavyweight architectures that prioritize accuracy alone. We show this trade\-off is unnecessary: building on YOLOv10, an NMS\-free dual\-head detector, we achieve state\-of\-the\-art adaptation accuracy while being faster and more compact. We observe that directly applying vanilla mean\-teacher self\-training to dual\-head detectors leads to suboptimal adaptation performance due to two key factors. First, simple pseudo\-label generation strategies, such as using a single head or directly combining high\-confidence predictions from both heads, yield suboptimal supervision under domain\-shift. We propose DHF \(Dual\-Head Pseudo\-Label Fusion\) which selectively admits one\-to\-one \(O2O\) and one\-to\-many \(O2M\) head predictions, preserving precision and recovering missed objects. Second, we observe domain\-shift collapses multi\-scale feature discriminability. We propose the use of our MARD \(Multi\-scale Adaptive Representation Diversification\) loss which mitigates this by enforcing detection\-aware variance and covariance constraints on multi\-scale feature maps. Both modules are training\-time only, leaving inference unchanged. Across domain\-shift benchmarks, our method, RT\-SFOD yields 1.4 to 3.5% mAP gains, 1.3$times$ higher throughput, with $sim$2$times$ fewer parameters than prior state\-of\-the\-art SFOD methods, thus advancing the Pareto frontier of the speed\-accuracy\-model size trade\-off. We report main results with YOLOv10, and demonstrate generalizability with additional YOLO\- and DETR\-based dual\-head detectors. Code is available here: https://github.com/Sairam13001/RT\-SFOD/
>
> **💻 代码链接：** https://github.com/Sairam13001/RT-SFOD/
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.31834v1)

---

> ### 4. Temporal Preservation over Processing: Diagnosing and Designing Spatiotemporal Single\-Stage Video Detectors
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-06-30 |
> | 👤 作者 | Karam Tomotaki\-Dawoud |
>
> **📄 英文摘要：**
> Single\-stage video object detectors are increasingly deployed in time\-critical applications, yet it remains unclear whether these models genuinely reason over temporal context or merely exploit a single informative frame\-a gap hidden by standard metrics, which reward correct predictions regardless of how they are reached. We address this from two complementary directions: first, we propose TemporalLens, a model\-agnostic diagnostic framework probing temporal dependence through controlled perturbations, structured occlusions, temporal shuffling, redundancy injection, and resolution degradation, revealing whether a detector actually uses information across time. Applied to stacked\-frame 2D detectors and our YOLO\-3D architecture, it exposes behavioural differences invisible to mAP: stacked 2D models collapse when the target frame is removed, while spatiotemporal models recover predictions from earlier frames, a signature of real temporal reliance. Second, we detail YOLO\-3D, a modular real\-time spatiotemporal detector built on YOLOv8, and show that simply preserving temporal depth through the backbone is the dominant performance driver \(\+3.7 pp mAP@50 at 32 frames averaged across scales\). Together, the diagnostics and architecture turn "does this detector reason over time?" into a measurable, actionable question.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.31421v1)

---

> ### 5. Character Recognition of Nepali Number Plate
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-06-27 |
> | 👤 作者 | Satyasa Khadka |
>
> **📄 英文摘要：**
> This paper presents a robust Automatic Number Plate Recognition \(ANPR\) system tailored for Nepali license plates written in Devanagari script. In this paper, a pipelined model was used that integrates YOLO\-based models for license plate and character detection, followed by a CNN classifier trained on 34 Devanagari characters. Two publicly available data sets were used that incorporate diverse lighting, fonts, and structural variations. Data augmentation and additional training on embossed plates enhanced the generalizability of the model. The system achieved a recognition accuracy of up to 93%, demonstrating strong performance under real\-world conditions and providing a scalable solution for traffic management in Nepal. Code: https://github.com/Satyasakhadka/Nepali\-NumberPlate\-Character\-Recognition
>
> **💻 代码链接：** https://github.com/Satyasakhadka/Nepali-NumberPlate-Character-Recognition
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.28946v1)

---

<div align="center">

*由 [YOLO-Arxiv-Daily](https://github.com/WangQvQ/YOLO-Arxiv-Daily) 自动生成*

</div>