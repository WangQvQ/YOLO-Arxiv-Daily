<div align="center">

# YOLO ArXiv Daily

[![Daily Papers](https://img.shields.io/badge/📅-每日更新-blue)]()
[![arXiv](https://img.shields.io/badge/arXiv-最新论文-red)](https://arxiv.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)

*自动追踪 YOLO 相关最新论文，提供中英文双语摘要*

</div>

---

## 📑 论文列表

> ### 1. NEO\-BENCH: A New Multi\-Source Benchmark for Generalizable Astronomical Streak Detection
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-09-06 |
> | 👤 作者 | Jiayou He |
>
> **📄 英文摘要：**
> Near\-Earth Objects \(NEOs\) can appear as faint streaks in long\-exposure astronomical images. Detecting these streaks across diverse observatories requires methods that remain reliable despite differences in image quality, orientation, sky background, and noise. However, existing detectors are commonly evaluated using data from only one source, providing limited evidence of cross\-source generalization.   We introduce NEO\-Bench, a multi\-source benchmark containing 8,376 images from five astronomical\-image datasets. The sources include the Hubble Space Telescope, a Stellina smart telescope, the United Arab Emirates Meteor Monitoring Network, a TETRA1 telescope using a Celestron C14 with Fastar, and the Roboflow Asteroid dataset. We converted the data to a common YOLO format, audited a sample of labels, and defined within\-source and leave\-one\-source\-out evaluation protocols. We evaluated four approaches: Hough, Radon, Gaussian PSF, and YOLO26L.   Leave\-one\-source\-out F1 decreased in 14 of 20 image\-level method\-source pairs and 13 of 20 IoU@0.50 localization pairs. Across the datasets categorized as medium or hard, F1 decreased in 11 of 12 image\-level pairs and 9 of 12 localization pairs. These results show that cross\-source performance remains inconsistent and that reliable generalization across astronomical imaging sources remains an open challenge. The benchmark, code, and data are publicly available.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2609.06774v1)

---

> ### 2. A Cloud\-Based Hybrid Model for Real\-Time Detection of BRTA\-Approved Licence Plates Using YOLO Tiny and Haar Cascade
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-09-06 |
> | 👤 作者 | Debashis Kar Suvra |
>
> **📄 英文摘要：**
> Accurate vehicle license plate detection is essential for applications such as intelligent transportation systems, toll collection, parking management, and law enforcement. In Bangladesh, this task presents distinct challenges due to the complexity of localized license plates and environmental factors like lighting, occlusion, motion blur, and obstructions such as dirt or mud. These challenges often render conventional methods ineffective. This paper introduces a novel hybrid approach, combining the YOLO Tiny deep learning model with the Haar\-Cascade classifier, for enhanced detection and localization of Bengali license plates. A key innovation of our system is the integration of a dynamic retraining pipeline, which allows the model to adapt to evolving real\-world conditions. This retraining mechanism significantly boosts performance in low\-confidence scenarios by continuously improving the model's accuracy as new data is encountered. Additionally, a publicly accessible dataset of BRTA\-compliant license plates, captured under diverse and challenging conditions, has been developed to support this approach. Experimental results demonstrate that our approach not only achieves superior detection accuracy and computational efficiency over conventional models but also ensures consistent performance in resource\-constrained environments, particularly in Bangladesh.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2609.06507v1)

---

> ### 3. Development of a Humanoid Robot Prototype for Multimodal Human\-Robot Interaction
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-09-04 |
> | 👤 作者 | Thang Tran Viet |
>
> **📄 英文摘要：**
> Human\-robot interaction \(HRI\) enables intuitive and intelligent collaboration between humans and robots in real\-world environments. This paper introduces a humanoid robot prototype designed as a flexible testbed for developing and integrating artificial intelligence \(AI\) modules in HRI tasks. The system features a 12 degree\-of\-freedom \(DOFs\) dual\-arm mechanism and a 2 DOFs head with an expressive LCD screen to express facial emotions. All hardware components are controlled by a custom\-designed controller board with real\-time AI processing supported by an onboard Jetson module. The system incorporates three AI modules: \(1\) gesture recognition using MediaPipe Pose and an LSTM classifier, \(2\) object detection with YOLO and 3D localization, and \(3\) voice\-command processing through speech recognition and large language model\(LLM\)\-based semantic parsing. The platform is validated through experiments on positioning accuracy, with results showing average manipulation errors of approximately 1.83 cm. To demonstrate its versatility, experimental results show over 90% task accuracy, with gesture recognition reaching 96%, speech recognition reaching 92%. The results confirm the effectiveness of the proposed system as a reproducible and accessible humanoid platform for research and prototyping in HRI.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2609.05361v1)

---

> ### 4. Hardware\-Accelerated Instance Segmentation for Resource\-Constrained Space Robotics with Criticality Analysis
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

> ### 5. DESA\-TTA: Dynamic EMA and Source Anchoring for Test\-Time Adaptation
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

<div align="center">

*由 [YOLO-Arxiv-Daily](https://github.com/WangQvQ/YOLO-Arxiv-Daily) 自动生成*

</div>