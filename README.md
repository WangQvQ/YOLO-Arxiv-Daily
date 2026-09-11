<div align="center">

# YOLO ArXiv Daily

[![Daily Papers](https://img.shields.io/badge/📅-每日更新-blue)]()
[![arXiv](https://img.shields.io/badge/arXiv-最新论文-red)](https://arxiv.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)

*自动追踪 YOLO 相关最新论文，提供中英文双语摘要*

</div>

---

## 📑 论文列表

> ### 1. ScopeMamba\-YOLO: Widening the Perceptual Scope Inward and Outward for Small Object Detection in Remote Sensing Imagery
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-09-09 |
> | 👤 作者 | Junjie Fan |
>
> **📄 英文摘要：**
> Small object detection in unmanned aerial vehicle \(UAV\) and remote sensing imagery requires preserving high\-resolution detail while modeling long\-range context. Adding a stride\-4 detection level and removing the stride\-32 stage benefits tiny targets but weakens peripheral spatial support, whereas directly inserting selective scanning into the main feature path can interfere with weak local cues. We propose ScopeMamba\-YOLO, built around an off\-path, zero\-gated selective\-scanning principle that decouples contextual modeling from the convolutional stream. The principle is instantiated by a Cascaded Global\-Context Module \(CGCM\) in the backbone and a Selective\-Scan PAN \(SS\-PAN\) in the neck. An Adaptive Multi\-scale Strip \(AMS\) Block reduces the cost of high\-resolution feature extraction, while a Scale\-Adaptive DFL \(SA\-DFL\) head reallocates distributional support and regression capacity across scales with only 0.008M additional parameters. Controlled experiments show that matched main\-path selective scanning reduces mAP50 by 0.98 pp, whereas off\-path CGCM improves the final configuration by 0.67 pp over the three\-seed no\-CGCM mean; operator controls indicate that this gain is not explained by auxiliary branch capacity alone. ERF analysis further shows that the complete context pathway increases the peripheral energy ratio from 0.008 to 0.090 at stride 8. On VisDrone\-2019, ScopeMamba\-S achieves 50.8% mAP50 with 3.57M parameters, exceeding YOLOv8s by 10.8 pp while using 32% of its parameters; ScopeMamba\-M reaches 52.6% mAP50 with 6.48M parameters. Consistent improvements are also observed on AI\-TOD, especially for very\-tiny and tiny objects.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2609.10156v1)

---

> ### 2. Vague2Detect: Handling Ambiguous Prompts in Knowledge\-Based Open\-World Detection
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-09-09 |
> | 👤 作者 | Ibrohimjon Muminov |
>
> **📄 英文摘要：**
> Real\-world detectors must often interpret functional or ambiguous prompts, yet conventional models such as YOLO remain restricted to fixed class lists. Even open\-vocabulary models like YOLO\-World frequently misalign vague language with the intended objects. Building on our prior work Commonsense\-Guided Open\-World Object Detection Using LLMs and Visual\-Semantic Matching, we address YOLO\-World's limitations in grounding task\-driven queries. We propose Vague2Detect, a hybrid pipeline in which a fine\-tuned Sentence\-BERT retrieves candidates from a structured household Knowledge Base \(KB\), and YOLO\-World verifies their presence in the image. For prompts outside the KB, a large language model \(GPT\-3.5\-turbo\) generates candidate descriptions, dynamically expanding the KB to cover novel concepts. On a benchmark of household scenes using custom images and an Open Images V7 subset, YOLO\-World alone achieves only 32% Vague Prompt Success Rate \(VPSR\), the ability to map ambiguous queries to correct detections. In contrast, Vague2Detect improves performance to 61% VPSR with high precision, and up to 85% when augmented with GPT fallback.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2609.09949v1)

---

> ### 3. NEO\-BENCH: A New Multi\-Source Benchmark for Generalizable Astronomical Streak Detection
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

> ### 4. A Cloud\-Based Hybrid Model for Real\-Time Detection of BRTA\-Approved Licence Plates Using YOLO Tiny and Haar Cascade
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

> ### 5. Development of a Humanoid Robot Prototype for Multimodal Human\-Robot Interaction
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

<div align="center">

*由 [YOLO-Arxiv-Daily](https://github.com/WangQvQ/YOLO-Arxiv-Daily) 自动生成*

</div>