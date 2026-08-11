<div align="center">

# YOLO ArXiv Daily

[![Daily Papers](https://img.shields.io/badge/📅-每日更新-blue)]()
[![arXiv](https://img.shields.io/badge/arXiv-最新论文-red)](https://arxiv.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)

*自动追踪 YOLO 相关最新论文，提供中英文双语摘要*

</div>

---

## 📑 论文列表

> ### 1. TriView\-YOLO: Early Multi\-View Fusion for Ground Penetrating Radar Cavity Detection in Soft, High\-Water\-Content Soils
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-08-10 |
> | 👤 作者 | Suphawut Thawinutchokaudom |
>
> **📄 英文摘要：**
> Automated detection of subsurface cavities from Ground Penetrating Radar \(GPR\) is most difficult in soft, high\-water\-content ground, where conductive, water\-saturated soil attenuates the signal and degrades cavity reflections, yet this is also the condition under which cavities most readily form. This paper proposes TriView\-YOLO, a multi\-view YOLOv12 detector for road cavity screening in such ground. Three co\-registered views \(longitudinal B\-scan, horizontal C\-scan, and cross\-section B\-scan\) form a 9\-channel input fused by a TripleInputConv layer that replaces the YOLOv12 stem; the rest of the network is unchanged, and bounding boxes are required on the longitudinal view only. Training used 1,600 expert\-verified field samples, principally metropolitan road surveys of Bangkok, Thailand, acquired with a vehicle\-mounted multichannel three\-dimensional GPR mobile mapping system, with surveys over the firmer subgrades of Japan added to training and validation only. The test set comes exclusively from the Bangkok surveys, over soft marine clay with 80\-140% water content and a water table at 1\-2 m depth, a ground condition for which no dedicated deep learning cavity\-detection evaluation has been reported. On this unaugmented, field\-only test set, split randomly within surveys, the proposed model attains mAP50 of 0.558 \+/\- 0.028 over three seeds at 23.6 GFLOPs and 3.1 ms per image. Ablations show that removing the auxiliary views lowers mAP50 and recall, whereas public and synthetic training images, DINOv3 features, larger model scale, and COCO pretraining bring no gain.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2608.09522v1)

---

> ### 2. SkySeaLand: A Wide\-Format Satellite Transportation Benchmark with an Ultra\-Lightweight Detection Baseline
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-08-07 |
> | 👤 作者 | Md. Zahid Hasan Riad |
>
> **📄 英文摘要：**
> Satellite object detection is challenged by small targets and wide\-format scenes that lose detail under standard square\-input resizing. We introduce SkySeaLand, a public dataset of 1,307 high\-resolution satellite images and 19,101 verified bounding boxes across airplane, boat, car, and ship classes in terrestrial and maritime scenes. Native COCO and YOLO annotations are provided. The collection is dominated by large source images and wide scene geometry: 84.5 percent exceed 3,836 pixels on the longest side and 73.1 percent are near a 3:1 aspect ratio. We evaluate twelve detectors from the YOLO, RT\-DETR, DETR, and Faster R\-CNN families using a common split and COCO metrics. The tested YOLO and RT\-DETR variants obtain 84.4\-\-88.2 mAP50, with no consistent accuracy gain from larger parameter counts under the reported model\-specific recipes. We also report SkyDet, a 1.22 M parameter anchor\-free baseline that obtains 60.5 mAP50 and 24.32 mAP50\-95 in a 4.90 MB footprint, with 13.74 ms latency \(72.8 FPS\) on a Tesla T4. SkySeaLand provides a compact benchmark for mixed land\-\-maritime transportation detection, while SkyDet establishes a documented low\-footprint reference rather than a state\-of\-the\-art accuracy claim.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2608.07382v1)

---

> ### 3. YOLO\-PEFT: Parameter\-Efficient Fine\-Tuning on YOLO Family
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-08-07 |
> | 👤 作者 | Xu Lin |
>
> **📄 英文摘要：**
> Generic parameter\-efficient fine\-tuning \(PEFT\) methods transferred from language models can fail silently on real\-time detectors, whose heterogeneous operators and detection\-specific components impose placement constraints absent from regular Transformer stacks. We propose YOLO\-PEFT, a structure\-aware framework that formulates adapter placement as an auditable constraint\-planning problem. Given a detector graph, a PEFT request, and a resource budget, YOLO\-PEFT assigns operator and semantic roles, evaluates explicit operator\-validity, detector\-semantic, graph\-interface, and deployment predicates, records a reason code for each excluded module, and either emits a budgeted target\-module plan or returns Refuse before training. Under the official VOC07\+12 trainval\-to\-VOC07 test protocol, planner\-selected RS\-LoRA reaches 0.7138 and 0.7307 mAP50\-95 on YOLO11s and YOLO12s, respectively, compared with 0.6428 and 0.6662 for Full\-SFT. On RT\-DETR\-L, all seven evaluated LoRA\-family configurations cross the predefined catastrophic threshold, supporting a calibrated Refuse\-to\-Full\-SFT decision within the evaluated coverage. A controlled YOLO11 audit further shows that LoRA reduces peak training memory by 43.9 percent, although training takes 1.72 times longer. Within the evaluated detector families, placement policies, and calibration coverage, YOLO\-PEFT replaces manual target\-module trial and error with explicit, inspectable planning while preserving verified train\-save\-merge\-export paths; refusal on unseen detector architectures remains an open validation problem. Project Page: github.com/Tencent/YOLO\-Master
>
> 🔗 [阅读论文](http://arxiv.org/abs/2608.07051v1)

---

> ### 4. From Transparent Labware Segmentation to Collision Avoidance: A Real\-Time Edge\-Aware Perception Pipeline
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-08-05 |
> | 👤 作者 | Shijun Ding |
>
> **📄 英文摘要：**
> This paper presents an edge\-aware instance segmentation framework that enables real\-time robotic collision avoidance with transparent laboratory glassware using purely visual perception. Transparent vessels defy conventional segmentation due to refraction, specular reflection, and the absence of stable interior texture, yet their boundary contours remain comparatively reliable visual cues. Exploiting this observation, we augment a one\-stage real\-time instance segmentation backbone with a lightweight edge\-detection branch, edge\-guided attention fusion, and a parameter\-free SimAM module, and further construct LabGlass\-IS, a 3485\-image, 21\-category instance segmentation dataset of real laboratory glassware. The enhanced model achieves the highest Boundary F\-score of 97.80 among compared methods, outperforming the YOLO\-prompted FastSAM framework by 18.93 BF points. Furthermore, it maintains an inference speed of 7.1ms per frame and requires only 2.85% of the parameters of the closest accuracy competitor. Multi\-view triangulation of mask centroids further provides 3D positions for conservative bounding\-volume collision constraints. Real\-robot trials achieve a 93.3% collision avoidance success rate, indicating the feasibility of the proposed perception\-to\-action pipeline for robot collision avoidance among fragile transparent objects. Our code is available at https://github.com/havishamy/TransYOLO\_3D. Our video is available at https://havishamy.github.io/paper\-videos/.
>
> **💻 代码链接：** https://github.com/havishamy/TransYOLO_3D.，https://havishamy.github.io/paper-videos/.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2608.04769v1)

---

> ### 5. Simile Understanding in Text\-to\-Image Models: An Evaluation Framework
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-08-05 |
> | 👤 作者 | Luecheng Wang |
>
> **📄 英文摘要：**
> Similes provide a compact and expressive way to describe visual characteristics in text prompts. Recent text\-to\-image models \(t2i models\) can produce visually compelling outputs from simile prompts, yet even frontier models frequently misinterpret the metaphorical vehicle and confuse it with the object. These systematic failures reveal a gap between figurative language and object\-level visual grounding in t2i models. To investigate this issue, we propose a scalable evaluation framework for simile understanding. Our framework includes \(1\) a controlled simile dataset in which metaphorical vehicles are drawn from a predefined set of object\-detectable categories and combined with diverse templates, \(2\) automatic grounding metrics based on YOLO \(You Only Look Once\) detection, and \(3\) text encoder layer analysis using Diffusion Lens to track how metaphorical vehicles emerge during generation. Experiments across architecturally diverse t2i models reveal consistent literalization failure patterns. We further discuss potential mitigation strategies for improving simile grounding in t2i models.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2608.04750v1)

---

<div align="center">

*由 [YOLO-Arxiv-Daily](https://github.com/WangQvQ/YOLO-Arxiv-Daily) 自动生成*

</div>