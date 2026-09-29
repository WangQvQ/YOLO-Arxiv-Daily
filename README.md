<div align="center">

# YOLO ArXiv Daily

[![Daily Papers](https://img.shields.io/badge/📅-每日更新-blue)]()
[![arXiv](https://img.shields.io/badge/arXiv-最新论文-red)](https://arxiv.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)

*自动追踪 YOLO 相关最新论文，提供中英文双语摘要*

</div>

---

## 📑 论文列表

> ### 1. A Multi\-Dataset Benchmark of YOLO\-Based Weed Detection in Precision Agriculture
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-09-27 |
> | 👤 作者 | Hristina Zdraveska |
>
> **📄 英文摘要：**
> Weed detection is an important component of precision agriculture, enabling site\-specific weed management and reducing unnecessary herbicide use. Although deep learning methods have achieved strong results for crop and weed detection, many studies rely on single\-dataset evaluation, making it difficult to assess robustness across different agricultural domains. This paper presents a multi\-dataset benchmark of deep object detectors for weed detection in precision agriculture, with a focused evaluation of YOLO26 models. We evaluate nano, small, and medium variants on seven public weed\-detection datasets covering different crops, weed species, field conditions, acquisition setups, and annotation protocols. The models are compared in terms of detection accuracy, model complexity, inference latency, FPS, and model size. In addition to in\-dataset evaluation, we investigate cross\-domain generalization using a unified one\-class weed setup and evaluate multi\-source training using the combined training subsets from all datasets. The results show that YOLO26 achieves strong in\-dataset performance, with YOLO26m obtaining the highest average accuracy and YOLO26s providing the best practical accuracy\-efficiency trade\-off. However, cross\-domain performance decreases substantially, with YOLO26s dropping from an average in\-domain mAP$\_\{50:95\}$ of 0.603 to 0.148 in the off\-domain setting. Multi\-source training improves performance on several datasets, but does not fully eliminate domain shift. Overall, the benchmark highlights the importance of dataset diversity, domain similarity, and target\-domain adaptation for robust weed detection in real\-world precision agriculture applications.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2609.33991v1)

---

> ### 2. Synthetic Thermal Image Generation for Real\-Time Animal Detection Under Low\-Visibility Conditions
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-09-26 |
> | 👤 作者 | James Momoh |
>
> **📄 英文摘要：**
> Wildlife\-vehicle collisions remain a significant road safety concern, particularly during nighttime and low\-visibility conditions when RGB\-based perception systems are often unreliable. Thermal imaging offers a promising alternative for detecting animals under poor illumination. However, the limited availability of annotated infrared animal datasets restricts the development of robust deep learning\-based detection models. This paper investigates synthetic thermal image generation as a scalable approach for real\-time animal detection under low\-visibility conditions. A subset of 514 annotated visible\-spectrum animal images from the NTLNP dataset is translated into synthetic thermal representations using CycleGAN\-Turbo, while a limited real thermal dataset of 60 images is expanded through thermal\-focused augmentation. Multiple object detection architectures, including YOLOv8, YOLOv9, YOLOv10, and RT\-DETR, are trained independently on synthetic and real thermal datasets and evaluated using precision, recall, mAP@0.5, mAP@0.5:0.95, model size, and inference latency. Experimental results show that synthetic thermal images provide competitive detection performance, with RT\-DETR achieving the highest synthetic\-data mAP@0.5 of 0.9613. Models trained on augmented real thermal data achieve the strongest overall performance, with YOLOv10s obtaining 0.9879 mAP@0.5 and 0.9571 mAP@0.5:0.95. Computational analysis further indicates that lightweight YOLO variants provide favorable inference latency, supporting their potential for real\-time deployment. These findings demonstrate that synthetic thermal imagery can reduce dependence on scarce infrared datasets and support the development of efficient animal detection systems for future vehicle\-mounted wildlife collision mitigation applications.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2609.32944v1)

---

> ### 3. Attribution Gaps in Zero\-Training LLM\+OVOD Pipelines: A Fine\-Grained Analysis of the CAAP\-\-SNAP Discrepancy
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-09-26 |
> | 👤 作者 | Yu\-Feng Yen |
>
> **📄 英文摘要：**
> LAOD and similar zero\-training LLM\+open\-vocabulary\-detector \(OVOD\) pipelines score two things separately: class\-agnostic localization accuracy \(CAAP\) and semantic naming accuracy \(SNAP\). The two consistently diverge, and nobody has asked why. This paper asks why, on the full 5,000\-image COCO\-Val split \(27,273 detections\) rather than the small subset the original work evaluated on. Object visual complexity turns out not to be the driver \-\- small and occluded objects are, if anything, localized better than large ones. Vocabulary novelty is: once the LLM's wording falls outside the detector's native category set, localization accuracy falls from 80.9% to 31.6%. That drop is not spread evenly across unfamiliar phrasing, though. Almost all of it comes from cases where the novel wording actually names a different object than the one COCO annotated \(true synonyms still score 89.3%; semantically unrelated "noise" labels score 12.0%\). A closer look at a further failure subset tells a similar story: 78\-88% of what looks like complete localization failure is really the model correctly finding a real object that COCO's non\-exhaustive 80\-category scheme simply never labeled, not hallucination. Swap the detector backbone \(YOLO\-World for Grounding DINO\) or the LLM \(Gemma\-3 for Qwen2.5\-VL\) and both the effect and its rough size hold up, so this looks like a general property of the pipeline family rather than a quirk of one model pairing. The upshot is that a large share of the apparent CAAP\-\-SNAP gap traces back to closed\-category annotation limits rather than a real grounding failure, which matters for how we detect hallucination, analyze failure modes, and design evaluation for grounded multimodal systems meant to work in the open world.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2609.32567v1)

---

> ### 4. Integrating Local Detail and Global Context: A Dual\-Input Multi\-Task Learning Framework for Bone Tumor Diagnosis
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-09-23 |
> | 👤 作者 | S. M. Nasif Uddin |
>
> **📄 英文摘要：**
> Primary bone tumors are rare but clinically aggressive neoplasms whose diagnosis from radiographs is challenged by heterogeneous morphology, subtle lesion margins, and overlapping bone structures. To address the limitations of existing single\-view models, we present a dual\-input, multi\-task learning framework that, to our knowledge, is the first to apply bidirectional cross\-modal attention between a lesion crop and the full radiograph for joint segmentation and subtype classification. Using the multi\-institutional Bone Tumor X\-ray Radiograph Dataset \(BTXRD, n=3,746\), we employ a YOLO\-based detector to generate regions of interest, which are paired with full images as inputs to a dual\-stream DenseNet121 architecture. Features are integrated via a novel cross\-modal attention fusion strategy, refined by Hierarchical Multi\-scale Feature Fusion, effectively balancing fine\-grained lesion detail with global anatomical context. Evaluated on a held\-out patient\-level test split, the model demonstrates superior performance over single\-input baselines, achieving an overall Dice Similarity Coefficient of 0.896 and a macro\-averaged classification F1\-score of 0.928. Notably, the system exhibits exceptional sensitivity for malignant osteosarcoma \(AUC 0.999\), validating the potential of dual\-stream context modeling to support radiologists in accurate, early decision\-making.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2609.28732v1)

---

> ### 5. A 3D Pose\-Based Ensemble Framework for Cricket Shot Classification and Automated Biomechanical Analysis
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-09-22 |
> | 👤 作者 | Sourav Shome |
>
> **📄 英文摘要：**
> Cricket is one of the most celebrated sports world\-wide, and technological advancement has become deeply embedded in how the modern game is analyzed and coached. Cricket shot classification and automated performance analysis add a further dimension to this trend. Traditional approaches rely on RGB video features or static images, which are sensitive to environmental variations such as camera angle, lighting, and background clutter, and often fail to capture the underlying biomechanics of batting actions. In this paper, we propose a system to improve cricket coaching that takes raw video data, extracts batsmen from video frames using YOLO, and extracts 3D pose data from video frames using MeTRAbs. The system produces sequential skeletal pose data of 30 body points and captures the biomechanical features of a batsman. As part of the system, we also propose a deep learning ensemble for shot classification of four shots: flick, pull, defense, and drive. The ensemble performed well, compared to existing classification works, achieving 97.68% accuracy. In addition, we analyzed the misclassification rates to identify cases where shots were incorrectly classified and examined their possible causes. Our proposed system allows novice players to obtain useful feedback, such as important joint angles relative to expert batsmen, which can also be useful for injury prevention. The shot classifier also helps track class\-wise shots over time for further analysis. In addition to novice players, coaches can use the system for player evaluation.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2609.26923v1)

---

<div align="center">

*由 [YOLO-Arxiv-Daily](https://github.com/WangQvQ/YOLO-Arxiv-Daily) 自动生成*

</div>