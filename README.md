<div align="center">

# YOLO ArXiv Daily

[![Daily Papers](https://img.shields.io/badge/📅-每日更新-blue)]()
[![arXiv](https://img.shields.io/badge/arXiv-最新论文-red)](https://arxiv.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)

*自动追踪 YOLO 相关最新论文，提供中英文双语摘要*

</div>

---

## 📑 论文列表

> ### 1. Spiking Neural Networks for Energy\-Efficient Object Detection in Forward\-Looking Sonar Imagery
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-08-22 |
> | 👤 作者 | Gwenevere Frank |
>
> **📄 英文摘要：**
> Autonomous underwater vehicles \(AUVs\) are increasingly important tools in industries ranging from research, to energy, to defense. AUVs are power\-constrained platforms operating in remote environments with fixed battery capacities, where propulsion competes with compute and sensors for power over lengthy mission durations. AUVs frequently operate in dark or turbid waters where optical sensing is of limited value, and rely on sonar as their primary sensing modality. Convolutional neural networks \(CNNs\) are the state\-of\-the\-art solution for object detection in forward\-looking sonar imagery, but are energy expensive \(e.g. YOLOv8m: 322 mJ/inference\). Spiking neural networks \(SNNs\) rely on binary spike activations and thus sparse accumulate\-only operations, allowing them to be remarkably energy efficient, particularly when paired with dedicated neuromorphic hardware. The sparse, high\-contrast structure of forward\-looking sonar \(FLS\) returns is structurally matched to spike coding in a way that optical imagery is not. No prior work has assessed the suitability of SNNs for object detection in FLS imagery. SpikeYOLO, a fully spiking network trained with surrogate gradients, was benchmarked against state\-of\-the\-art CNN baselines on three FLS object detection datasets. Key results: SpikeYOLO T=2 achieves 3.3$times$ lower theoretical compute energy on UATD \(97 vs 322 mJ\) at competitive accuracy \(0.529 mAP@0.5:0.95 vs. YOLOv8m's 0.575\); SpikeYOLO matches YOLOv8m on mAP@0.5 and outperforms YOLO\-SONAR and Fast R\-CNN baselines on the sparse Marine\-Debris\-FLS dataset at 4.4$times$ lower energy; SpikeYOLO demonstrates superior robustness to multiplicative speckle noise \(3.0% degradation at $σ\{=\}0.4$ vs. 8.9% for YOLOv8m\), outperforming YOLOv8m outright at $σ\{=\}0.6$, directly relevant to real\-world FLS deployment.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2608.22072v1)

---

> ### 2. A Modular Agent for Reliable and Auditable Spatial Relation Verification in CT Scans
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-08-21 |
> | 👤 作者 | Simon Vincent Abel |
>
> **📄 英文摘要：**
> Reliable spatial understanding is an important prerequisite for future medical vision\-language systems that aim to support radiological report generation and structured image understanding. While modern vision\-language models \(VLMs\) show promising performance on many medical imaging tasks, recent evidence suggests they remain weak in controlled spatial reasoning and often fail to reliably ground spatial relations in image evidence. Given that radiological reasoning hinges on understanding the relative positions of anatomical structures and findings, this spatial weakness poses risks to diagnostic accuracy. We present a modular medical imaging agent for binary spatial relation verification in axial CT slices. Instead of directly predicting spatial answers end\-to\-end, the system decomposes the task into explicit stages: language parsing, anatomical localization, and deterministic geometric verification. Natural\-language queries are converted into structured relation tuples, queried organs are localized with a YOLO\-based detector, and the final spatial decision is computed from object centers using deterministic geometric rules. We evaluate the approach on the held\-out MIRP spatial QA benchmark and compare it against representative end\-to\-end VLM baselines. The best\-performing hybrid configuration reaches 94.1% accuracy and 94.2% F1, outperforming direct Qwen2\-VL prompting by 42.5 percentage points in accuracy, while preserving interpretable intermediate representations and auditable reasoning stages. The results suggest that explicit modular spatial verification can serve as a promising building block for future report\-oriented medical imaging agents.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2608.21140v1)

---

> ### 3. Radio Galaxies detection and characterization using deep learning techniques
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-08-21 |
> | 👤 作者 | Sanjay Khatik |
>
> **📄 英文摘要：**
> Future radio telescopes will generate data volumes that are increasingly difficult to analyse using traditional statistical methods, motivating the adoption of machine\-learning techniques. In this work, we present YOLO\-Chars \(YOLO\-based Detection and Characterisation of Radio Sources\), a two\-stage deep\-learning framework for the automated detection and characterisation of radio galaxies in survey images. The framework is developed and evaluated using the Square Kilometre Array Science Data Challenge 1 \(SKA SDC1\) dataset. In the first stage, customised YOLO\-based multi\-scale detection models are used to localise compact and extended sources across large sky maps. In the second stage, a dedicated source\-characterisation network estimates the physical properties of the detected sources. We focus on three key parameters: flux density, angular size, and position angle. Our results show that YOLO\-Chars achieves competitive detection and characterisation performance on the SKA SDC1 benchmark, demonstrating its potential as a scalable framework for next\-generation radio continuum surveys.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2608.21474v1)

---

> ### 4. Comparative Study of Out\-of\-the\-Box Technology for Automatic Target Detection and Recognition
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-08-18 |
> | 👤 作者 | Alma M. Liezenga |
>
> **📄 英文摘要：**
> Automatic Target Detection and Recognition \(ATD/R\) is critical for military decision support and \(semi\-\)autonomous operations. Recent advances in object detection and artificial intelligence \(AI\) significantly boosted the potential performance of ATD/R. However, the scarcity of publicly available military datasets limits the application of these systems. As a solution, this paper explores the use of publicly available models and civilian datasets to achieve reasonable performance in military contexts. We benchmark several state\-of\-the\-art models, including six iterations of the YOLO series and two variations on the DETR framework, on a newly acquired military relevant dataset. This dataset features military vehicles and challenging circumstances, including various degrees of occlusions and small targets. The out\-of\-the\-box version of each model is validated alongside a version finetuned on the VisDrone dataset. This dataset features small objects, an Air\-to\-Ground \(A2G\) perspective and relevant classes, potentially generalizing to our military ATD/R task. We compare the performance of the models using mAP@0.5 and mAP@0.5:0.95, across A2G and Ground\-to\-Ground \(G2G\) perspective, target size and model size, giving insight into the real\-time capabilities of models. Our main findings are: \(1\) bigger models outperform smaller models, \(2\) DETR\-based models show promising results compared to the YOLO series,\(3\) fine\-tuning models on an out\-of\-domain A2G dataset, improves their A2G performance and slightly improves their performance on small objects, but \(4\) all models still struggle with detecting small objects in an A2G scenario. We conclude that, despite recent advances in object detection, in\-domain training is still crucial for creating capable ATD/R systems.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2608.17917v1)

---

> ### 5. Continuity\-Driven Representation Learning for Industrial Defect Detection
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-08-18 |
> | 👤 作者 | Minjong Kim |
>
> **📄 英文摘要：**
> Industrial defect detection differs from natural\-image object detection because inspection images are captured under controlled conditions and contain large normal\-dominant regions with repetitive structures. Defects therefore appear as localized disruptions of otherwise predictable patterns, while conventional detectors rely mainly on sparse bounding\-box supervision, resulting in weakly constrained normal\-region representations. We propose a continuity\-driven representation regularization framework that exploits normal\-dominant regions as dense auxiliary supervision. The framework introduces two detector\-agnostic objectives: Multi\-Continuity Loss, which combines 1D patch\-sequence prediction and 2D masked spatial prediction, and Differencing Loss, which regularizes first\-order feature variation and second\-order curvature between neighboring patch embeddings. Both objectives are applied with box\-derived region weighting to stabilize normal\-region representations while preserving defect\-related discontinuities.   Experiments on two real\-world industrial datasets and the public NEU\-DET benchmark, using six detector architectures including YOLO\-family models, MambaYOLO, and DETR, demonstrate consistent improvements over native detector baselines. In the full\-data setting, the proposed regularizers improve average mAP@0.5:0.95 by up to 3.49 percentage points on Industrial Metal, 5.38 percentage points on MEA, and 5.03 percentage points on NEU\-DET. Under limited\-data conditions, the gains become more pronounced, with Differencing Loss achieving improvements of up to 21.07 percentage points in mAP@0.5 and 8.23 percentage points in mAP@0.5:0.95 on NEU\-DET using only 25% of the training data. These results suggest that continuity\-driven regularization provides an effective prior for improving industrial defect detection, particularly when annotated data are scarce.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2608.17362v1)

---

<div align="center">

*由 [YOLO-Arxiv-Daily](https://github.com/WangQvQ/YOLO-Arxiv-Daily) 自动生成*

</div>