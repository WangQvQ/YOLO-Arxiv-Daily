<div align="center">

# YOLO ArXiv Daily

[![Daily Papers](https://img.shields.io/badge/📅-每日更新-blue)]()
[![arXiv](https://img.shields.io/badge/arXiv-最新论文-red)](https://arxiv.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)

*自动追踪 YOLO 相关最新论文，提供中英文双语摘要*

</div>

---

## 📑 论文列表

> ### 1. From Pixels to Semantics: Edge AI for UAV\-Based Critical Infrastructure Inspection
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-09-16 |
> | 👤 作者 | Reza Farahani |
>
> **📄 英文摘要：**
> Critical infrastructure assets such as bridges, tunnels, dams, and power line networks require timely and scalable inspection. While conventional manual inspection remains costly and hazardous, unmanned aerial vehicle \(UAV\)\-based inspection has emerged as an efficient alternative for monitoring difficult\-to\-access structures. Existing UAV inspection pipelines have evolved from cloud\-centric offline processing toward edge\-based perception using lightweight object detectors such as YOLO for real\- time defect localization. This article explores the transition toward fully edge\-native semantic inspection powered by lightweight vision language models \(VLMs\), where UAVs move beyond object detection toward contextual structural understanding. It categorizes existing UAV inspection architectures, identifies their key system challenges and architectural requirements, and experimentally assesses the feasibility of semantic edge intelligence on NVIDIA Jetson UAV\-class hardware using the COCO\-Bridge dataset. The evaluation integrates a fine\-tuned YOLO\-26M for object localization and a lightweight SmolVLM\-256 for semantic reasoning. Finally, it outlines future directions toward agentic, autonomous, trustworthy, and collaborative semantic UAV inspection across the edge\-cloud continuum.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2609.18448v1)

---

> ### 2. BrainFocus: EEG\-Guided ROI Selection for Efficient Vision\-Language Models
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-09-15 |
> | 👤 作者 | Yihui Peng |
>
> **📄 英文摘要：**
> Vision\-language models \(VLMs\) achieve strong visual question answering \(VQA\) performance, but processing large cluttered images is computationally expensive when only a small region is relevant. Electroencephalography \(EEG\) signals, which capture human neural responses to visual stimuli, can provide a human\-derived semantic cue about the region of interest \(ROI\). However, EEG\-guided visual category decoding remains imperfect, making direct ROI routing unreliable. In this work, we propose BrainFocus, a reliable EEG\-guided efficient VLM framework for VQA. An EEG classifier predicts a target category, and a YOLO detector localizes the matching ROI. The VLM receives the cropped ROI only when both predictions pass confidence thresholds; otherwise, it processes the full image. For evaluation, we build on EEG\-ImageNet to construct a 40\-class benchmark comprising generated cluttered images and real object\-centric images, with target\-ROI annotations and 600 English visual question\-answer pairs. Across Qwen3.5\-VL 2B, 4B, and 9B models, BrainFocus improves VQA accuracy by 4.14\-9.87 percentage points \(pp\) on cluttered scenes while reducing input tokens and total tokens by 23.2%\-39.4% and 23.2%\-39.3%, and end\-to\-end floating\-point operations \(FLOPs\) by 23.2%\-39.5%. These results demonstrate that EEG can guide efficient VLM inference even when its semantic decoding is imperfect.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2609.17443v1)

---

> ### 3. BVB: Benchmarking Agentic Video Understanding via Programmatic Reconstruction in Blender
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-09-14 |
> | 👤 作者 | Yolo Y. Tang |
>
> **📄 英文摘要：**
> Multimodal agents can create complex videos in software such as Blender by coding without relying on diffusion models. Yet video understanding benchmarks still evaluate models mainly through question answering. If an agent truly understands a video, it can reconstruct it programmatically. We introduce BVB, Blender\-VideoBench, a benchmark that tests this ability by asking agents to reconstruct real\-world videos as animated Blender scenes. To ensure fair comparison, each agent programs the reconstruction through a lightweight harness, Mini\-BVB, in an identical sandbox under a shared cost limit. The benchmark renders each reconstruction from its animated camera and evaluates it on two axes: \(1\) Dual VQA measures how many spatiotemporal facts the reconstruction preserves. \(2\) Latent Similarity measures how closely the reconstruction matches the source video perceptually. Our overall score, a square\-root mean, favors balanced performance. We evaluate 51 configurations from 10 model families and analyze semantic retention, perceptual similarity, reasoning effort, and cost. The best model reaches 88.6 Latent Similarity but retains only 53.7% of the source\-correct spatiotemporal answers. Additional reasoning improves visual similarity but does not close this gap in factual accuracy. In a blind study with 15 raters and five configurations, Latent Similarity correlates strongly with human preference. These results show that programmatic reconstruction is a viable test of agentic video understanding, and that semantic retention remains the main challenge.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2609.15478v1)

---

> ### 4. SpermYOLO: A Coordinated YOLO\-Based Detector for Accurate and Efficient Sperm and Impurity Detection in Microscopic Images
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-09-13 |
> | 👤 作者 | Shengqi Chen |
>
> **📄 英文摘要：**
> Accurate sperm detection is essential for computer\-assisted semen analysis, yet it remains challenging in microscopic images due to dense distributions, visually similar artifacts, and sperm\-like impurities. In this paper, we propose SpermYOLO, a coordinated and compact YOLOv11\-derived framework for joint sperm and impurity detection in microscopic images. SpermYOLO introduces four architectural improvements: C3k2\-IDB for channel\-wise discriminative feature extraction, D2SEM for spatial\-\-spectral semantic enhancement, MFM for adaptive multi\-scale feature fusion, and the DESD Head for detail\-enhanced shared prediction. Experiments on the SVIA semen microscopic imaging benchmark show that SpermYOLO achieves 97.2% sperm AP and 75.4% impurity AP, outperforming generic detectors, dedicated sperm detection models, and improved YOLO variants. Compared with the baseline model, SpermYOLO improves sperm AP, impurity AP, $mathrm\{mAP\}\_\{50\}$, and $mathrm\{mAP\}\_\{50:95\}$ by 1.6, 10.0, 5.8, and 2.7 percentage points, respectively, while preserving a lightweight model scale. Cross\-scene evaluation on the SDTB testicular\-biopsy microscopy benchmark shows that SpermYOLO remains effective with extremely small sperm targets and complex tissue backgrounds, achieving the highest $mathrm\{mAP\}\_\{50\}$ and $mathrm\{mAP\}\_\{50:95\}$ of 74.8% and 31.2%, respectively. Ablation studies and qualitative analyses further support these improvements by demonstrating the contributions of the proposed modules and showing more focused feature response patterns than the baseline model. These findings suggest that SpermYOLO is an effective and efficient approach for sperm detection in challenging microscopic imaging scenarios.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2609.14278v1)

---

> ### 5. Quantum\-Gated LiteSSD: A Parameter\-Efficient Lightweight Hybrid Quantum\-Classical Framework for Forward\-Looking Sonar Object Detection
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-09-12 |
> | 👤 作者 | Niloy Kumar Mondal |
>
> **📄 英文摘要：**
> Forward\-looking sonar object detection is essential for underwater perception, yet deployment on embedded platforms requires highly compact models. To address this challenge, we explore quantum computing and introduce Quantum\-Gated LiteSSD, a parameter\-efficient hybrid quantum\-\-classical detector that reformulates QuCNet\-style multi\-circuit quantum processing as an identity\-centered channel\-gating mechanism for spatial feature modulation. Experiments on the Marine Debris Watertank dataset and UATD forward\-looking sonar benchmarks demonstrate an effective parameter\-\-accuracy trade\-off. The proposed detector achieves 90.84% $mathrm\{mAP\}\_\{50\}$ on Watertank with approximately $62times$ fewer parameters than YOLO26s and $164.3times$ fewer parameters than SSD\-VGG16. On UATD, the model achieves 70.37% $mathrm\{mAP\}\_\{50\}$ with only 0.150M parameters, making it approximately $4.1times$ smaller than SSGA\-YOLO while retaining meaningful multi\-class detection capability.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2609.14025v1)

---

<div align="center">

*由 [YOLO-Arxiv-Daily](https://github.com/WangQvQ/YOLO-Arxiv-Daily) 自动生成*

</div>