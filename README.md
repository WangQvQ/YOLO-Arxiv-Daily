<div align="center">

# YOLO ArXiv Daily

[![Daily Papers](https://img.shields.io/badge/📅-每日更新-blue)]()
[![arXiv](https://img.shields.io/badge/arXiv-最新论文-red)](https://arxiv.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)

*自动追踪 YOLO 相关最新论文，提供中英文双语摘要*

</div>

---

## 📑 论文列表

> ### 1. HDMamba\-YOLO: Efficient State\-Space Perception and Local Spatial Reconstruction for UAV Small Object
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-09-19 |
> | 👤 作者 | Linduo Wei |
>
> **📄 英文摘要：**
> Small\-object detection in UAV imagery is challenged by weak visual evidence, ambiguous boundaries, dense object distributions, and complex backgrounds. Effective detection therefore requires long\-range contextual information for target\-background discrimination while preserving explicit local two\-dimensional structures for accurate localization. These requirements arise at different stages of the detection pipeline and are not naturally addressed by a uniform feature\-processing strategy. We propose Hybrid Dual\-domain Mamba\-YOLO \(HDMamba\-YOLO\), a stage\-wise heterogeneous SSM\-CNN detector organized according to a perception\-reconstruction\-alignment\-interaction rationale. EfficientVMamba\-based EVSS establishes long\-range contextual perception in the backbone, while PhasePatchMerging2D provides phase\-aware hierarchical transitions. DST\-Wrapper and Native C3k2\-ASSAF then perform perception\-to\-reconstruction transition and repeated local two\-dimensional reconstruction during FPN/PAN aggregation. DySample provides content\-adaptive cross\-scale resampling, while OS\-CVTIA introduces macro\-micro interaction and task\-specific modulation for localization and classification. On VisDrone2019, HDMamba\-YOLO\-B achieves 42.737% mAP50 and 25.713% mAP50:95 with 10.042M parameters and 29.879 corrected GFLOPs. HDMamba\-YOLO\-Lite achieves 41.140% mAP50 and 24.741% mAP50:95 with 5.344M parameters. Under the unified AI\-TOD evaluation protocol, HDMamba\-YOLO\-B obtains 21.621% AP and 47.881% AP50. Controlled ablations further support the stage\-wise allocation of state\-space perception, convolutional reconstruction, dynamic alignment, and task interaction for UAV small\-object detection.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2609.23061v1)

---

> ### 2. Towards Robust Classroom Attendance: A Comprehensive Evaluation of Face Detection and Recognition Models
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-09-19 |
> | 👤 作者 | Himani Trivedi |
>
> **📄 英文摘要：**
> Manual attendance methods, such as paper or register\-based systems, take a lot of time, can lead to errors, and are easy to falsify. Face recognition is more reliable, but it frequently struggles in classrooms because lighting and other conditions can vary. Face recognition datasets are designed for regulated environments and do not capture the actual challenges found in classrooms. To address this, a new face detection and recognition dataset, the Visage Face dataset, comprising 16,234 face samples, is proposed for the task of face detection and recognition. The photos are taken from different angles and under varying lighting conditions, with students showing a range of expressions, and some faces partly covered to reflect real\-life situations. A YOLO\-based system is used to detect faces and tested seven advanced face recognition models with thirteen configurations: LVFace, QCFace, FaceLiVTv2, TopoFR, EdgeFace, TransFace, and GhostFaceNets. Of these, FaceLiVTv2\-M performed best, with 99.75% Top\-1/Top\-5 accuracy and an inference time of 6.459 ms. These results show that the Visage Face Dataset is a realistic and challenging benchmark for face recognition in classroom attendance.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2609.22750v1)

---

> ### 3. From Pixels to Semantics: Edge AI for UAV\-Based Critical Infrastructure Inspection
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

> ### 4. BrainFocus: EEG\-Guided ROI Selection for Efficient Vision\-Language Models
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

> ### 5. BVB: Benchmarking Agentic Video Understanding via Programmatic Reconstruction in Blender
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

<div align="center">

*由 [YOLO-Arxiv-Daily](https://github.com/WangQvQ/YOLO-Arxiv-Daily) 自动生成*

</div>