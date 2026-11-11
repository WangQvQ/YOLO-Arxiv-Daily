<div align="center">

# YOLO ArXiv Daily

[![Daily Papers](https://img.shields.io/badge/📅-每日更新-blue)]()
[![arXiv](https://img.shields.io/badge/arXiv-最新论文-red)](https://arxiv.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)

*自动追踪 YOLO 相关最新论文，提供中英文双语摘要*

</div>

---

## 📑 论文列表

> ### 1. When Does Sliced Inference Pay Off? Resolution, Tiling and Dense Fusion for Small Object Detection
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-11-09 |
> | 👤 作者 | Stefan Becker |
>
> **📄 英文摘要：**
> Sliced inference such as SAHI is the standard remedy for small objects in high\-resolution imagery, but its gains are usually measured against low\-resolution baselines. We revisit sliced inference with scale as the explicit variable and ask when tiling actually pays off. We compare seven dense detectors from four families on VisDrone and DOTA, under pre\-specified hypotheses, calibration\-only tuning and a single evaluation on held\-out test data. Two findings emerge. First, much of the benefit attributed to slicing is a resolution effect: on VisDrone, a single pass trained at a higher resolution outperforms tiling with box fusion \(merging decoded boxes\) for every detector that could be trained at that resolution, and at equal compute a higher resolution helps more than a larger model. Second, tiles carry complementary information that box fusion wastes. We introduce SCAHI \(Slicing with Coverage\-Aware Hyper Inference\), which accumulates the dense outputs of all passes with weights that give every image position the same total coverage, and reads objects off a peak grid fine enough to keep crowded neighbours apart. On identical passes, SCAHI outperforms box fusion for every YOLO detector on both datasets and matches it for FCOS. On DOTA, whose images far exceed the resolution a single pass can be trained at, tiles with SCAHI beat the best single pass for every detector, by 4.3 to 5.6 mAP50 at 2048 px, and replacing only SAHI's merge step by SCAHI adds 2.3 to 4.0 mAP50. An ablation attributes the gain to the border weighting and the resolution of the peak grid. Source code is available at https://github.com/beckerio/scahi.
>
> **💻 代码链接：** https://github.com/beckerio/scahi.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2611.12657v1)

---

> ### 2. Compute Regeneration through Re\-Productization: Reorganizing Retired Edge AI Silicon into a Delivered Multilingual Service Robot
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-11-08 |
> | 👤 作者 | Hongyu Luo |
>
> **📄 英文摘要：**
> Useful edge hardware can lose product support while retaining substantial computation. We treat this as systems design: compute regeneration here means re\-productization, retaining the compute module while redesigning the surrounding product organization. Our case is GEdgeBot, a multilingual service robot built around a retired 32 GB Jetson AGX Xavier. A source\-grounded inventory shows the box alone supplies inference, while the combined system exposes five on\-device AI services, a seven\-language speech chain, three sensing domains, and one deterministic seven\-byte control boundary. AI and orchestration stay on the Xavier; finite motor\-pulse generation on an STM32F407. Audits cover the six\-mode control state machine, readiness barriers and sound orientation. Historical single\-prototype measurements report 1,782 MB GPU memory with YOLO, Whisper\-small, and NLLB\-200 resident, a direct\-control p99 deviation rising from 0.24 to 49.93 ms under concurrent AI load, and six of six trigger\-to\-follow demonstrations at 4.35 s mean latency, with conversational voice unavailable. A November campaign restores an isolated inference stack, re\-measures AI workloads, and repeats control\-frame runs. Driving two execution organisations with the same 6,000 measured 50 ms frame intervals shows the finite\-pulse boundary removes host jitter from the pulse train but not the command start phase; its stopping bound is host\-enforced, while the MCU 16\-bit pulse field is unclamped. A six\-check static audit exposes a host\-\-MCU queue mismatch, and a 14\-scenario replay executes the audited main\(\) dispatcher across six modes with zero physical device opens and 54 contract\-checked frames. Results support an existence claim and a reproducible architectural hypothesis, not fleet\-level reliability. The 208 direct\-control ratio remains unreplicated; MCU waveform capture and on\-site fault injection remain open.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2611.11698v1)

---

> ### 3. Query\-Only Prompt\-Agnostic Attacks on Visual In\-Context Learning
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-11-08 |
> | 👤 作者 | Xiao Hu |
>
> **📄 英文摘要：**
> Visual In\-Context Learning \(VICL\) enables promptable segmentation by conditioning RGB predictions on visual demonstrations, making it attractive for auto\-labeling, but its robustness under query\-only access and unknown deployment prompts remains underexplored. We study query\-only, prompt\-agnostic targeted attacks where the attacker observes only returned RGB outputs, without access to model internals or deployment prompts. Treating prompt\-conditioned RGB rendering as a distinct attack surface, we introduce two appearance\-space attacks: Color Hijack \(CH\) drives foreground colors toward the background to suppress visibility, while Task Hijack \(TH\) collapses instance\-distinguishing colors toward a shared color assignment. To transfer across unseen prompts, we propose a Self\-Prompt Surrogate \(SPS\) constructed from the clean query and its output. In a paired, matched\-budget SegGPT comparison, SPS outperforms the Random\-Prompt Surrogate \(RPS\), reducing CH instance foreground retention from 0.918 to 0.155 and TH palette\-aware tolerance distance from 0.345 to 0.225. Across four VICL models on COCO and HierText, CH suppresses foregrounds across semantic\-, instance\-, and text\-style outputs, while TH collapses instance colors under held\-out deployment prompts. After palette\-based decoding, TH also reduces the decoded instance count and mask retention while increasing instance merges. Finally, at 30% corruption, CH\-corrupted annotations from a publicly accessible VICL\-based auto\-labeling pipeline reduce downstream YOLO\-seg mAP50 from 0.761 to 0.281.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2611.11116v1)

---

> ### 4. Keyline Detection with YOLO: Runway Perception for Vision\-Based Aircraft Landing
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-11-06 |
> | 👤 作者 | Leander Fritsch |
>
> **📄 英文摘要：**
> Modern object detectors can predict keypoints as per\-object geometric features, but keypoints are inherently vulnerable to occlusion and truncation. Object edges can provide complementary information, often remaining observable even when partially occluded or when their defining endpoints leave the field of view. We therefore introduce object keyline detection, the task of predicting a fixed, semantically ordered set of infinite lines per object instance. Unlike the outputs of general line\-segment detectors, each keyline is associated with an object instance and identified by a semantic index, providing a line\-level analog of keypoints. To solve this task, we extend YOLO with keyline prediction parameterized in continuous Hesse\-normal form $\[ρ,cosθ,sinθ\]$, and introduce Object Keyline Similarity \(OKLS\), which serves as an evaluation metric, target assignment signal, and regression loss. We instantiate this formulation on runway perception in vision\-based landing, a motivating example containing semantic linear features that remain partially observable throughout the approach. Experiments show that our representation achieves the highest detection accuracy among four evaluated line parameterizations, and keylines integrate directly into standard YOLO heads as additional output channels without degrading keypoint performance or adding meaningful latency.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2611.09316v1)

---

> ### 5. UAV\-Based Computer Vision for Cattle Detection and Feed Bunk Spatial Analysis
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-11-06 |
> | 👤 作者 | Pedro Oliveira Torrente |
>
> **📄 英文摘要：**
> Beef cattle production is central to Brazilian agribusiness and faces growing demands for efficiency, sustainability, and animal welfare. Precision Livestock Farming \(PLF\) offers tools to address these challenges, particularly through monitoring feeding\-related spatial patterns in confined systems. Feeding management plays a decisive role in animal performance, yet its evaluation still relies largely on subjective, labor\-intensive visual inspections that are hard to scale and prone to bias. This work proposes a computer vision framework based on Unmanned Aerial Vehicles \(UAVs\) to quantify cattle alignment and spatial dispersion at the feed bunk under real\-world feedlot conditions. A UAV dataset was collected during flight campaigns at a commercial feedlot in Mato Grosso do Sul, Brazil. Images were manually annotated into cattle head and cattle back classes to support object detection and image\-level spatial analysis of cattle distribution. You Only Look Once \(YOLO\) models, namely YOLOv8 and YOLOv11, were evaluated, with YOLOv8 small used for the downstream spatial analysis. The feed bunk region was segmented using the Segment Anything Model \(SAM\), and geometric metrics based on detected head positions were developed to characterize alignment, spacing regularity, and clustering. The selected detector achieved 84.30% mAP50 and 35.50% mAP50\-95, with an inference time of approximately 2.7 ms per image on an NVIDIA RTX 5060 Ti GPU. The resulting metrics distinguished representative regular and irregular spatial distributions by Root Mean Square Deviation \- RMSD \(1.06 m and 4.26 m\), Coefficient of Variation \- CV \(42.70% and 96.31%\), and clustering \(0 and 2 clusters\), respectively. These findings indicate that the proposed approach can provide indicators for assessing feed bunk spatial patterns in PLF.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2611.08928v1)

---

<div align="center">

*由 [YOLO-Arxiv-Daily](https://github.com/WangQvQ/YOLO-Arxiv-Daily) 自动生成*

</div>