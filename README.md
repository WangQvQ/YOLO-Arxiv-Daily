<div align="center">

# YOLO ArXiv Daily

[![Daily Papers](https://img.shields.io/badge/📅-每日更新-blue)]()
[![arXiv](https://img.shields.io/badge/arXiv-最新论文-red)](https://arxiv.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)

*自动追踪 YOLO 相关最新论文，提供中英文双语摘要*

</div>

---

## 📑 论文列表

> ### 1. Query\-Only Prompt\-Agnostic Attacks on Visual In\-Context Learning
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

> ### 2. Keyline Detection with YOLO: Runway Perception for Vision\-Based Aircraft Landing
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

> ### 3. UAV\-Based Computer Vision for Cattle Detection and Feed Bunk Spatial Analysis
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

> ### 4. Computer Vision by YOLOs for Detection of Major Diseases in Vietnamese Custard Apple Using Fruit Imagery
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-11-06 |
> | 👤 作者 | Tri Nhut Do |
>
> **📄 英文摘要：**
> Amid Vietnam rapid digital transformation in agriculture, the integration of artificial intelligence technologies for identifying, monitoring, and predicting crop pests and diseases has emerged as an inevitable trend.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2611.08886v1)

---

> ### 5. Energy\-Aware Open\-Vocabulary Detection Scheduling for Low\-Power Robotic Autonomy
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-11-05 |
> | 👤 作者 | Abdul Basit |
>
> **📄 英文摘要：**
> Open\-vocabulary object detection \(OVD\) lets robots search for run\-time object queries, but fixed\-rate inference spends embedded compute on low\-value frames during idle motion, map revisits, or stable viewpoints. This paper presents an energy\-aware OVD scheduler for low\-power mobile robots. The scheduler uses odometry, map change, and detector confidence to switch among low\-, mid\-, and high\-power YOLO\-World\-S modes on a Jetson Orin Nano. We evaluate the energy\-quality trade\-off on an AgileX LIMO using 40 real navigation trials: two target objects, five placements per target\-policy pair, and four perception policies. Across bottle and cup trials, the scheduler records 8/10 mission success compared with 7/10 for each fixed\-rate baseline, while reducing mean Jetson input power \(a Jetson\-side power\-draw metric\) by 5.2% relative to an 8 Hz high\-rate baseline and lowering achieved OVD call rate by 34.0%. It does not minimize mission\-window Jetson input energy because longer missions can erase instantaneous power savings. A separate 60 s stationary microbenchmark isolates the perception workload: the scheduler lowers mean power from 8.045 W to 7.301 W and energy from 484.7 J to 438.1 J relative to the 8 Hz baseline. These results support robot\-state\-aware OVD scheduling as a promising power\-management mechanism with competitive task reliability, while showing that mission\-window Jetson input energy must be read together with task progress and runtime.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2611.07663v1)

---

<div align="center">

*由 [YOLO-Arxiv-Daily](https://github.com/WangQvQ/YOLO-Arxiv-Daily) 自动生成*

</div>