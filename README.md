<div align="center">

# YOLO ArXiv Daily

[![Daily Papers](https://img.shields.io/badge/📅-每日更新-blue)]()
[![arXiv](https://img.shields.io/badge/arXiv-最新论文-red)](https://arxiv.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)

*自动追踪 YOLO 相关最新论文，提供中英文双语摘要*

</div>

---

## 📑 论文列表

> ### 1. UAV\-Based Computer Vision for Cattle Detection and Feed Bunk Spatial Analysis
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

> ### 2. Computer Vision by YOLOs for Detection of Major Diseases in Vietnamese Custard Apple Using Fruit Imagery
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

> ### 3. Energy\-Aware Open\-Vocabulary Detection Scheduling for Low\-Power Robotic Autonomy
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

> ### 4. Comprehensive Evaluation of Monocular and Multi\-view 3D Human Pose Estimation Models for Fitness Exercise Analysis
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-11-02 |
> | 👤 作者 | Néstor Ortega |
>
> **📄 英文摘要：**
> Three\-dimensional human pose estimation has become a fundamental component of computer vision systems for sports analytics, rehabilitation, and motion understanding. Although recent deep learning approaches have achieved remarkable accuracy on standard benchmarks, their performance under realistic sports scenarios remains insufficiently studied. Furthermore, the trade\-offs between recent monocular and multi\-view approaches, as well as the benefits of post\-processing techniques for improving temporal consistency, have not been systematically evaluated on fitness\-oriented datasets. This paper evaluates recent state\-of\-the\-art 3D human pose estimation methods on Fit3D, a dataset of synchronized multi\-view recordings of a wide variety of fitness exercises with accurate ground\-truth annotations. We study two recent monocular lifting approaches, TCPFormer and PoseMamba\-L, and a multi\-view model, MVGFormer, under a unified evaluation protocol. To investigate the influence of the input data, monocular methods are evaluated using ground\-truth keypoints and detections generated by recent 2D pose estimators, RTMPose and YOLO\-Pose. Additionally, we analyze the effectiveness of three temporal filters for reducing prediction jitter while preserving motion dynamics. Experimental results provide a detailed comparison of accuracy, temporal consistency, and robustness to imperfect 2D detections across the evaluated methods. Our analysis highlights the impact of detector quality on lifting\-based approaches, compares transformer\- and state\-space\-based temporal architectures, and discusses the practical trade\-offs between monocular and multi\-view systems for fitness motion analysis. Our results show that 2D detection quality, temporal modeling, and post\-processing must be considered jointly in pose estimation pipelines for exercise analysis.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2611.02929v1)

---

> ### 5. Attention Residual U\-Net for Enhanced Cross\-Dataset Polyp Localization in Endoscopy
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-11-02 |
> | 👤 作者 | Romil Imtiaz |
>
> **📄 英文摘要：**
> Precise localization of pathologies in endoscopy is a critical task in medical imaging, with deep learning \(DL\) frameworks now serving as the primary methodology for automated lesion detection. Today, generic object detection methods, such as YOLO, provide state\-of\-the\-art performance in this context. However, their robustness often decreases under real\-world clinical conditions due to variations across endoscopic devices, intestinal content, dataset bias, and the presence of small or low\-contrast lesions. Motivated by these challenges, we propose a novel polyp detection framework based on a Attention Residual U\-Net offering enhanced polyp localization. Instead of directly regressing bounding boxes from global feature representations, the proposed method learns lesion\-focused spatial representations and converts them into bounding\-box predictions during inference. Unlike conventional encoder\-decoder localization frameworks that directly propagate all encoder activations into the decoding pathway, the proposed method selectively suppresses irrelevant background responses before feature reconstruction. This encourages the network to focus on lesion\-relevant spatial evidence under heterogeneous imaging conditions. To achieve this, the framework integrates residual feature propagation, attention\-guided encoder filtering, and multi\-scale contextual learning. The proposed framework was evaluated on multiple public endoscopic datasets acquired using different devices. The comparative experimental results validate its effectiveness over the relevant state\-of\-the\-art methods, with an improvement up to 22.0% in terms of the area under the receiving operating characteristic \(AUC\).
>
> 🔗 [阅读论文](http://arxiv.org/abs/2611.02726v1)

---

<div align="center">

*由 [YOLO-Arxiv-Daily](https://github.com/WangQvQ/YOLO-Arxiv-Daily) 自动生成*

</div>