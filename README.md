<div align="center">

# YOLO ArXiv Daily

[![Daily Papers](https://img.shields.io/badge/📅-每日更新-blue)]()
[![arXiv](https://img.shields.io/badge/arXiv-最新论文-red)](https://arxiv.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)

*自动追踪 YOLO 相关最新论文，提供中英文双语摘要*

</div>

---

## 📑 论文列表

> ### 1. Comparative Study of Out\-of\-the\-Box Technology for Automatic Target Detection and Recognition
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

> ### 2. Continuity\-Driven Representation Learning for Industrial Defect Detection
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

> ### 3. Calibration\-Free Vehicle Speed Estimation: A Monocular Keypoint\-Template Approach
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-08-17 |
> | 👤 作者 | Gaofeng Su |
>
> **📄 英文摘要：**
> This paper proposes a calibration\-free framework for reliably and effectively estimating vehicle speeds from monocular videos, without relying on roadway features, camera calibration, or roadway\-feature\-based reference objects. The proposed framework estimates vehicle speeds using a 36\-keypoint vehicle template and a homography matrix updated at each frame. A YOLO\-based keypoint detection module is trained on diverse datasets, and two estimation strategies are compared: keypoint\-only tracking and warped optical flow with dense spatial aggregation. Speed is estimated by projecting displacements into metric space using the homography, with validation conducted on over 400 video clips from roadside and overhead datasets, covering speeds from 30 to 100 mph. The method achieves reliable speed estimation on the VS13 and BrnoCompSpeed datasets, with the warped optical flow method delivering MAEs of 15.0% and 9.7%, respectively, and 77.9% and 93.1% of estimates falling within \+/\-20% error. After applying a 10% trim to remove edge\-of\-frame outliers, performance improves to MAEs of 11.7% and 7.6%, with within\-\+/\-20% accuracy increasing to 85.3% and 95.4%. This work addresses key limitations of existing vision\-based approaches and enables low\-cost and efficient speed enforcement using portable devices such as dashcams and smartphones, thereby supporting citizen\-based enforcement programs for traffic safety.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2608.16785v1)

---

> ### 4. Beyond Clear Skies: Synthetic Seasonal and Weather Variations for Real\-World Drone Detection
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-08-17 |
> | 👤 作者 | Tamara R. Lenhard |
>
> **📄 英文摘要：**
> Reliable drone detection under real\-world deployment conditions requires training data that spans the full operational design domain, including adverse weather and seasonal appearance variation. However, acquiring and annotating such data at scale remains highly resource\-intensive, as adverse\-weather conditions are inherently difficult to control, reproduce, and sample systematically. Existing datasets therefore typically provide only limited coverage of such conditions. Conversely, synthetic data offers a scalable alternative: environmental variation becomes controllable, while modern game\-engine\-based pipelines provide realistic rendering and automatic annotations. Leveraging this potential, we introduce SynDroneVision\-Weather \(SDV\-W\), an systematic extension of SynDroneVision \(SDV\) targeting adverse\-weather and seasonal domain shifts in urban drone detection. SDV\-W comprises 55,187 annotated high\-resolution images from three urban environments, rendered across three seasonal configurations and diverse weather conditions, including rain, snow, and fog at multiple severity levels. By preserving SDV's scene and trajectory configuration, SDV\-W enables matched clean\-adverse comparisons and quantification of condition\-specific detector degradation. Across representative YOLO models and real\-world datasets, we show that SDV\-W improves detector reliability under adverse appearance shifts, reduces missed detections and false alarms, and is most effective as a complement to general\-purpose synthetic drone\-detection data. SDV\-W will be publicly released upon paper acceptance.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2608.16191v1)

---

> ### 5. MITE\-Net: SWaP\-Optimized 4K Video Tiny Target Perception for Embodied Edge SAR
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-08-16 |
> | 👤 作者 | Mingshuo Xu |
>
> **📄 英文摘要：**
> Real\-time tiny target perception in high\-resolution imagery is critical for embodied Search\-and\-Rescue \(SAR\) missions. However, strict Size, Weight, and Power \(SWaP\) constraints on edge devices like UAVs create a bottleneck: traditional image downsampling causes severe feature loss, while slice\-based processing incurs prohibitive latency. To address this gap, this paper introduces a comprehensive framework encompassing a novel architecture, specialized datasets, and hardware\-level benchmarks. First, we propose MITE\-Net, a SWaP\-optimized cascaded architecture, which couples a bio\-inspired, learning\-free Tiny Target Motion\-Based Region Proposal Network \(TTM\-RPN\) with a sub\-0.14M\-parameter R\-CNN\-like head. Second, to standardize 4K tiny target evaluation, we construct the SAR\-Tiny Datasets by relabeling two challenging UAV datasets: SeaDroneSee\-Tiny \(dynamic maritime scenes, tiny targets predominantly of 64\-256 pixels \) and UAVID\-Tiny \(cluttered urban scenes, extremely tiny targets, less than 64 pixels\). Third, we benchmark against state\-of\-the\-art YOLO models on an edge device, NVIDIA Jetson AGX Xavier, where MITE\-Net directly processes 4K maritime imagery, achieving a 100% search success rate at 30.33 FPS. Consuming merely 3.19 W \(9.51 FPS/W\), MITE\-Net vastly outperforms YOLO baselines in target recall and energy efficiency. Conversely, UAVID\-Tiny evaluations expose a compound structural limitation: the learning\-free bionic front\-end struggles against urban backgrounds, while the ultra\-lightweight head lacks representational capacity for complex features. Ultimately, this work delivers an efficient onboard perception paradigm and a rigorous baseline guiding future end\-to\-end SAR architectures.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2608.15830v1)

---

<div align="center">

*由 [YOLO-Arxiv-Daily](https://github.com/WangQvQ/YOLO-Arxiv-Daily) 自动生成*

</div>