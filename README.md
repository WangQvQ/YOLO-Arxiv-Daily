<div align="center">

# YOLO ArXiv Daily

[![Daily Papers](https://img.shields.io/badge/📅-每日更新-blue)]()
[![arXiv](https://img.shields.io/badge/arXiv-最新论文-red)](https://arxiv.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)

*自动追踪 YOLO 相关最新论文，提供中英文双语摘要*

</div>

---

## 📑 论文列表

> ### 1. Calibration\-Free Vehicle Speed Estimation: A Monocular Keypoint\-Template Approach
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

> ### 2. Beyond Clear Skies: Synthetic Seasonal and Weather Variations for Real\-World Drone Detection
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

> ### 3. MITE\-Net: SWaP\-Optimized 4K Video Tiny Target Perception for Embodied Edge SAR
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

> ### 4. LightTeaNet: A Weakly Supervised Lightweight CNN for Multi\-Label Tea Leaf Disease Detection and Localization
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-08-14 |
> | 👤 作者 | Naif Haider Chowdhury |
>
> **📄 英文摘要：**
> Tea is known as an important crop in many parts of South and Southeast Asia, yet the production of tea is still hampered by the multiple diseases that decrease the quantity and quality. Traditional methods of inspection, which are manual, are not consistent, labor\-intensive, and depend on extensive monitoring. This paper introduces a lightweight convolutional neural network \(CNN\) designed for weakly supervised multi\-label classification and disease localization in tea leaves called LightTeaNet. LightTeaNet learns directly from image\-level labels and employs Class Activation Mapping \(CAM\) to localize disease\-affected regions automatically, unlike conventional object detection models such as YOLO, which require extensive bounding box annotations. For Parameter efficiency, the network integrates Depthwise Separable Convolutions, and for enhanced feature discrimination, it integrates Channel Attention. LightTeaNet has achieved a Precision of 0.9615, a Recall of 0.8772, and an F1\-score of 0.9179, while it shows mAP@0.50=0.1810 without any manual annotations, which delivers a competitive localization performance in the experimental results. These results validate the model as an interpretable as well as a resource\-efficient framework for intelligent disease monitoring in agriculture.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2608.14178v1)

---

> ### 5. The Role of Natural Language Understanding in Multimodal Video\-Based Dengue Diagnosis
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-08-13 |
> | 👤 作者 | Danial Sharifrazi |
>
> **📄 英文摘要：**
> Detecting infection\-related behavioral changes in mosquitoes from video data is challenging because mosquitoes are small, move rapidly and irregularly, and are affected by environmental factors such as background, lighting, and shadows, which can make reliable feature extraction difficult. In this study, a YOLO\- and Contrastive Language\-Image Pre\-training \(CLIP\)\-based vision\-language framework is proposed to classify mosquito flight frames of uninfected and Dengue virus serotype 2 \(DENV2\)\-infected mosquitoes. First, YOLO is used to isolate mosquito regions from the background. Then, visual features extracted from video frames are aligned with biologically meaningful textual prompts in a shared embedding space. The multimodal model was fine\-tuned using supervised bidirectional contrastive learning and evaluated through frame\-level image\-text similarity\-based classification. The results show that the proposed method achieved 98.54% accuracy and 99.91% sensitivity at the frame level. After temporal aggregation of frame\-level information, the model achieved complete video\-level performance. The ablation results showed that fine\-tuning and CLIP\-based representations were essential for this domain, while the textual branch provided semantic image\-text alignment rather than an accuracy advantage over the vision\-only model. These findings suggest that vision\-language models can provide a useful framework for analyzing infection\-related biological behaviors from video data.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2608.12677v1)

---

<div align="center">

*由 [YOLO-Arxiv-Daily](https://github.com/WangQvQ/YOLO-Arxiv-Daily) 自动生成*

</div>