<div align="center">

# YOLO ArXiv Daily

[![Daily Papers](https://img.shields.io/badge/📅-每日更新-blue)]()
[![arXiv](https://img.shields.io/badge/arXiv-最新论文-red)](https://arxiv.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)

*自动追踪 YOLO 相关最新论文，提供中英文双语摘要*

</div>

---

## 📑 论文列表

> ### 1. Towards Hierarchical Structure Understanding of Newspaper Images
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-07-16 |
> | 👤 作者 | William Mocaër |
>
> **📄 英文摘要：**
> Understanding newspaper images remains a challenging task due to their complex, nested hierarchical structures and dense, heterogeneous layouts. In this paper, we explore two complementary approaches for newspaper structure understanding. First, we present a modular bottom\-up pipeline that combines state\-of\-the\-art open\-source models: YOLO for layout detection, LayoutReader for reading order prediction, and a custom algorithm for article segmentation. This approach leverages existing robust components while maintaining flexibility and interpretability. Second, we introduce Tiramisu \(Tiered Transformers for Hierarchical Structure Understanding\), a novel end\-to\-end transformer\-based architecture that explicitly models document hierarchy through an iterative tiered process. Tiramisu performs section and article separation, block localization, semantic categorization, and reading order prediction using highly parallelized attention mechanisms. Finally, we release Finlam La Liberté, a new dataset designed specifically for evaluating hierarchical information retrieval in historical newspapers. Experimental results demonstrate the effectiveness of both approaches in reconstructing complex newspaper hierarchies, with comparative analysis highlighting their respective strengths for scalable document digitization. The Tiramisu training code, including the synthetic newspaper generator, is available at https://git.litislab.fr/tiramisu/tiramisu\-newspaper\-articles\-extractor.
>
> **💻 代码链接：** https://git.litislab.fr/tiramisu/tiramisu-newspaper-articles-extractor.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2607.15082v1)

---

> ### 2. An Intelligent\-Cloud Edge Multimodal Interaction System for Robots
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-07-16 |
> | 👤 作者 | Zihan Guo |
>
> **📄 英文摘要：**
> Robust human\-robot interaction in complex environments requires accurate gesture perception, semantic scene understanding, and reliable task planning under limited onboard computing resources. This paper presents a cloud\-edge multimodal interaction framework that integrates an enhanced YOLO\-based gesture detector with coordinated large language model \(LLM\) and vision\-language model \(VLM\) agents. The proposed detector, incorporates the Convolutional Block Attention Module \(CBAM\) into the neck and replaces the baseline bounding\-box regression objective with Distance\-IoU \(DIoU\) loss. These modifications improve feature discrimination and localization for small or partially occluded gestures in complex backgrounds. The cloud layer performs gesture detection, scene understanding, multimodal fusion, and action planning, whereas the TonyPi robot locally handles data acquisition, communication, action execution, and feedback. Experiments on a public gesture dataset and a custom dataset show that YOLO\-DC achieves precision values of 98.9% and 95.0%, with mAP@0.5 values of 90.7% and 92.7%, respectively. System\-level evaluation yields success rates of 95%, 88%, and 82% for single\-action, composite\-action, and vision\-dependent tasks. A 30 participant evaluation yields an overall mean satisfaction score of 3.69 out of 5. These results demonstrate the feasibility of combining refined gesture detection with multimodal agents for resource\-constrained robotic interaction.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2607.14675v1)

---

> ### 3. Cotton\-SF YOLO: Learning Structural and Frequency Cues for Early Cotton Square Detection in Complex Field Environments
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-07-16 |
> | 👤 作者 | Chengjia Zhang |
>
> **📄 英文摘要：**
> Cotton squares are important phenotypic indicators of the early reproductive growth of cotton, and automatic field detection of cotton squares provides an important basis for cotton growth monitoring and precision cultivation management. However, early cotton square detection in complex field environments remains insufficiently explored, as cotton squares are small, frequently occluded, easily blurred, subject to illumination variations, and exhibit low contrast against surrounding cotton leaves. To address these challenges, we propose a task\-oriented framework based on YOLO26m, named Cotton\-SF YOLO, for cotton square detection under natural field conditions. To improve the perception of small and irregular cotton square boundaries, we introduce Dynamic Snake Convolution into the detector, enabling adaptive extraction of deformable edge features. Furthermore, a frequency\-domain feature modulation module is designed by incorporating spectral enhancement into the C2f structure, which recalibrate frequency\-domain representations and strengthen discriminative edge and texture cues while reducing interference from complex cotton leaf backgrounds. Trained and evaluated on our newly constructed and annotated field dataset with manually annotated cotton squares, the proposed model achieves mAP$\_\{50\}$, mAP$\_\{50:95\}$, and recall values of 0.8196, 0.4942, and 0.7939, improving over the baseline YOLO26m by 1.25%, 3.45%, and 2.96%, respectively. Ablation experiments and visualization demonstrate that the best performance is achieved with the complementary effects of structural and frequency cues.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2607.14445v1)

---

> ### 4. C\-Norm: Cell\-Distribution Normalization Enables Precision Recognition of Medical\-Cell Image
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-07-14 |
> | 👤 作者 | Yang Qianl |
>
> **📄 英文摘要：**
> ThinPrep Cytologic Test \(TCT\) enables early cervical cancer screening, but manual reading is time\-consuming and yields inconsistent diagnostic results among cytopathologists. Existing AI detection models perform poorly under real clinical conditions, primarily restricted by two key constraints: unbalanced spatial distribution of cell populations in TCT slides, and limited high\-quality annotated cytology data relying on professional pathologist labeling. To address these limitations, we propose a Cell\-Distribution Normalization \(C\-Norm\) method. By decoupling abnormal and normal cells from the original TCT images and re\-synthesizing them, this method ensures a uniform distribution of cell populations, thereby mitigating generalization degradation caused by distribution bias. Building upon this, we integrate the YOLOv12 framework with a DINOv3 module. This hybrid architecture leverages the advanced detection capability of YOLO models and the superior feature representations of DINOv3 to capture subtle morphological nuances essential for precise recognition of TCT images. Extensive experiments demonstrate that our proposed method achieves state\-of\-the\-art performance, significantly outperforming mainstream detection algorithms. The complete implementation is available at: https://github.com/ddw2AIGROUP2CQUPT/Cell\-Norm
>
> **💻 代码链接：** https://github.com/ddw2AIGROUP2CQUPT/Cell-Norm
>
> 🔗 [阅读论文](http://arxiv.org/abs/2607.13116v1)

---

> ### 5. Autonomous Tracking and Terminal Guidance of Moving Targets for Fixed\-Wing UAVs
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-07-14 |
> | 👤 作者 | Wei\-Hao Liou |
>
> **📄 英文摘要：**
> This study introduces a unified control framework for fixed\-wing unmanned aerial vehicles \(UAVs\) fitted with a pan\-tilt \(PT\) camera, intended to perform an end\-to\-end mission spanning from initial target detection to accurate terminal engagement. The proposed system employs a three\-phase strategy: a vision\-based target acquisition phase, an NMPC\-based tracking phase, and a terminal guidance phase. During tracking, the framework uses an Unscented Kalman Filter \(UKF\) to fuse YOLO\-based visual detections with inertial measurements, enabling robust target state estimation under unknown dynamics. To ensure reliable visual contact, we introduce a constraint\-aware Nonlinear Model Predictive Control \(NMPC\) strategy that incorporates Control Barrier Functions \(CBFs\) to explicitly prevent UAV self\-occlusion \-\- a common limitation in fixed\-wing tracking. Upon satisfying terminal engagement conditions, the system seamlessly transitions control to a quaternion\-based Biased Proportional Navigation Guidance \(BPNG\) law, enforcing precise impact angle constraints. High\-fidelity simulations demonstrate that the framework achieves stable, robust tracking and accurate terminal interception while strictly respecting the vehicle's dynamic limits and camera field\-of\-view constraints.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2607.12801v1)

---

<div align="center">

*由 [YOLO-Arxiv-Daily](https://github.com/WangQvQ/YOLO-Arxiv-Daily) 自动生成*

</div>