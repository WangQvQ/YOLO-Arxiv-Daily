<div align="center">

# YOLO ArXiv Daily

[![Daily Papers](https://img.shields.io/badge/📅-每日更新-blue)]()
[![arXiv](https://img.shields.io/badge/arXiv-最新论文-red)](https://arxiv.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)

*自动追踪 YOLO 相关最新论文，提供中英文双语摘要*

</div>

---

## 📑 论文列表

> ### 1. MicroCharNet: Less is More for License Plate Character Detection
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-07-13 |
> | 👤 作者 | Huy Che |
>
> **📄 英文摘要：**
> License plate character detection is a crucial component of intelligent transportation systems, where high accuracy and computational efficiency are required for real\-time deployment. Although recent deep learning\-based methods have substantially improved detection performance, many high\-accuracy models rely on large\-scale architectures that incur substantial computational overhead, limiting their applicability to resource\-constrained devices. In this paper, we propose MicroCharNet, an ultra\-lightweight model specifically designed for license plate character detection. The proposed architecture employs a compact backbone composed of C2f blocks, integrated with CoordAtt module to enhance feature extraction while preserving spatial information. A lightweight C3k2\-based neck fuses multi\-level features, followed by a single\-level anchor\-free detection head that enables end\-to\-end prediction. Experiments conducted on the UFPR\-ALPR dataset demonstrate that MicroCharNet achieves competitive detection accuracy with only 0.08M parameters and 0.096 GFLOPs, while outperforming several recent YOLO\-based baselines. Hardware\-level evaluations further confirm its efficiency for real\-time deployment on edge devices. These results indicate that carefully designed ultra\-lightweight architectures can effectively balance accuracy and efficiency in license plate character detection. The source code is available at https://github.com/chequanghuy/MicroCharNet.
>
> **💻 代码链接：** https://github.com/chequanghuy/MicroCharNet.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2607.11830v1)

---

> ### 2. Confidence Scores in Open\-Vocabulary Detection Are a Biased Mixture of Scale and Semantics
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-07-13 |
> | 👤 作者 | Yi Tang Soon |
>
> **📄 英文摘要：**
> Foundation models such as CLIP have enabled open\-vocabulary object detectors that generalise to novel categories via vision\-language similarity. However, the confidence scores these detectors produce are not reliable localization probability estimates: they conflate visual scale and semantic query specificity with the true detection signal. Through controlled experiments on COCO across three foundation\-model\-based detectors \(GroundingDINO, OWL\-ViT, YOLO\-World\), with the scale\-bias finding further replicated on LVIS \(1,203 categories\) using GroundingDINO, we show that s=cos\(v,t\) is a biased mixture of two effects. Scale bias \(alpha = \+0.064, r = 0.579, p = 1.29 x 10^\-58\) systematically inflates scores for large objects. Semantic bias \(beta = \-0.705, p = 5.23 x 10^\-41\) suppresses scores for generic queries. Both biases are structurally inevitable from CLIP's image\-level pretraining. Threshold adjustment cannot remove them: oracle per\-scale thresholding yields Delta F1 = \+0.001 for small objects versus \+0.102 for large. A parameter\-free temperature scaling correction improves small\-object Recall@10 by 19.6% \(p < 0.01\) without retraining. This comes at a modest, measurable cost to pooled\-ranking precision, so the bias is partially, not freely, reversible at inference time. These findings reveal a fundamental limitation of adapting image\-level foundation models to region\-level detection tasks.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2607.10993v1)

---

> ### 3. REMIND: RE\-Identification with Memory for INDoor Navigation
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-07-10 |
> | 👤 作者 | Pablo Diaz\-Pereda |
>
> **📄 英文摘要：**
> Mobile robots operating indoors must re\-identify previously observed objects after long temporal gaps, significant viewpoint changes, and severe illumination variations. This remains a challenging problem: multi\-object tracking methods are optimized for short\-term association of pedestrians and vehicles at video rates, person and vehicle re\-identification approaches lack persistent memory mechanisms, and state\-of\-the\-art video object segmentation techniques rely on reactive distractor filtering rather than enforcing global identity consistency.   To address these limitations, we present REMIND, an online tracker designed for long\-term multi\-object re\-identification of generic indoor objects from monocular RGB imagery, requiring neither camera pose nor depth. Motivated by evidence from visual cognition that humans rely on accumulated appearance familiarity and spatial context rather than explicit self\-localization, REMIND combines frozen DINOv3 features with a dual\-bank multi\-prototype appearance memory, part\- and background\-level descriptors, a neighbour\-context reasoning module exploiting spatial co\-occurrence, and joint Hungarian assignment with ambiguity\-aware safeguards. On a purpose\-built indoor dataset featuring controlled revisits and dense same\-class clutter, REMIND reaches 90.35% IDF1, nearly 20 points above a state\-of\-the\-art video object segmentation baseline and more than 36 above a strong tracking\-by\-detection baseline. On ScanNet\+\+, it attains the highest IDF1 in every setting but one, end\-to\-end detection over all scenes, where the tracking\-by\-detection baseline is marginally ahead while REMIND still associates and recovers identities more accurately; it also completes every scene, whereas the video object segmentation baseline exhausts GPU memory on 66.9% under YOLO detections. The complete system, evaluation framework, and dataset are publicly released.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2607.09267v1)

---

> ### 4. LDFE: Laplacian Decoupled Feature Enhancement Block for Dual\-Stream CNN\-based RGB\-IR Object Detection
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-07-09 |
> | 👤 作者 | Wenhao Dong |
>
> **📄 英文摘要：**
> The complementary information between RGB and IR images can significantly enhance object detection performance under extreme conditions. Existing methods prefer dual\-stream CNN backbones built upon YOLO for feature extraction and focus on the design of feature fusion. In this paper, we introduce the Laplacian Decoupled Feature Enhancement block \(LDFE\) to fuse features from different stages of the dual\-stream CNN backbone. By design, LDFE simultaneously considers the characteristics of modalities and structures for feature fusion by employing global\-local decomposition, denoising, fusion, and reconstruction, sequentially. The LDFE first separates features into global and local components based on Laplacian Pyramid, and then performs denoising and fusion based on Global State Space Enhancement module \(GS2E\) and Local Convolutional Correlation Enhancement module \(LC2E\) separately. Specifically, the GS2E conducts a two\-branch architecture for the main and auxiliary modalities. It dynamically suppresses noise in the main modality through cross\-modal attention derived from the auxiliary modality, while employing a State Space Model to capture long\-range dependencies within the global feature representations of the main modality. To obtain bidirectional interaction, the two modalities systematically alternate their main/auxiliary roles. Moreover, the LC2E suppresses noise in local features and leverages spatial and channel dimension along with triple convolution to extract fine\-grained details for fusion. These innovative designs achieve a significant performance improvement, with mAP surpassing the SOTA methods 6.2%, 3.7%, 4.7%, 2.3%, 4.1% and 2.0% on M3FD, DroneVehicle, LLVIP, FLIR\-Aligned, KAIST and VEDAI datasets,respectively.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2607.08076v1)

---

> ### 5. HAJJv2\-CrowdCount: Zero\-Shot Benchmark for Dense Crowd Counting
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-07-08 |
> | 👤 作者 | Reem AlYabis |
>
> **📄 英文摘要：**
> Automated crowd counting in Hajj video is difficult not because current models lack capacity, but because the footage violates the assumptions those models were built on: cameras observe the crowd from steep, near\-vertical angles, individuals occlude one another extensively, and a single frame can contain well over a thousand people. Benchmarks that test crowd counting in such an environment are either private or not detailed per second. We revisit the HAJJv2 dataset and contribute HAJJv2\-CrowdCount: per\-second human\-annotated crowd counts for its testing videos. Using these annotations, we benchmark three recent zero\-shot counting paradigms: an open\-vocabulary detector \(YOLO\-World\), a point\-based counter \(APGCC\), and a promptable segmentation\-based counter \(SAM3Count\). SAM3Count attains the lowest overall mean absolute error \(MAE 70.4, 95% CI 56.0\-86.1\), ahead of YOLO\-World \(92.0\) and APGCC \(152.9\). This ordering reverses, however, in the regime most relevant to deployment: on the densest frames, the detection\- and segmentation\-based counters both degrade sharply \(MAE exceeding 300\), while the point\-based counter degrades far more gracefully \(MAE 114.9\). This inversion is decision\-relevant for Hajj crowd management, where reliable counts are needed most precisely in the densest and most occluded scenes. The annotations are released to support reproduction and extension of these results.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2607.07322v1)

---

<div align="center">

*由 [YOLO-Arxiv-Daily](https://github.com/WangQvQ/YOLO-Arxiv-Daily) 自动生成*

</div>