<div align="center">

# YOLO ArXiv Daily

[![Daily Papers](https://img.shields.io/badge/📅-每日更新-blue)]()
[![arXiv](https://img.shields.io/badge/arXiv-最新论文-red)](https://arxiv.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)

*自动追踪 YOLO 相关最新论文，提供中英文双语摘要*

</div>

---

## 📑 论文列表

> ### 1. REMIND: RE\-Identification with Memory for INDoor Navigation
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

> ### 2. LDFE: Laplacian Decoupled Feature Enhancement Block for Dual\-Stream CNN\-based RGB\-IR Object Detection
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

> ### 3. HAJJv2\-CrowdCount: Zero\-Shot Benchmark for Dense Crowd Counting
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

> ### 4. Evaluating Vision\-Language Models as a Zero\-Shot Learning Alternative to You Only Look Once and Optical Character Recognition for Nigerian License Plate Recognition
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-07-02 |
> | 👤 作者 | Ismail Ismail Tijjani |
>
> **📄 英文摘要：**
> License Plate Recognition \(LPR\) systems are critical tools in traffic monitoring, security enforcement, and urban mobility management. Traditional LPR systems often rely on a multi\-stage pipeline involving object detection using You Only Look Once \(YOLO\) and Optical Character Recognition \(OCR\), which suffer from limitations such as high resource demands, poor performance in unstructured environments, and the need for large annotated datasets. This study explores the potential of Vision\-Language Models \(VLMs\) as a unified, zeroshot learning solution for Nigerian license plate recognition. Using a curated dataset of 88 challenging real\-world images collected in Nigeria, we evaluate five selected VLMs: Gemini 2.0 Flash Exp \(Google DeepMind\), Qwen2.5\-VL\-7B\-Instruct \(Alibaba\), GPT\-4o \(OpenAI\), Claude 4 Sonnet \(Anthropic\), and Llama 3.2 Vision 90b \(Meta\). Results based on Character Error Rate \(CER\) reveal that Gemini and Qwen significantly outperform other models in both accuracy and robustness, on the challenging image scenarios. This work highlights the practical advantages of VLMs over YOLO\+OCR, questions the claims by model providers, and compares the performances of the VLMs.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2607.02025v1)

---

> ### 5. Computer Vision for Wildlife Monitoring: Detecting Brown Howler Monkeys using YOLO
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-07-01 |
> | 👤 作者 | Gabriel Ferri Schneider |
>
> **📄 英文摘要：**
> Urban expansion threatens global biodiversity, especially affecting arboreal species due to the fragmentation of forest habitats. The movement of arboreal species across disjointed forest patches increases mortality risk and, thus, compromises their conservation. In this context, the installation of canopy bridges can be a viable strategy; yet continuous monitoring of their use by arboreal species is essential for ensuring their effectiveness, typically carried out with the aid of camera traps. However, this method often produces false\-positive images that demand time from conservationists for review. In this context, computer vision algorithms can optimize the task of detecting target species using the canopy bridges. In this study, we explored the automatic detection of brown howler monkeys \(Alouatta guariba\) in videos obtained by camera traps. Given the need for a large number of annotated images of the target animals to train the algorithms, we tested the incorporation of auxiliary data to improve detection models, fine\-tuning the YOLOv10 framework using varying proportions of them. The improvement of these automatic detection techniques contributes to conservation efforts, by providing automatic tools to monitor solutions that minimize the impact of human interference in animals habitats.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2607.01396v1)

---

<div align="center">

*由 [YOLO-Arxiv-Daily](https://github.com/WangQvQ/YOLO-Arxiv-Daily) 自动生成*

</div>