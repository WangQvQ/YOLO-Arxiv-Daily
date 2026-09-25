<div align="center">

# YOLO ArXiv Daily

[![Daily Papers](https://img.shields.io/badge/📅-每日更新-blue)]()
[![arXiv](https://img.shields.io/badge/arXiv-最新论文-red)](https://arxiv.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)

*自动追踪 YOLO 相关最新论文，提供中英文双语摘要*

</div>

---

## 📑 论文列表

> ### 1. Integrating Local Detail and Global Context: A Dual\-Input Multi\-Task Learning Framework for Bone Tumor Diagnosis
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-09-23 |
> | 👤 作者 | S. M. Nasif Uddin |
>
> **📄 英文摘要：**
> Primary bone tumors are rare but clinically aggressive neoplasms whose diagnosis from radiographs is challenged by heterogeneous morphology, subtle lesion margins, and overlapping bone structures. To address the limitations of existing single\-view models, we present a dual\-input, multi\-task learning framework that, to our knowledge, is the first to apply bidirectional cross\-modal attention between a lesion crop and the full radiograph for joint segmentation and subtype classification. Using the multi\-institutional Bone Tumor X\-ray Radiograph Dataset \(BTXRD, n=3,746\), we employ a YOLO\-based detector to generate regions of interest, which are paired with full images as inputs to a dual\-stream DenseNet121 architecture. Features are integrated via a novel cross\-modal attention fusion strategy, refined by Hierarchical Multi\-scale Feature Fusion, effectively balancing fine\-grained lesion detail with global anatomical context. Evaluated on a held\-out patient\-level test split, the model demonstrates superior performance over single\-input baselines, achieving an overall Dice Similarity Coefficient of 0.896 and a macro\-averaged classification F1\-score of 0.928. Notably, the system exhibits exceptional sensitivity for malignant osteosarcoma \(AUC 0.999\), validating the potential of dual\-stream context modeling to support radiologists in accurate, early decision\-making.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2609.28732v1)

---

> ### 2. A 3D Pose\-Based Ensemble Framework for Cricket Shot Classification and Automated Biomechanical Analysis
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-09-22 |
> | 👤 作者 | Sourav Shome |
>
> **📄 英文摘要：**
> Cricket is one of the most celebrated sports world\-wide, and technological advancement has become deeply embedded in how the modern game is analyzed and coached. Cricket shot classification and automated performance analysis add a further dimension to this trend. Traditional approaches rely on RGB video features or static images, which are sensitive to environmental variations such as camera angle, lighting, and background clutter, and often fail to capture the underlying biomechanics of batting actions. In this paper, we propose a system to improve cricket coaching that takes raw video data, extracts batsmen from video frames using YOLO, and extracts 3D pose data from video frames using MeTRAbs. The system produces sequential skeletal pose data of 30 body points and captures the biomechanical features of a batsman. As part of the system, we also propose a deep learning ensemble for shot classification of four shots: flick, pull, defense, and drive. The ensemble performed well, compared to existing classification works, achieving 97.68% accuracy. In addition, we analyzed the misclassification rates to identify cases where shots were incorrectly classified and examined their possible causes. Our proposed system allows novice players to obtain useful feedback, such as important joint angles relative to expert batsmen, which can also be useful for injury prevention. The shot classifier also helps track class\-wise shots over time for further analysis. In addition to novice players, coaches can use the system for player evaluation.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2609.26923v1)

---

> ### 3. mbariml: a curation pipeline for turning deep\-sea imagery and video into object\-detection training data
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-09-21 |
> | 👤 作者 | Lonny Lundsten |
>
> **📄 英文摘要：**
> Training data quantity and quality greatly affect object detection model performance, regardless of model architecture. When using object detection models on video and images from the deep sea, in which the objects of interest, primarily organisms, are sparse, faint, and hard to identify, incremental improvements to object detector performance may require an iterative approach to data labeling and management. This paper presents mbariml, a python\-based video and image analysis pipeline built around the data labeling management process. mbariml uses an Ultralytics YOLO detection model, runs it over still images or video, stores every detection as a reviewable region of interest, groups those regions by visual similarity so that a human can accept or reject them in bulk, and exports the result as training data, statistics, image sidecars, and additional metadata. The human review stage is the centre of the design: an annotator can validate, relabel, resize, delete, and draw entirely new localizations, and every one of those edits is written back to the same database the detector wrote to. Video receives particular attention: the software treats each tracker\-produced track as a provisional observation and selects one representative frame instead of retaining every detection in the track. We describe the pipeline stage by stage, including the operational middle\-third heuristic used for track observation selection.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2609.25500v1)

---

> ### 4. NPU Accelerator: Quantized Real\-Time Vehicle Detection on PYNQ\-Z1 Using FINN
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-09-21 |
> | 👤 作者 | Daniel Gutierrez |
>
> **📄 英文摘要：**
> This paper presents the design, optimization, implementation, and on\-board validation of a neural processing unit \(NPU\) accelerator for real\-time vehicle detection on the resource\-constrained Xilinx Zynq XC7Z020 device of the PYNQ\-Z1 board. The work follows a hardware/software co\-design methodology that combines quantization\-aware training \(QAT\), lightweight YOLO\-derived detectors, Brevitas/QONNX model export, FINN dataflow compilation, Vivado implementation, and physical benchmarking on the target board. Four simultaneous engineering requirements define successful deployment: throughput above 30 frames/s \(FPS\), energy efficiency above 7 FPS/W, programmable\-logic \(PL\) hardware latency below 50 ms, and Pascal VOC detection accuracy above 0.55 mAP@0.5. The design space includes LP\-YOLO and LP\-YOLO Slim variants, a custom YOLOv3\-tiny reference, 4\-bit and mixed low\-bit quantization, 320$times$320 and 256$times$256 inputs, manual and automatic FIFO sizing, and programmable\-logic clocks from 100 to 200 MHz. The final LP\-YOLO Slim configuration uses a 256$times$256 input, w2a4 quantization, and a 142.86 MHz PL clock. With batch 100 it reaches 35.66 FPS at 2.91 W, corresponding to 12.25 FPS/W, while measured PL latency is 45.11 ms and VOC mAP@0.5 is 0.594. This is the only evaluated configuration for which the supplied measurements satisfy all four requirements simultaneously. The results show that low\-bit QAT, architectural slimming, FINN folding and FIFO optimization, and moderate clock scaling can jointly provide a practical real\-time detector on a small Zynq FPGA.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2609.24757v1)

---

> ### 5. Ev\-YOLO: Uncertainty\-Aware Object Detection via a Unified Evidential Formulation
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-09-21 |
> | 👤 作者 | Simon Barbarit\-Gaboriau |
>
> **📄 英文摘要：**
> Reliable uncertainty estimation is essential for deploying object detectors in autonomous systems operating in uncertain environments. Evidential Deep Learning \(EDL\) provides a principled framework for uncertainty\-aware classification by representing network outputs as evidence and interpreting predictions through subjective logic. However, existing evidential object detectors typically combine evidential classification with regression uncertainty models that do not share the same theoretical foundation. In this work, we propose an evidential version of YOLOv8 in which both classification and bounding\-box regression are formulated within a common evidential framework. Our approach exploits YOLOv8's distribution\-based bounding\-box representation, allowing the evidential formulation to be applied not only to classification but also to localisation. As a result, both tasks produce belief, uncertainty, and probability estimates that can be interpreted within the Dempster\-\-Shafer framework. Experiments on KITTI, MUSES, and nuScenes show that the resulting detector remains broadly competitive with standard YOLOv8 in terms of detection accuracy while providing a localisation uncertainty that effectively discriminates between correct and erroneous detections. Moreover, this uncertainty becomes increasingly discriminative under domain shift.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2609.24668v1)

---

<div align="center">

*由 [YOLO-Arxiv-Daily](https://github.com/WangQvQ/YOLO-Arxiv-Daily) 自动生成*

</div>