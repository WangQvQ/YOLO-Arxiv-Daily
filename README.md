<div align="center">

# YOLO ArXiv Daily

[![Daily Papers](https://img.shields.io/badge/📅-每日更新-blue)]()
[![arXiv](https://img.shields.io/badge/arXiv-最新论文-red)](https://arxiv.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)

*自动追踪 YOLO 相关最新论文，提供中英文双语摘要*

</div>

---

## 📑 论文列表

> ### 1. mbariml: a curation pipeline for turning deep\-sea imagery and video into object\-detection training data
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

> ### 2. NPU Accelerator: Quantized Real\-Time Vehicle Detection on PYNQ\-Z1 Using FINN
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

> ### 3. Ev\-YOLO: Uncertainty\-Aware Object Detection via a Unified Evidential Formulation
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

> ### 4. Infectious Bovine Pinkeye Detection Using Computer Vision and Imbalance\-Aware Learning
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-09-20 |
> | 👤 作者 | Michael Abalo |
>
> **📄 英文摘要：**
> Infectious bovine pinkeye is a contagious ocular disease that adversely affects cattle health, welfare, and agricultural productivity. Conventional diagnosis relies primarily on clinical observation, which can be subjective, time\-consuming, and difficult to implement efficiently in large herds or remote settings. This study evaluated and compared You Only Look Once \(YOLO\) v11 and YOLOv26 for automated bovine pinkeye classification and investigated the effects of class\-balancing strategies on model performance. Five variants \(n, s, m, l, and x\) of each architecture were trained and evaluated using the original imbalanced dataset, Random Minority Oversampling \(RMO\), and an adapted Synthetic Minority Oversampling Technique \(SMOTE\). Both YOLOv11 and YOLOv26 demonstrated strong classification performance, although the effects of class balancing varied across model variants. For YOLOv11, RMO\-s achieved an accuracy of 0.99, a macro F1\-score of 0.98, and a true positive rate \(TPR\) of 1.00, with no false\-negative classifications. RMO\-m also achieved a TPR of 1.00 with no false negatives. For YOLOv26, the original l, RMO\-m, and RMO\-l variants each achieved an accuracy of 0.99 and a macro F1\-score of 0.98, with RMO\-l attaining a TPR of 1.00 and no false negatives. Overall, RMO generally provided greater improvements in minority\-class detection than adapted SMOTE, whereas the strong performance of the original YOLOv26\-l demonstrates that oversampling was not necessary for all model variants. These findings demonstrate the potential of YOLOv11 and YOLOv26 for automated detection of bovine pinkeye and support further evaluation for livestock health monitoring.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2609.23714v1)

---

> ### 5. HDMamba\-YOLO: Efficient State\-Space Perception and Local Spatial Reconstruction for UAV Small Object
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

<div align="center">

*由 [YOLO-Arxiv-Daily](https://github.com/WangQvQ/YOLO-Arxiv-Daily) 自动生成*

</div>