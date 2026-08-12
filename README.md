<div align="center">

# YOLO ArXiv Daily

[![Daily Papers](https://img.shields.io/badge/📅-每日更新-blue)]()
[![arXiv](https://img.shields.io/badge/arXiv-最新论文-red)](https://arxiv.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)

*自动追踪 YOLO 相关最新论文，提供中英文双语摘要*

</div>

---

## 📑 论文列表

> ### 1. A Comparative Evaluation of Deep Learning Object Detection Models on a Real\-World Multi\-Plant Dataset from Africa
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-08-11 |
> | 👤 作者 | Ismail Ismail Tijjani |
>
> **📄 英文摘要：**
> The application of computer vision in agriculture has shown significant potential for improving crop monitoring and precision farming. However, many existing approaches rely on controlled datasets that do not adequately represent realworld farming conditions, particularly in underrepresented regions such as Africa. This study presents a comparative evaluation of six object detection models YOLOv5, YOLOv8, YOLO11, YOLO26, Faster R\-CNN, and RT\-DETR using a real\-world dataset, AgriAISeg 1 , collected manually from Nigerian farms. AgriAISeg comprises 3,382 images of sesame, cabbage, and tomato crops captured under varying environmental conditions, including changes in illumination, occlusion, and viewing perspectives. Models were trained, and performance was assessed using precision, recall, mAP@0.5, and mAP@0.5:0.95. The results show that RT\-DETR achieved the highest overall performance with a precision of 0.768 and mAP@0.5:0.95 of 0.624, while YOLOv8 and YOLO11 also demonstrated strong and consistent performance. In contrast, Faster R\-CNN recorded significantly lower accuracy, with an overall mAP@0.5 of 0.466, indicating reduced effectiveness under complex field conditions. In addition, YOLO\-based models exhibited superior training efficiency compared to Faster R\-CNN.These findings demonstrate that modern one\-stage and transformer\-based detectors provide more reliable and efficient solutions for plant detection in realworld agricultural environments.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2608.11053v1)

---

> ### 2. MammoMix: Leveraging Mixture of Experts for Robust Mammogram Breast Detection
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-08-11 |
> | 👤 作者 | Dinh Tan Nguyen |
>
> **📄 英文摘要：**
> Breast lesion detection in mammography remains a challenging task due to variations in image quality, lesion appearance, and population demographics across datasets. While current object detectors such as YOLO and DETR achieve strong results on individual datasets, their performance often degrades when trained on or applied across heterogeneous sources. To address this, we propose MammoMix, a novel framework based on Mixture\-of\-Experts \(MoE\) paradigm for robust and generalizable lesion detection. In MammoMix, each expert model is trained on a specific domain, allowing it to specialize in distinct characteristics of its source data. A gating mechanism adaptively weighs contributions from each expert based on input image, combining their outputs to enable domain\-adaptive inference. To improve reliability, we further incorporate a calibration module, MoCAE, which adjusts confidence scores to reflect true predictive uncertainty. We evaluate MammoMix on 3 public mammography datasets: CSAW, DDSM, and DMID, covering diverse clinical settings. Results show that MammoMix outperforms baseline detectors in both average precision and reliability, particularly on datasets with greater variability. Our findings demonstrate that expert specialization and calibrated ensemble fusion significantly enhance model generalization and robustness. MammoMix offers a promising step toward dependable AI\-assisted breast cancer screening across real\-world clinical domains.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2608.10437v1)

---

> ### 3. TriView\-YOLO: Early Multi\-View Fusion for Ground Penetrating Radar Cavity Detection in Soft, High\-Water\-Content Soils
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-08-10 |
> | 👤 作者 | Suphawut Thawinutchokaudom |
>
> **📄 英文摘要：**
> Automated detection of subsurface cavities from Ground Penetrating Radar \(GPR\) is most difficult in soft, high\-water\-content ground, where conductive, water\-saturated soil attenuates the signal and degrades cavity reflections, yet this is also the condition under which cavities most readily form. This paper proposes TriView\-YOLO, a multi\-view YOLOv12 detector for road cavity screening in such ground. Three co\-registered views \(longitudinal B\-scan, horizontal C\-scan, and cross\-section B\-scan\) form a 9\-channel input fused by a TripleInputConv layer that replaces the YOLOv12 stem; the rest of the network is unchanged, and bounding boxes are required on the longitudinal view only. Training used 1,600 expert\-verified field samples, principally metropolitan road surveys of Bangkok, Thailand, acquired with a vehicle\-mounted multichannel three\-dimensional GPR mobile mapping system, with surveys over the firmer subgrades of Japan added to training and validation only. The test set comes exclusively from the Bangkok surveys, over soft marine clay with 80\-140% water content and a water table at 1\-2 m depth, a ground condition for which no dedicated deep learning cavity\-detection evaluation has been reported. On this unaugmented, field\-only test set, split randomly within surveys, the proposed model attains mAP50 of 0.558 \+/\- 0.028 over three seeds at 23.6 GFLOPs and 3.1 ms per image. Ablations show that removing the auxiliary views lowers mAP50 and recall, whereas public and synthetic training images, DINOv3 features, larger model scale, and COCO pretraining bring no gain.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2608.09522v1)

---

> ### 4. SkySeaLand: A Wide\-Format Satellite Transportation Benchmark with an Ultra\-Lightweight Detection Baseline
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-08-07 |
> | 👤 作者 | Md. Zahid Hasan Riad |
>
> **📄 英文摘要：**
> Satellite object detection is challenged by small targets and wide\-format scenes that lose detail under standard square\-input resizing. We introduce SkySeaLand, a public dataset of 1,307 high\-resolution satellite images and 19,101 verified bounding boxes across airplane, boat, car, and ship classes in terrestrial and maritime scenes. Native COCO and YOLO annotations are provided. The collection is dominated by large source images and wide scene geometry: 84.5 percent exceed 3,836 pixels on the longest side and 73.1 percent are near a 3:1 aspect ratio. We evaluate twelve detectors from the YOLO, RT\-DETR, DETR, and Faster R\-CNN families using a common split and COCO metrics. The tested YOLO and RT\-DETR variants obtain 84.4\-\-88.2 mAP50, with no consistent accuracy gain from larger parameter counts under the reported model\-specific recipes. We also report SkyDet, a 1.22 M parameter anchor\-free baseline that obtains 60.5 mAP50 and 24.32 mAP50\-95 in a 4.90 MB footprint, with 13.74 ms latency \(72.8 FPS\) on a Tesla T4. SkySeaLand provides a compact benchmark for mixed land\-\-maritime transportation detection, while SkyDet establishes a documented low\-footprint reference rather than a state\-of\-the\-art accuracy claim.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2608.07382v1)

---

> ### 5. YOLO\-PEFT: Parameter\-Efficient Fine\-Tuning on YOLO Family
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-08-07 |
> | 👤 作者 | Xu Lin |
>
> **📄 英文摘要：**
> Generic parameter\-efficient fine\-tuning \(PEFT\) methods transferred from language models can fail silently on real\-time detectors, whose heterogeneous operators and detection\-specific components impose placement constraints absent from regular Transformer stacks. We propose YOLO\-PEFT, a structure\-aware framework that formulates adapter placement as an auditable constraint\-planning problem. Given a detector graph, a PEFT request, and a resource budget, YOLO\-PEFT assigns operator and semantic roles, evaluates explicit operator\-validity, detector\-semantic, graph\-interface, and deployment predicates, records a reason code for each excluded module, and either emits a budgeted target\-module plan or returns Refuse before training. Under the official VOC07\+12 trainval\-to\-VOC07 test protocol, planner\-selected RS\-LoRA reaches 0.7138 and 0.7307 mAP50\-95 on YOLO11s and YOLO12s, respectively, compared with 0.6428 and 0.6662 for Full\-SFT. On RT\-DETR\-L, all seven evaluated LoRA\-family configurations cross the predefined catastrophic threshold, supporting a calibrated Refuse\-to\-Full\-SFT decision within the evaluated coverage. A controlled YOLO11 audit further shows that LoRA reduces peak training memory by 43.9 percent, although training takes 1.72 times longer. Within the evaluated detector families, placement policies, and calibration coverage, YOLO\-PEFT replaces manual target\-module trial and error with explicit, inspectable planning while preserving verified train\-save\-merge\-export paths; refusal on unseen detector architectures remains an open validation problem. Project Page: github.com/Tencent/YOLO\-Master
>
> 🔗 [阅读论文](http://arxiv.org/abs/2608.07051v1)

---

<div align="center">

*由 [YOLO-Arxiv-Daily](https://github.com/WangQvQ/YOLO-Arxiv-Daily) 自动生成*

</div>