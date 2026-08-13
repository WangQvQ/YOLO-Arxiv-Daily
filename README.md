<div align="center">

# YOLO ArXiv Daily

[![Daily Papers](https://img.shields.io/badge/📅-每日更新-blue)]()
[![arXiv](https://img.shields.io/badge/arXiv-最新论文-red)](https://arxiv.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)

*自动追踪 YOLO 相关最新论文，提供中英文双语摘要*

</div>

---

## 📑 论文列表

> ### 1. Do Not Forget the Obvious \- RISC: A Risk\-Informed Slice\-Coverage Protocol for Safe Autonomous Driving
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-08-12 |
> | 👤 作者 | Fabian Hüger |
>
> **📄 英文摘要：**
> Aggregate metrics may not fully reflect performance in insufficiently examined high\-risk driving conditions. We propose RISC \(Risk\-Informed Slice Coverage\), a practical protocol for risk\-guided stress testing and coverage\-qualified evaluation. Risk\-guided stress testing directs a finite audit budget toward risk\-relevant sub\-datasets, called risk slices, while coverage\-qualified evaluation reports results together with explicit statements about which slices are sufficiently or insufficiently covered. The protocol translates safety concerns into machine\-readable risk slices, uses lightweight signals to tag candidate data, selects a compact audit set by risk, and qualifies the results using coverage evidence. An LLM can optionally support this process by surfacing relevant but potentially overlooked conditions during test planning, thereby helping engineers not to forget the obvious. RISC is model\-agnostic and can be applied to perception modules, driving models, and other autonomous\-driving subsystems. We instantiate the protocol for monocular pedestrian perception using 1,000 frames from the Zenseact Open Dataset, image statistics, and a YOLO\-based detector proxy. In this proof\-of\-concept study, risk\-guided selection increases critical failure discovery from 34.0% under random sampling to 98.5%. RISC provides a lightweight, assurance\-oriented evaluation layer that complements scenario categorization, coverage assessment, and broader testing\-and\-verification workflows.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2608.12051v1)

---

> ### 2. A Hybrid Framework of Vision Transformer and Gated Recurrent Unit for Detection of Mosquito Diseases
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-08-12 |
> | 👤 作者 | Danial Sharifrazi |
>
> **📄 英文摘要：**
> Identifying dengue virus\-infected mosquitoes from control mosquitoes is a major challenge in analyzing mosquito locomotion behavior due to the small size and complexity of the video background. Conventional AI methods are often unable to extract accurate features from video frames and produce erroneous features. In this study, a three\-step framework is introduced: first, mosquitoes are identified and the background is removed using the YOLO 11M model, then visual features are extracted using the Vision Transformer \(ViT\), and finally the videos are classified with a convolutional GRU \(ConvGRU\) classifier. A comparative analysis of different models, including Recurrent Neural Network \(RNN\), Long Short\-Term Memory \(LSTM\), Gated Recurrent Unit \(GRU\), and their convolutional versions showed that the ConvGRU model achieved the best performance; it achieved 88.88% accuracy, 84.45% precision, 82.82% recall, and 82.81% F1 score. These results demonstrate that combining convolutional models with sequence\-based networks, especially in the ConvGRU model, allows the simultaneous extraction of precise spatial features and long\-term temporal dependencies from mosquito movements. Finally, the proposed framework provides a reliable solution for analyzing mosquito behavior in complex environments.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2608.11582v1)

---

> ### 3. A Comparative Evaluation of Deep Learning Object Detection Models on a Real\-World Multi\-Plant Dataset from Africa
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

> ### 4. MammoMix: Leveraging Mixture of Experts for Robust Mammogram Breast Detection
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

> ### 5. TriView\-YOLO: Early Multi\-View Fusion for Ground Penetrating Radar Cavity Detection in Soft, High\-Water\-Content Soils
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

<div align="center">

*由 [YOLO-Arxiv-Daily](https://github.com/WangQvQ/YOLO-Arxiv-Daily) 自动生成*

</div>