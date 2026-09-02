<div align="center">

# YOLO ArXiv Daily

[![Daily Papers](https://img.shields.io/badge/📅-每日更新-blue)]()
[![arXiv](https://img.shields.io/badge/arXiv-最新论文-red)](https://arxiv.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)

*自动追踪 YOLO 相关最新论文，提供中英文双语摘要*

</div>

---

## 📑 论文列表

> ### 1. Real\-Time Video Anomaly Detection Using YOLO Pose Estimation and CLIP\-Based Semantic Scoring
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-08-31 |
> | 👤 作者 | Vanodhya G. Warnasooriya |
>
> **📄 英文摘要：**
> We propose a lightweight two\-stage framework for real\-time video anomaly detection. The first stage employs YOLO v11n\-pose to detect persons and extract seventeen skeletal keypoints in a single forward pass. The second stage encodes each cropped person region through CLIP ViT\-B/32 and computes cosine similarity against predefined textual descriptions of anomalous behaviors. This architecture eliminates the need for optical flow, standalone pose estimators, and density\-based scoring modules. Experiments on CUHK Avenue, ShanghaiTech Campus, and a custom indoor dataset collected at Chulalongkorn University demonstrate an end\-to\-end throughput of approximately 51 FPS on an NVIDIA Titan XP GPU, a 3.36x speedup over the multi\-feature baseline, while maintaining frame\-level AUROC values of 89.26%, 70.26%, and 84.13%, respectively.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2608.31074v1)

---

> ### 2. SynCrash: A Multi\-Stage Pipeline for Zero\-Shot Accident Detection and Localization in Traffic Surveillance Video
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-08-30 |
> | 👤 作者 | Arkya Jyoti Bagchi |
>
> **📄 英文摘要：**
> We present SynCrash, a multi\-stage pipeline for zero\-shot accident detection, spatial localization, and collision\-type classification in fixed\-view CCTV surveillance video. Our approach addresses the ACCIDENT at CVPR 2026 Challenge, which requires predicting when an accident occurs, where in the frame the impact happens, and what type of collision it is, all without access to labeled real\-world training data. The pipeline operates in three decoupled stages: \(1\) Temporal localization via a VideoMAEv2\-giant backbone fine\-tuned on CARLA\-based synthetic clips with metadata\-aware embeddings and dense sliding\-window inference; \(2\) Spatial localization using YOLO for object detection combined with a physics\-informed hybrid heuristic that leverages bounding\-box overlap and trajectory\-based reasoning to predict the impact point; and \(3\) Collision\-type classification using a lightweight rule\-based strategy derived from the number and configuration of detected vehicles. The key insight is that temporal understanding benefits from supervised fine\-tuning on synthetic data, whereas spatial understanding is better served by pretrained object detectors and physics priors that transfer naturally across domains.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2608.29759v1)

---

> ### 3. CF\-YOLO: Context\-Aware Feature Refinement for Camouflaged Industrial Micro\-Defect Detection
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-08-28 |
> | 👤 作者 | Xinda Yu |
>
> **📄 英文摘要：**
> Automated detection of surface micro\-defects on industrial components, such as copper tubes, is critically important for quality assurance but remains challenging due to the minute scale of anomalies and their visual camouflage against complex backgrounds. These factors lead to weak feature representations and high rates of false positives and missed detections. To address these issues, we propose a novel real\-time detection framework designed for efficient context perception and feature refinement. Our method integrates a Context\-Perception Aggregation Module \(CPAM\), which synergises large\-kernel perception for macro\-texture context and small\-kernel aggregation for sharp boundary delineation, effectively breaking the background camouflage. Furthermore, a Feature Additive Refinement Module \(FARM\) employs a linear\-complexity additive token mixer to globally verify and refine the representation of fine\-grained anomalies, suppressing noise\-induced errors. To support research in this domain, we introduce the Copper Tube Defect Dataset \(CTDD\), a manually annotated benchmark containing 1,847 images and 4,898 boundingbox defect instances from copper\-tube inspection scenarios. Extensive experiments demonstrate that our detector achieves strong and consistent performance on CTDD, outperforming representative baseline detectors, including YOLOv11, by 2.2% in mAP@50 and 3.9% in Precision while maintaining real\-time inference speed. This work provides a robust and efficient solution for high\-precision industrial inspection, bridging the gap between contextual understanding and detailed feature analysis. Our code and model are available at: https://github.com/Yu\-Xinda/CFYOLO\-Context\-Aware\-Feature\-Refinement\-for\-Camouflaged\-Industrial\-Micro\-Defect\-Detection
>
> **💻 代码链接：** https://github.com/Yu-Xinda/CFYOLO-Context-Aware-Feature-Refinement-for-Camouflaged-Industrial-Micro-Defect-Detection
>
> 🔗 [阅读论文](http://arxiv.org/abs/2608.28070v1)

---

> ### 4. Depth\-Aware Pothole Detection Using YOLO and RT\-DETR at the Edge
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-08-27 |
> | 👤 作者 | Md Monjurul Ahsan Prodhan |
>
> **📄 英文摘要：**
> Pothole detection and its severity measurement is still an important challenges in urban infrastructure management, where late maintenance directly contributes to vehicle damage, road accidents, and escalating repair costs. Existing automated approaches depend on 2D RGB images and cannot measure physical depth of potholes. In this paper, we present a depthaware pothole detection framework and then compare five architectures: YOLOv8n, YOLOv8nSeg, YOLOv9t, RTDETRL, and RTDETRX for RGB\-D sensor fusion\-based detection and automated depth measurement. A custom offline augmentation pipeline is used here to simulate adverse road monitoring conditions. All models are trained on the PothRGBD dataset with an 80% training and 20% validation split and evaluated using Precision, Recall, mAP@50, and mAP@50\_95. Before measuring the depth data, all depth maps are corrected for camera tilt using RANSAC ground\-plane orthorectification and all zero\-valued sensor pixels are cast to NaN before any statistic is computed. YOLOv8nSeg achieves the highest mAP@50 of 0.9556 and mAP@50\_95 of 0.6758 with the most accurate depth estimate of 2.96 cm with the pixel\-precise Dseg algorithm. YOLOv8n achieves the fastest inference at 3.6ms. RTDETRX achieves the highest detection confidence at 92.70%. An important finding is that even after full RANSAC orthorectification, bounding box models overestimate pothole depth by 0.16 to 0.21 cm compared to pixel precise segmentation masks. This confirms that the pavement inclusion bias is structural rather than a calibration artifact.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2608.27633v1)

---

> ### 5. Lowering the Barrier to AI\-Driven Inspection: A No\-Code Workflow for Automated Structural Defect Detection
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-08-25 |
> | 👤 作者 | Michael Holm |
>
> **📄 英文摘要：**
> Structural health monitoring \(SHM\) is essential in modern engineering, providing data for condition\-based maintenance, lifecycle assessment, and predictive decision\-making. Traditionally, SHM relied on visual inspection to detect defects such as cracks and deformations. Early computer vision \(CV\) methods, including thresholding, edge detection, and handcrafted features, aimed to automate this process but were highly sensitive to noise, imaging variations, and multiscale defects, limiting their reliability.   Recent advances in machine learning, particularly convolutional neural networks \(CNNs\) and You Only Look Once \(YOLO\), have improved defect detection accuracy and enabled real\-time analysis. However, adoption in SHM remains limited due to technical barriers such as data labeling, model training, and deployment, which typically require programming expertise.   To address this gap, we introduce YOLOEZ, an open\-source, GUI\-based tool for end\-to\-end YOLO model application. YOLOEZ integrates data labeling, training, and inference into a single interface, enabling high\-performance model development without code while supporting reproducible workflows.   Evaluation against existing software and classical image processing demonstrates that YOLOEZ not only outperforms traditional methods across most detection metrics, but also lowers adoption barriers present in other modern CV tools. By combining accuracy with accessibility, YOLOEZ facilitates wider use of AI\-driven monitoring for predictive maintenance, digital twins, and intelligent structural systems.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2608.25176v1)

---

<div align="center">

*由 [YOLO-Arxiv-Daily](https://github.com/WangQvQ/YOLO-Arxiv-Daily) 自动生成*

</div>