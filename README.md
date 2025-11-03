# 每日从arXiv中获取最新YOLO相关论文


## Mask\-to\-Height: A YOLOv11\-Based Architecture for Joint Building Instance Segmentation and Height Classification from Satellite Imagery / 

发布日期：2025-10-31

作者：Mahmoud El Hussieni

摘要：Accurate building instance segmentation and height classification are critical for urban planning, 3D city modeling, and infrastructure monitoring. This paper presents a detailed analysis of YOLOv11, the recent advancement in the YOLO series of deep learning models, focusing on its application to joint building extraction and discrete height classification from satellite imagery. YOLOv11 builds on the strengths of earlier YOLO models by introducing a more efficient architecture that better combines features at different scales, improves object localization accuracy, and enhances performance in complex urban scenes. Using the DFC2023 Track 2 dataset \-\- which includes over 125,000 annotated buildings across 12 cities \-\- we evaluate YOLOv11's performance using metrics such as precision, recall, F1 score, and mean average precision \(mAP\). Our findings demonstrate that YOLOv11 achieves strong instance segmentation performance with 60.4% mAP@50 and 38.3% mAP@50\-\-95 while maintaining robust classification accuracy across five predefined height tiers. The model excels in handling occlusions, complex building shapes, and class imbalance, particularly for rare high\-rise structures. Comparative analysis confirms that YOLOv11 outperforms earlier multitask frameworks in both detection accuracy and inference speed, making it well\-suited for real\-time, large\-scale urban mapping. This research highlights YOLOv11's potential to advance semantic urban reconstruction through streamlined categorical height modeling, offering actionable insights for future developments in remote sensing and geospatial intelligence.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2510.27224v1)

---


## maxVSTAR: Maximally Adaptive Vision\-Guided CSI Sensing with Closed\-Loop Edge Model Adaptation for Robust Human Activity Recognition / 

发布日期：2025-10-30

作者：Kexing Liu

摘要：WiFi Channel State Information \(CSI\)\-based human activity recognition \(HAR\) provides a privacy\-preserving, device\-free sensing solution for smart environments. However, its deployment on edge devices is severely constrained by domain shift, where recognition performance deteriorates under varying environmental and hardware conditions. This study presents maxVSTAR \(maximally adaptive Vision\-guided Sensing Technology for Activity Recognition\), a closed\-loop, vision\-guided model adaptation framework that autonomously mitigates domain shift for edge\-deployed CSI sensing systems. The proposed system integrates a cross\-modal teacher\-student architecture, where a high\-accuracy YOLO\-based vision model serves as a dynamic supervisory signal, delivering real\-time activity labels for the CSI data stream. These labels enable autonomous, online fine\-tuning of a lightweight CSI\-based HAR model, termed Sensing Technology for Activity Recognition \(STAR\), directly at the edge. This closed\-loop retraining mechanism allows STAR to continuously adapt to environmental changes without manual intervention. Extensive experiments demonstrate the effectiveness of maxVSTAR. When deployed on uncalibrated hardware, the baseline STAR model's recognition accuracy declined from 93.52% to 49.14%. Following a single vision\-guided adaptation cycle, maxVSTAR restored the accuracy to 81.51%. These results confirm the system's capacity for dynamic, self\-supervised model adaptation in privacy\-conscious IoT environments, establishing a scalable and practical paradigm for long\-term autonomous HAR using CSI sensing at the network edge.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2510.26146v1)

---


## DINO\-YOLO: Self\-Supervised Pre\-training for Data\-Efficient Object Detection in Civil Engineering Applications / 

发布日期：2025-10-29

作者：Malaisree P

摘要：Object detection in civil engineering applications is constrained by limited annotated data in specialized domains. We introduce DINO\-YOLO, a hybrid architecture combining YOLOv12 with DINOv3 self\-supervised vision transformers for data\-efficient detection. DINOv3 features are strategically integrated at two locations: input preprocessing \(P0\) and mid\-backbone enhancement \(P3\). Experimental validation demonstrates substantial improvements: Tunnel Segment Crack detection \(648 images\) achieves 12.4% improvement, Construction PPE \(1K images\) gains 13.7%, and KITTI \(7K images\) shows 88.6% improvement, while maintaining real\-time inference \(30\-47 FPS\). Systematic ablation across five YOLO scales and nine DINOv3 variants reveals that Medium\-scale architectures achieve optimal performance with DualP0P3 integration \(55.77% mAP@0.5\), while Small\-scale requires Triple Integration \(53.63%\). The 2\-4x inference overhead \(21\-33ms versus 8\-16ms baseline\) remains acceptable for field deployment on NVIDIA RTX 5090. DINO\-YOLO establishes state\-of\-the\-art performance for civil engineering datasets \(<10K images\) while preserving computational efficiency, providing practical solutions for construction safety monitoring and infrastructure inspection in data\-constrained environments.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2510.25140v2)

---


## Delving into Cascaded Instability: A Lipschitz Continuity View on Image Restoration and Object Detection Synergy / 

发布日期：2025-10-28

作者：Qing Zhao

摘要：To improve detection robustness in adverse conditions \(e.g., haze and low light\), image restoration is commonly applied as a pre\-processing step to enhance image quality for the detector. However, the functional mismatch between restoration and detection networks can introduce instability and hinder effective integration \-\- an issue that remains underexplored. We revisit this limitation through the lens of Lipschitz continuity, analyzing the functional differences between restoration and detection networks in both the input space and the parameter space. Our analysis shows that restoration networks perform smooth, continuous transformations, while object detectors operate with discontinuous decision boundaries, making them highly sensitive to minor perturbations. This mismatch introduces instability in traditional cascade frameworks, where even imperceptible noise from restoration is amplified during detection, disrupting gradient flow and hindering optimization. To address this, we propose Lipschitz\-regularized object detection \(LROD\), a simple yet effective framework that integrates image restoration directly into the detector's feature learning, harmonizing the Lipschitz continuity of both tasks during training. We implement this framework as Lipschitz\-regularized YOLO \(LR\-YOLO\), extending seamlessly to existing YOLO detectors. Extensive experiments on haze and low\-light benchmarks demonstrate that LR\-YOLO consistently improves detection stability, optimization smoothness, and overall accuracy.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2510.24232v1)

---


## CSST Slitless Spectra: Target Detection and Classification with YOLO / 

发布日期：2025-10-28

作者：Yingying Zhou

摘要：Addressing the spatial uncertainty and spectral blending challenges in CSST slitless spectroscopy, we present a deep learning\-driven, end\-to\-end framework based on the You Only Look Once \(YOLO\) models. This approach directly detects, classifies, and analyzes spectral traces from raw 2D images, bypassing traditional, error\-accumulating pipelines. YOLOv5 effectively detects both compact zero\-order and extended first\-order traces even in highly crowded fields. Building on this, YOLO11 integrates source classification \(star/galaxy\) and discrete astrophysical parameter estimation \(e.g., redshift bins\), showcasing complete spectral trace analysis without other manual preprocessing. Our framework processes large images rapidly, learning spectral\-spatial features holistically to minimize errors. We achieve high trace detection precision \(YOLOv5\) and demonstrate successful quasar identification and binned redshift estimation \(YOLO11\). This study establishes machine learning as a paradigm shift in slitless spectroscopy, unifying detection, classification, and preliminary parameter estimation in a scalable system. Future research will concentrate on direct, continuous prediction of astrophysical parameters from raw spectral traces.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2510.24087v1)

---

