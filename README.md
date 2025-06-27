# 每日从arXiv中获取最新YOLO相关论文


## YOLO\-FDA: Integrating Hierarchical Attention and Detail Enhancement for Surface Defect Detection / 

发布日期：2025-06-26

作者：Jiawei Hu

摘要：Surface defect detection in industrial scenarios is both crucial and technically demanding due to the wide variability in defect types, irregular shapes and sizes, fine\-grained requirements, and complex material textures. Although recent advances in AI\-based detectors have improved performance, existing methods often suffer from redundant features, limited detail sensitivity, and weak robustness under multiscale conditions. To address these challenges, we propose YOLO\-FDA, a novel YOLO\-based detection framework that integrates fine\-grained detail enhancement and attention\-guided feature fusion. Specifically, we adopt a BiFPN\-style architecture to strengthen bidirectional multilevel feature aggregation within the YOLOv5 backbone. To better capture fine structural changes, we introduce a Detail\-directional Fusion Module \(DDFM\) that introduces a directional asymmetric convolution in the second\-lowest layer to enrich spatial details and fuses the second\-lowest layer with low\-level features to enhance semantic consistency. Furthermore, we propose two novel attention\-based fusion strategies, Attention\-weighted Concatenation \(AC\) and Cross\-layer Attention Fusion \(CAF\) to improve contextual representation and reduce feature noise. Extensive experiments on benchmark datasets demonstrate that YOLO\-FDA consistently outperforms existing state\-of\-the\-art methods in terms of both accuracy and robustness across diverse types of defects and scales.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2506.21135v1)

---


## Lightweight Multi\-Frame Integration for Robust YOLO Object Detection in Videos / 

发布日期：2025-06-25

作者：Yitong Quan

摘要：Modern image\-based object detection models, such as YOLOv7, primarily process individual frames independently, thus ignoring valuable temporal context naturally present in videos. Meanwhile, existing video\-based detection methods often introduce complex temporal modules, significantly increasing model size and computational complexity. In practical applications such as surveillance and autonomous driving, transient challenges including motion blur, occlusions, and abrupt appearance changes can severely degrade single\-frame detection performance. To address these issues, we propose a straightforward yet highly effective strategy: stacking multiple consecutive frames as input to a YOLO\-based detector while supervising only the output corresponding to a single target frame. This approach leverages temporal information with minimal modifications to existing architectures, preserving simplicity, computational efficiency, and real\-time inference capability. Extensive experiments on the challenging MOT20Det and our BOAT360 datasets demonstrate that our method improves detection robustness, especially for lightweight models, effectively narrowing the gap between compact and heavy detection networks. Additionally, we contribute the BOAT360 benchmark dataset, comprising annotated fisheye video sequences captured from a boat, to support future research in multi\-frame video object detection in challenging real\-world scenarios.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2506.20550v1)

---


## From Codicology to Code: A Comparative Study of Transformer and YOLO\-based Detectors for Layout Analysis in Historical Documents / 

发布日期：2025-06-25

作者：Sergio Torres Aguilar

摘要：Robust Document Layout Analysis \(DLA\) is critical for the automated processing and understanding of historical documents with complex page organizations. This paper benchmarks five state\-of\-the\-art object detection architectures on three annotated datasets representing a spectrum of codicological complexity: The e\-NDP, a corpus of Parisian medieval registers \(1326\-1504\); CATMuS, a diverse multiclass dataset derived from various medieval and modern sources \(ca.12th\-17th centuries\) and HORAE, a corpus of decorated books of hours \(ca.13th\-16th centuries\). We evaluate two Transformer\-based models \(Co\-DETR, Grounding DINO\) against three YOLO variants \(AABB, OBB, and YOLO\-World\). Our findings reveal significant performance variations dependent on model architecture, data set characteristics, and bounding box representation. In the e\-NDP dataset, Co\-DETR achieves state\-of\-the\-art results \(0.752 mAP@.50:.95\), closely followed by YOLOv11X\-OBB \(0.721\). Conversely, on the more complex CATMuS and HORAE datasets, the CNN\-based YOLOv11x\-OBB significantly outperforms all other models \(0.564 and 0.568, respectively\). This study unequivocally demonstrates that using Oriented Bounding Boxes \(OBB\) is not a minor refinement but a fundamental requirement for accurately modeling the non\-Cartesian nature of historical manuscripts. We conclude that a key trade\-off exists between the global context awareness of Transformers, ideal for structured layouts, and the superior generalization of CNN\-OBB models for visually diverse and complex documents.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2506.20326v1)

---


## Computer Vision based Automated Quantification of Agricultural Sprayers Boom Displacement / 

发布日期：2025-06-24

作者：Aryan Singh Dalal

摘要：Application rate errors when using self\-propelled agricultural sprayers for agricultural production remain a concern. Among other factors, spray boom instability is one of the major contributors to application errors. Spray booms' width of 38m, combined with 30 kph driving speeds, varying terrain, and machine dynamics when maneuvering complex field boundaries, make controls of these booms very complex. However, there is no quantitative knowledge on the extent of boom movement to systematically develop a solution that might include boom designs and responsive boom control systems. Therefore, this study was conducted to develop an automated computer vision system to quantify the boom movement of various agricultural sprayers. A computer vision system was developed to track a target on the edge of the sprayer boom in real time. YOLO V7, V8, and V11 neural network models were trained to track the boom's movements in field operations to quantify effective displacement in the vertical and transverse directions. An inclinometer sensor was mounted on the boom to capture boom angles and validate the neural network model output. The results showed that the model could detect the target with more than 90 percent accuracy, and distance estimates of the target on the boom were within 0.026 m of the inclinometer sensor data. This system can quantify the boom movement on the current sprayer and potentially on any other sprayer with minor modifications. The data can be used to make design improvements to make sprayer booms more stable and achieve greater application accuracy.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2506.19939v1)

---


## YOLOv13: Real\-Time Object Detection with Hypergraph\-Enhanced Adaptive Visual Perception / 

发布日期：2025-06-21

作者：Mengqi Lei

摘要：The YOLO series models reign supreme in real\-time object detection due to their superior accuracy and computational efficiency. However, both the convolutional architectures of YOLO11 and earlier versions and the area\-based self\-attention mechanism introduced in YOLOv12 are limited to local information aggregation and pairwise correlation modeling, lacking the capability to capture global multi\-to\-multi high\-order correlations, which limits detection performance in complex scenarios. In this paper, we propose YOLOv13, an accurate and lightweight object detector. To address the above\-mentioned challenges, we propose a Hypergraph\-based Adaptive Correlation Enhancement \(HyperACE\) mechanism that adaptively exploits latent high\-order correlations and overcomes the limitation of previous methods that are restricted to pairwise correlation modeling based on hypergraph computation, achieving efficient global cross\-location and cross\-scale feature fusion and enhancement. Subsequently, we propose a Full\-Pipeline Aggregation\-and\-Distribution \(FullPAD\) paradigm based on HyperACE, which effectively achieves fine\-grained information flow and representation synergy within the entire network by distributing correlation\-enhanced features to the full pipeline. Finally, we propose to leverage depthwise separable convolutions to replace vanilla large\-kernel convolutions, and design a series of blocks that significantly reduce parameters and computational complexity without sacrificing performance. We conduct extensive experiments on the widely used MS COCO benchmark, and the experimental results demonstrate that our method achieves state\-of\-the\-art performance with fewer parameters and FLOPs. Specifically, our YOLOv13\-N improves mAP by 3.0% over YOLO11\-N and by 1.5% over YOLOv12\-N. The code and models of our YOLOv13 model are available at: https://github.com/iMoonLab/yolov13.

中文摘要：


代码链接：https://github.com/iMoonLab/yolov13.

论文链接：[阅读更多](http://arxiv.org/abs/2506.17733v1)

---

