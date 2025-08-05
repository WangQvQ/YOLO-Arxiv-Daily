# 每日从arXiv中获取最新YOLO相关论文


## YOLOv1 to YOLOv11: A Comprehensive Survey of Real\-Time Object Detection Innovations and Challenges / 

发布日期：2025-08-04

作者：Manikanta Kotthapalli

摘要：Over the past decade, object detection has advanced significantly, with the YOLO \(You Only Look Once\) family of models transforming the landscape of real\-time vision applications through unified, end\-to\-end detection frameworks. From YOLOv1's pioneering regression\-based detection to the latest YOLOv9, each version has systematically enhanced the balance between speed, accuracy, and deployment efficiency through continuous architectural and algorithmic advancements.. Beyond core object detection, modern YOLO architectures have expanded to support tasks such as instance segmentation, pose estimation, object tracking, and domain\-specific applications including medical imaging and industrial automation. This paper offers a comprehensive review of the YOLO family, highlighting architectural innovations, performance benchmarks, extended capabilities, and real\-world use cases. We critically analyze the evolution of YOLO models and discuss emerging research directions that extend their impact across diverse computer vision domains.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2508.02067v1)

---


## Self\-Supervised YOLO: Leveraging Contrastive Learning for Label\-Efficient Object Detection / 

发布日期：2025-08-04

作者：Manikanta Kotthapalli

摘要：One\-stage object detectors such as the YOLO family achieve state\-of\-the\-art performance in real\-time vision applications but remain heavily reliant on large\-scale labeled datasets for training. In this work, we present a systematic study of contrastive self\-supervised learning \(SSL\) as a means to reduce this dependency by pretraining YOLOv5 and YOLOv8 backbones on unlabeled images using the SimCLR framework. Our approach introduces a simple yet effective pipeline that adapts YOLO's convolutional backbones as encoders, employs global pooling and projection heads, and optimizes a contrastive loss using augmentations of the COCO unlabeled dataset \(120k images\). The pretrained backbones are then fine\-tuned on a cyclist detection task with limited labeled data. Experimental results show that SSL pretraining leads to consistently higher mAP, faster convergence, and improved precision\-recall performance, especially in low\-label regimes. For example, our SimCLR\-pretrained YOLOv8 achieves a mAP@50:95 of 0.7663, outperforming its supervised counterpart despite using no annotations during pretraining. These findings establish a strong baseline for applying contrastive SSL to one\-stage detectors and highlight the potential of unlabeled data as a scalable resource for label\-efficient object detection.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2508.01966v1)

---


## SBP\-YOLO:A Lightweight Real\-Time Model for Detecting Speed Bumps and Potholes / 

发布日期：2025-08-02

作者：Chuanqi Liang

摘要：With increasing demand for ride comfort in new energy vehicles, accurate real\-time detection of speed bumps and potholes is critical for predictive suspension control. This paper proposes SBP\-YOLO, a lightweight detection framework based on YOLOv11, optimized for embedded deployment. The model integrates GhostConv for efficient computation, VoVGSCSPC for multi\-scale feature enhancement, and a Lightweight Efficiency Detection Head \(LEDH\) to reduce early\-stage feature processing costs. A hybrid training strategy combining NWD loss, knowledge distillation, and Albumentations\-based weather augmentation improves detection robustness, especially for small and distant targets. Experiments show SBP\-YOLO achieves 87.0% mAP \(outperforming YOLOv11n by 5.8%\) and runs at 139.5 FPS on a Jetson AGX Xavier with TensorRT FP16 quantization. The results validate its effectiveness for real\-time road condition perception in intelligent suspension systems.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2508.01339v1)

---


## YOLO\-Count: Differentiable Object Counting for Text\-to\-Image Generation / 

发布日期：2025-08-01

作者：Guanning Zeng

摘要：We propose YOLO\-Count, a differentiable open\-vocabulary object counting model that tackles both general counting challenges and enables precise quantity control for text\-to\-image \(T2I\) generation. A core contribution is the 'cardinality' map, a novel regression target that accounts for variations in object size and spatial distribution. Leveraging representation alignment and a hybrid strong\-weak supervision scheme, YOLO\-Count bridges the gap between open\-vocabulary counting and T2I generation control. Its fully differentiable architecture facilitates gradient\-based optimization, enabling accurate object count estimation and fine\-grained guidance for generative models. Extensive experiments demonstrate that YOLO\-Count achieves state\-of\-the\-art counting accuracy while providing robust and effective quantity control for T2I systems.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2508.00728v1)

---


## Towards Field\-Ready AI\-based Malaria Diagnosis: A Continual Learning Approach / 

发布日期：2025-07-31

作者：Louise Guillon

摘要：Malaria remains a major global health challenge, particularly in low\-resource settings where access to expert microscopy may be limited. Deep learning\-based computer\-aided diagnosis \(CAD\) systems have been developed and demonstrate promising performance on thin blood smear images. However, their clinical deployment may be hindered by limited generalization across sites with varying conditions. Yet very few practical solutions have been proposed. In this work, we investigate continual learning \(CL\) as a strategy to enhance the robustness of malaria CAD models to domain shifts. We frame the problem as a domain\-incremental learning scenario, where a YOLO\-based object detector must adapt to new acquisition sites while retaining performance on previously seen domains. We evaluate four CL strategies, two rehearsal\-based and two regularization\-based methods, on real\-life conditions thanks to a multi\-site clinical dataset of thin blood smear images. Our results suggest that CL, and rehearsal\-based methods in particular, can significantly improve performance. These findings highlight the potential of continual learning to support the development of deployable, field\-ready CAD tools for malaria.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2507.23648v1)

---

