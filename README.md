# 每日从arXiv中获取最新YOLO相关论文


## SEP\-YOLO: Fourier\-Domain Feature Representation for Transparent Object Instance Segmentation / 

发布日期：2026-03-03

作者：Fengming Zhang

摘要：Transparent object instance segmentation presents significant challenges in computer vision, due to the inherent properties of transparent objects, including boundary blur, low contrast, and high dependence on background context. Existing methods often fail as they depend on strong appearance cues and clear boundaries. To address these limitations, we propose SEP\-YOLO, a novel framework that integrates a dual\-domain collaborative mechanism for transparent object instance segmentation. Our method incorporates a Frequency Domain Detail Enhancement Module, which separates and enhances weak highfrequency boundary components via learnable complex weights. We further design a multi\-scale spatial refinement stream, which consists of a Content\-Aware Alignment Neck and a Multi\-scale Gated Refinement Block, to ensure precise feature alignment and boundary localization in deep semantic features. We also provide high\-quality instance\-level annotations for the Trans10K dataset, filling the critical data gap in transparent object instance segmentation. Extensive experiments on the Trans10K and GVD datasets show that SEP\-YOLO achieves state\-of\-the\-art \(SOTA\) performance.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2603.02648v1)

---


## Towards Khmer Scene Document Layout Detection / 

发布日期：2026-02-28

作者：Marry Kong

摘要：While document layout analysis for Latin scripts has advanced significantly, driven by the advent of large multimodal models \(LMMs\), progress for the Khmer language remains constrained because of the scarcity of annotated training data. This gap is particularly acute for scene documents, where perspective distortions and complex backgrounds challenge traditional methods. Given the structural complexities of Khmer script, such as diacritics and multi\-layer character stacking, existing Latin\-based layout analysis models fail to accurately delineate semantic layout units, particularly for dense text regions \(e.g., list items\). In this paper, we present the first comprehensive study on Khmer scene document layout detection. We contribute a novel framework comprising three key elements: \(1\) a robust training and benchmarking dataset specifically for Khmer scene layouts; \(2\) an open\-source document augmentation tool capable of synthesizing realistic scene documents to scale training data; and \(3\) layout detection baselines utilizing YOLO\-based architectures with oriented bounding boxes \(OBB\) to handle geometric distortions. To foster further research in the Khmer document analysis and recognition \(DAR\) community, we release our models, code, and datasets in this gated repository \(in review\).

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2603.00707v1)

---


## Denoising\-Enhanced YOLO for Robust SAR Ship Detection / 

发布日期：2026-02-27

作者：Xiaojing Zhao

摘要：With the rapid advancement of deep learning, synthetic aperture radar \(SAR\) imagery has become a key modality for ship detection. However, robust performance remains challenging in complex scenes, where clutter and speckle noise can induce false alarms and small targets are easily missed. To address these issues, we propose CPN\-YOLO, a high\-precision ship detection framework built upon YOLOv8 with three targeted improvements. First, we introduce a learnable large\-kernel denoising module for input pre\-processing, producing cleaner representations and more discriminative features across diverse ship types. Second, we design a feature extraction enhancement strategy based on the PPA attention mechanism to strengthen multi\-scale modeling and improve sensitivity to small ships. Third, we incorporate a Gaussian similarity loss derived from the normalized Wasserstein distance \(NWD\) to better measure similarity under complex bounding\-box distributions and improve generalization. Extensive experiments on HRSID and SSDD demonstrate the effectiveness of our method. On SSDD, CPN\-YOLO surpasses the YOLOv8 baseline, achieving 97.0% precision, 95.1% recall, and 98.9% mAP, and consistently outperforms other representative deep\-learning detectors in overall performance.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2602.23820v1)

---


## SPMamba\-YOLO: An Underwater Object Detection Network Based on Multi\-Scale Feature Enhancement and Global Context Modeling / 

发布日期：2026-02-26

作者：Guanghao Liao

摘要：Underwater object detection is a critical yet challenging research problem owing to severe light attenuation, color distortion, background clutter, and the small scale of underwater targets. To address these challenges, we propose SPMamba\-YOLO, a novel underwater object detection network that integrates multi\-scale feature enhancement with global context modeling. Specifically, a Spatial Pyramid Pooling Enhanced Layer Aggregation Network \(SPPELAN\) module is introduced to strengthen multi\-scale feature aggregation and expand the receptive field, while a Pyramid Split Attention \(PSA\) mechanism enhances feature discrimination by emphasizing informative regions and suppressing background interference. In addition, a Mamba\-based state space modeling module is incorporated to efficiently capture long\-range dependencies and global contextual information, thereby improving detection robustness in complex underwater environments. Extensive experiments on the URPC2022 dataset demonstrate that SPMamba\-YOLO outperforms the YOLOv8n baseline by more than 4.9% in mAP@0.5, particularly for small and densely distributed underwater objects, while maintaining a favorable balance between detection accuracy and computational cost.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2602.22674v1)

---


## Don't let the information slip away / 

发布日期：2026-02-26

作者：Taozhe Li

摘要：Real\-time object detection has advanced rapidly in recent years. The YOLO series of detectors is among the most well\-known CNN\-based object detection models and cannot be overlooked. The latest version, YOLOv26, was recently released, while YOLOv12 achieved state\-of\-the\-art \(SOTA\) performance with 55.2 mAP on the COCO val2017 dataset. Meanwhile, transformer\-based object detection models, also known as DEtection TRansformer \(DETR\), have demonstrated impressive performance. RT\-DETR is an outstanding model that outperformed the YOLO series in both speed and accuracy when it was released. Its successor, RT\-DETRv2, achieved 53.4 mAP on the COCO val2017 dataset. However, despite their remarkable performance, all these models let information to slip away. They primarily focus on the features of foreground objects while neglecting the contextual information provided by the background. We believe that background information can significantly aid object detection tasks. For example, cars are more likely to appear on roads rather than in offices, while wild animals are more likely to be found in forests or remote areas rather than on busy streets. To address this gap, we propose an object detection model called Association DETR, which achieves state\-of\-the\-art results compared to other object detection models on the COCO val2017 dataset.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2602.22595v2)

---

