# 每日从arXiv中获取最新YOLO相关论文


## BakuFlow: A Streamlining Semi\-Automatic Label Generation Tool / 

发布日期：2025-06-10

作者：Jerry Lin

摘要：Accurately labeling \(or annotation\) data is still a bottleneck in computer vision, especially for large\-scale tasks where manual labeling is time\-consuming and error\-prone. While tools like LabelImg can handle the labeling task, some of them still require annotators to manually label each image. In this paper, we introduce BakuFlow, a streamlining semi\-automatic label generation tool. Key features include \(1\) a live adjustable magnifier for pixel\-precise manual corrections, improving user experience; \(2\) an interactive data augmentation module to diversify training datasets; \(3\) label propagation for rapidly copying labeled objects between consecutive frames, greatly accelerating annotation of video data; and \(4\) an automatic labeling module powered by a modified YOLOE framework. Unlike the original YOLOE, our extension supports adding new object classes and any number of visual prompts per class during annotation, enabling flexible and scalable labeling for dynamic, real\-world datasets. These innovations make BakuFlow especially effective for object detection and tracking, substantially reducing labeling workload and improving efficiency in practical computer vision and industrial scenarios.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2506.09083v1)

---


## CBAM\-STN\-TPS\-YOLO: Enhancing Agricultural Object Detection through Spatially Adaptive Attention Mechanisms / 

发布日期：2025-06-09

作者：Satvik Praveen

摘要：Object detection is vital in precision agriculture for plant monitoring, disease detection, and yield estimation. However, models like YOLO struggle with occlusions, irregular structures, and background noise, reducing detection accuracy. While Spatial Transformer Networks \(STNs\) improve spatial invariance through learned transformations, affine mappings are insufficient for non\-rigid deformations such as bent leaves and overlaps.   We propose CBAM\-STN\-TPS\-YOLO, a model integrating Thin\-Plate Splines \(TPS\) into STNs for flexible, non\-rigid spatial transformations that better align features. Performance is further enhanced by the Convolutional Block Attention Module \(CBAM\), which suppresses background noise and emphasizes relevant spatial and channel\-wise features.   On the occlusion\-heavy Plant Growth and Phenotyping \(PGP\) dataset, our model outperforms STN\-YOLO in precision, recall, and mAP. It achieves a 12% reduction in false positives, highlighting the benefits of improved spatial flexibility and attention\-guided refinement. We also examine the impact of the TPS regularization parameter in balancing transformation smoothness and detection performance.   This lightweight model improves spatial awareness and supports real\-time edge deployment, making it ideal for smart farming applications requiring accurate and efficient monitoring.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2506.07357v1)

---


## Gen\-n\-Val: Agentic Image Data Generation and Validation / 

发布日期：2025-06-05

作者：Jing\-En Huang

摘要：Recently, Large Language Models \(LLMs\) and Vision Large Language Models \(VLLMs\) have demonstrated impressive performance as agents across various tasks while data scarcity and label noise remain significant challenges in computer vision tasks, such as object detection and instance segmentation. A common solution for resolving these issues is to generate synthetic data. However, current synthetic data generation methods struggle with issues, such as multiple objects per mask, inaccurate segmentation, and incorrect category labels, limiting their effectiveness. To address these issues, we introduce Gen\-n\-Val, a novel agentic data generation framework that leverages Layer Diffusion \(LD\), LLMs, and VLLMs to produce high\-quality, single\-object masks and diverse backgrounds. Gen\-n\-Val consists of two agents: \(1\) The LD prompt agent, an LLM, optimizes prompts for LD to generate high\-quality foreground instance images and segmentation masks. These optimized prompts ensure the generation of single\-object synthetic data with precise instance masks and clean backgrounds. \(2\) The data validation agent, a VLLM, which filters out low\-quality synthetic instance images. The system prompts for both agents are refined through TextGrad. Additionally, we use image harmonization to combine multiple instances within scenes. Compared to state\-of\-the\-art synthetic data approaches like MosaicFusion, our approach reduces invalid synthetic data from 50% to 7% and improves performance by 1% mAP on rare classes in COCO instance segmentation with YOLOv9c and YOLO11m. Furthermore, Gen\-n\-Val shows significant improvements \(7. 1% mAP\) over YOLO\-Worldv2\-M in open\-vocabulary object detection benchmarks with YOLO11m. Moreover, Gen\-n\-Val improves the performance of YOLOv9 and YOLO11 families in instance segmentation and object detection.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2506.04676v1)

---


## MS\-YOLO: A Multi\-Scale Model for Accurate and Efficient Blood Cell Detection / 

发布日期：2025-06-04

作者：Guohua Wu

摘要：Complete blood cell detection holds significant value in clinical diagnostics. Conventional manual microscopy methods suffer from time inefficiency and diagnostic inaccuracies. Existing automated detection approaches remain constrained by high deployment costs and suboptimal accuracy. While deep learning has introduced powerful paradigms to this field, persistent challenges in detecting overlapping cells and multi\-scale objects hinder practical deployment. This study proposes the multi\-scale YOLO \(MS\-YOLO\), a blood cell detection model based on the YOLOv11 framework, incorporating three key architectural innovations to enhance detection performance. Specifically, the multi\-scale dilated residual module \(MS\-DRM\) replaces the original C3K2 modules to improve multi\-scale discriminability; the dynamic cross\-path feature enhancement module \(DCFEM\) enables the fusion of hierarchical features from the backbone with aggregated features from the neck to enhance feature representations; and the light adaptive\-weight downsampling module \(LADS\) improves feature downsampling through adaptive spatial weighting while reducing computational complexity. Experimental results on the CBC benchmark demonstrate that MS\-YOLO achieves precise detection of overlapping cells and multi\-scale objects, particularly small targets such as platelets, achieving an mAP@50 of 97.4% that outperforms existing models. Further validation on the supplementary WBCDD dataset confirms its robust generalization capability. Additionally, with a lightweight architecture and real\-time inference efficiency, MS\-YOLO meets clinical deployment requirements, providing reliable technical support for standardized blood pathology assessment.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2506.03972v1)

---


## MambaNeXt\-YOLO: A Hybrid State Space Model for Real\-time Object Detection / 

发布日期：2025-06-04

作者：Xiaochun Lei

摘要：Real\-time object detection is a fundamental but challenging task in computer vision, particularly when computational resources are limited. Although YOLO\-series models have set strong benchmarks by balancing speed and accuracy, the increasing need for richer global context modeling has led to the use of Transformer\-based architectures. Nevertheless, Transformers have high computational complexity because of their self\-attention mechanism, which limits their practicality for real\-time and edge deployments. To overcome these challenges, recent developments in linear state space models, such as Mamba, provide a promising alternative by enabling efficient sequence modeling with linear complexity. Building on this insight, we propose MambaNeXt\-YOLO, a novel object detection framework that balances accuracy and efficiency through three key contributions: \(1\) MambaNeXt Block: a hybrid design that integrates CNNs with Mamba to effectively capture both local features and long\-range dependencies; \(2\) Multi\-branch Asymmetric Fusion Pyramid Network \(MAFPN\): an enhanced feature pyramid architecture that improves multi\-scale object detection across various object sizes; and \(3\) Edge\-focused Efficiency: our method achieved 66.6% mAP at 31.9 FPS on the PASCAL VOC dataset without any pre\-training and supports deployment on edge devices such as the NVIDIA Jetson Xavier NX and Orin NX.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2506.03654v2)

---

