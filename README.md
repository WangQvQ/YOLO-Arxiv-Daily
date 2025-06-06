# 每日从arXiv中获取最新YOLO相关论文


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


## DiagNet: Detecting Objects using Diagonal Constraints on Adjacency Matrix of Graph Neural Network / 

发布日期：2025-06-04

作者：Chong Hyun Lee

摘要：We propose DaigNet, a new approach to object detection with which we can detect an object bounding box using diagonal constraints on adjacency matrix of a graph convolutional network \(GCN\). We propose two diagonalization algorithms based on hard and soft constraints on adjacency matrix and two loss functions using diagonal constraint and complementary constraint. The DaigNet eliminates the need for designing a set of anchor boxes commonly used. To prove feasibility of our novel detector, we adopt detection head in YOLO models. Experiments show that the DiagNet achieves 7.5% higher mAP50 on Pascal VOC than YOLOv1. The DiagNet also shows 5.1% higher mAP on MS COCO than YOLOv3u, 3.7% higher mAP than YOLOv5u, and 2.9% higher mAP than YOLOv8.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2506.03571v1)

---


## Efficient Endangered Deer Species Monitoring with UAV Aerial Imagery and Deep Learning / 

发布日期：2025-05-30

作者：Agustín Roca

摘要：This paper examines the use of Unmanned Aerial Vehicles \(UAVs\) and deep learning for detecting endangered deer species in their natural habitats. As traditional identification processes require trained manual labor that can be costly in resources and time, there is a need for more efficient solutions. Leveraging high\-resolution aerial imagery, advanced computer vision techniques are applied to automate the identification process of deer across two distinct projects in Buenos Aires, Argentina. The first project, Pantano Project, involves the marsh deer in the Paran'a Delta, while the second, WiMoBo, focuses on the Pampas deer in Campos del Tuy'u National Park. A tailored algorithm was developed using the YOLO framework, trained on extensive datasets compiled from UAV\-captured images. The findings demonstrate that the algorithm effectively identifies marsh deer with a high degree of accuracy and provides initial insights into its applicability to Pampas deer, albeit with noted limitations. This study not only supports ongoing conservation efforts but also highlights the potential of integrating AI with UAV technology to enhance wildlife monitoring and management practices.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2506.00164v1)

---

