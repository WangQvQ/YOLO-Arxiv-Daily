# 每日从arXiv中获取最新YOLO相关论文


## Concept\-based explanations of Segmentation and Detection models in Natural Disaster Management / 

发布日期：2026-03-24

作者：Samar Heydari

摘要：Deep learning models for flood and wildfire segmentation and object detection enable precise, real\-time disaster localization when deployed on embedded drone platforms. However, in natural disaster management, the lack of transparency in their decision\-making process hinders human trust required for emergency response. To address this, we present an explainability framework for understanding flood segmentation and car detection predictions on the widely used PIDNet and YOLO architectures. More specifically, we introduce a novel redistribution strategy that extends Layer\-wise Relevance Propagation \(LRP\) explanations for sigmoid\-gated element\-wise fusion layers. This extension allows LRP relevances to flow through the fusion modules of PIDNet, covering the entire computation graph back to the input image. Furthermore, we apply Prototypical Concept\-based Explanations \(PCX\) to provide both local and global explanations at the concept level, revealing which learned features drive the segmentation and detection of specific disaster semantic classes. Experiments on a publicly available flood dataset show that our framework provides reliable and interpretable explanations while maintaining near real\-time inference capabilities, rendering it suitable for deployment on resource\-constrained platforms, such as Unmanned Aerial Vehicles \(UAVs\).

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2603.23020v1)

---


## MS\-CustomNet: Controllable Multi\-Subject Customization with Hierarchical Relational Semantics / 

发布日期：2026-03-22

作者：Pengxiang Cai

摘要：Diffusion\-based text\-to\-image generation has advanced significantly, yet customizing scenes with multiple distinct subjects while maintaining fine\-grained control over their interactions remains challenging. Existing methods often struggle to provide explicit user\-defined control over the compositional structure and precise spatial relationships between subjects. To address this, we introduce MS\-CustomNet, a novel framework for multi\-subject customization. MS\-CustomNet allows zero\-shot integration of multiple user\-provided objects and, crucially, empowers users to explicitly define these hierarchical arrangements and spatial placements within the generated image. Our approach ensures individual subject identity preservation while learning and enacting these user\-specified inter\-subject compositions. We also present the MSI dataset, derived from COCO, to facilitate training on such complex multi\-subject compositions. MS\-CustomNet offers enhanced, fine\-grained control over multi\-subject image generation. Our method achieves a DINO\-I score of 0.61 for identity preservation and a YOLO\-L score of 0.94 for positional control in multi\-subject customization tasks, demonstrating its superior capability in generating high\-fidelity images with precise, user\-directed multi\-subject compositions and spatial control.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2603.21136v1)

---


## Anatomical Prior\-Driven Framework for Autonomous Robotic Cardiac Ultrasound Standard View Acquisition / 

发布日期：2026-03-22

作者：Zhiyan Cao

摘要：Cardiac ultrasound diagnosis is critical for cardiovascular disease assessment, but acquiring standard views remains highly operator\-dependent. Existing medical segmentation models often yield anatomically inconsistent results in images with poor textural differentiation between distinct feature classes, while autonomous probe adjustment methods either rely on simplistic heuristic rules or black\-box learning. To address these issues, our study proposed an anatomical prior \(AP\)\-driven framework integrating cardiac structure segmentation and autonomous probe adjustment for standard view acquisition. A YOLO\-based multi\-class segmentation model augmented by a spatial\-relation graph \(SRG\) module is designed to embed AP into the feature pyramid. Quantifiable anatomical features of standard views are extracted. Their priors are fitted to Gaussian distributions to construct probabilistic APs. The probe adjustment process of robotic ultrasound scanning is formalized as a reinforcement learning \(RL\) problem, with the RL state built from real\-time anatomical features and the reward reflecting the AP matching. Experiments validate the efficacy of the framework. The SRG\-YOLOv11s improves mAP50 by 11.3% and mIoU by 6.8% on the Special Case dataset, while the RL agent achieves a 92.5% success rate in simulation and 86.7% in phantom experiments.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2603.21134v1)

---


## Can Large Multimodal Models Inspect Buildings? A Hierarchical Benchmark for Structural Pathology Reasoning / 

发布日期：2026-03-20

作者：Hui Zhong

摘要：Automated building facade inspection is a critical component of urban resilience and smart city maintenance. Traditionally, this field has relied on specialized discriminative models \(e.g., YOLO, Mask R\-CNN\) that excel at pixel\-level localization but are constrained to passive perception and worse generization without the visual understandng to interpret structural topology. Large Multimodal Models \(LMMs\) promise a paradigm shift toward active reasoning, yet their application in such high\-stakes engineering domains lacks rigorous evaluation standards. To bridge this gap, we introduce a human\-in\-the\-loop semi\-automated annotation framework, leveraging expert\-proposal verification to unify 12 fragmented datasets into a standardized, hierarchical ontology. Building on this foundation, we present textit\{DefectBench\}, the first multi\-dimensional benchmark designed to interrogate LMMs beyond basic semantic recognition. textit\{DefectBench\} evaluates 18 state\-of\-the\-art \(SOTA\) LMMs across three escalating cognitive dimensions: Semantic Perception, Spatial Localization, and Generative Geometry Segmentation. Extensive experiments reveal that while current LMMs demonstrate exceptional topological awareness and semantic understanding \(effectively diagnosing "what" and "how"\), they exhibit significant deficiencies in metric localization precision \("where"\). Crucially, however, we validate the viability of zero\-shot generative segmentation, showing that general\-purpose foundation models can rival specialized supervised networks without domain\-specific training. This work provides both a rigorous benchmarking standard and a high\-quality open\-source database, establishing a new baseline for the advancement of autonomous AI agents in civil engineering.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2603.20148v1)

---


## Real\-Time Structural Detection for Indoor Navigation from 3D LiDAR Using Bird's\-Eye\-View Images / 

发布日期：2026-03-20

作者：Guanliang Li

摘要：Efficient structural perception is essential for mapping and autonomous navigation on resource\-constrained robots. Existing 3D methods are computationally prohibitive, while traditional 2D geometric approaches lack robustness. This paper presents a lightweight, real\-time framework that projects 3D LiDAR data into 2D Bird's\-Eye\-View \(BEV\) images to enable efficient detection of structural elements relevant to mapping and navigation. Within this representation, we systematically evaluate several feature extraction strategies, including classical geometric techniques \(Hough Transform, RANSAC, and LSD\) and a deep learning detector based on YOLO\-OBB. The resulting detections are integrated through a spatiotemporal fusion module that improves stability and robustness across consecutive frames. Experiments conducted on a standard mobile robotic platform highlight clear performance trade\-offs. Classical methods such as Hough and LSD provide fast responses but exhibit strong sensitivity to noise, with LSD producing excessive segment fragmentation that leads to system congestion. RANSAC offers improved robustness but fails to meet real\-time constraints. In contrast, the YOLO\-OBB\-based approach achieves the best balance between robustness and computational efficiency, maintaining an end\-to\-end latency \(satisfying 10 Hz operation\) while effectively filtering cluttered observations in a low\-power single\-board computer \(SBC\) without using GPU acceleration. The main contribution of this work is a computationally efficient BEV\-based perception pipeline enabling reliable real\-time structural detection from 3D LiDAR on resource\-constrained robotic platforms that cannot rely on GPU\-intensive processing.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2603.19830v1)

---

