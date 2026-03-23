# 每日从arXiv中获取最新YOLO相关论文


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


## Template\-based Object Detection Using a Foundation Model / 

发布日期：2026-03-20

作者：Valentin Braeutigam

摘要：Most currently used object detection methods are learning\-based, and can detect objects under varying appearances. Those models require training and a training dataset. We focus on use cases with less data variation, but the requirement of being free of generation of training data and training. Such a setup is for example desired in automatic testing of graphical interfaces during software development, especially for continuous integration testing. In our approach, we use segments from segmentation foundation models and combine them with a simple feature\-based classification method. This saves time and cost when changing the object to be searched or its design, as nothing has to be retrained and no dataset has to be created. We evaluate our method on the task of detecting and classifying icons in navigation maps, which is used to simplify and automate the testing of user interfaces in automotive industry. Our methods achieve results almost on par with learning\-based object detection methods like YOLO, without the need for training.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2603.19773v1)

---


## EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task\-Specialized Distillation / 

发布日期：2026-03-19

作者：Longfei Liu

摘要：Deploying high\-performance dense prediction models on resource\-constrained edge devices remains challenging due to strict limits on computation and memory. In practice, lightweight systems for object detection, instance segmentation, and pose estimation are still dominated by CNN\-based architectures such as YOLO, while compact Vision Transformers \(ViTs\) often struggle to achieve similarly strong accuracy efficiency tradeoff, even with large scale pretraining. We argue that this gap is largely due to insufficient task specific representation learning in small scale ViTs, rather than an inherent mismatch between ViTs and edge dense prediction. To address this issue, we introduce EdgeCrafter, a unified compact ViT framework for edge dense prediction centered on ECDet, a detection model built from a distilled compact backbone and an edge\-friendly encoder decoder design. On the COCO dataset, ECDet\-S achieves 51.7 AP with fewer than 10M parameters using only COCO annotations. For instance segmentation, ECInsSeg achieves performance comparable to RF\-DETR while using substantially fewer parameters. For pose estimation, ECPose\-X reaches 74.8 AP, significantly outperforming YOLO26Pose\-X \(71.6 AP\) despite the latter's reliance on extensive Objects365 pretraining. These results show that compact ViTs, when paired with task\-specialized distillation and edge\-aware design, can be a practical and competitive option for edge dense prediction. Code is available at: https://intellindust\-ai\-lab.github.io/projects/EdgeCrafter/

中文摘要：


代码链接：https://intellindust-ai-lab.github.io/projects/EdgeCrafter/

论文链接：[阅读更多](http://arxiv.org/abs/2603.18739v1)

---


## HOMEY: Heuristic Object Masking with Enhanced YOLO for Property Insurance Risk Detection / 

发布日期：2026-03-19

作者：Teerapong Panboonyuen

摘要：Automated property risk detection is a high\-impact yet underexplored frontier in computer vision with direct implications for real estate, underwriting, and insurance operations. We introduce HOMEY \(Heuristic Object Masking with Enhanced YOLO\), a novel detection framework that combines YOLO with a domain\-specific masking mechanism and a custom\-designed loss function. HOMEY is trained to detect 17 risk\-related property classes, including structural damages \(e.g., cracked foundations, roof issues\), maintenance neglect \(e.g., dead yards, overgrown bushes\), and liability hazards \(e.g., falling gutters, garbage, hazard signs\). Our approach introduces heuristic object masking to amplify weak signals in cluttered backgrounds and risk\-aware loss calibration to balance class skew and severity weighting. Experiments on real\-world property imagery demonstrate that HOMEY achieves superior detection accuracy and reliability compared to baseline YOLO models, while retaining fast inference. Beyond detection, HOMEY enables interpretable and cost\-efficient risk analysis, laying the foundation for scalable AI\-driven property insurance workflows.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2603.18502v1)

---

