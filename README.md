# 每日从arXiv中获取最新YOLO相关论文


## An Investigation of Visual Foundation Models Robustness / 

发布日期：2025-08-22

作者：Sandeep Gupta

摘要：Visual Foundation Models \(VFMs\) are becoming ubiquitous in computer vision, powering systems for diverse tasks such as object detection, image classification, segmentation, pose estimation, and motion tracking. VFMs are capitalizing on seminal innovations in deep learning models, such as LeNet\-5, AlexNet, ResNet, VGGNet, InceptionNet, DenseNet, YOLO, and ViT, to deliver superior performance across a range of critical computer vision applications. These include security\-sensitive domains like biometric verification, autonomous vehicle perception, and medical image analysis, where robustness is essential to fostering trust between technology and the end\-users. This article investigates network robustness requirements crucial in computer vision systems to adapt effectively to dynamic environments influenced by factors such as lighting, weather conditions, and sensor characteristics. We examine the prevalent empirical defenses and robust training employed to enhance vision network robustness against real\-world challenges such as distributional shifts, noisy and spatially distorted inputs, and adversarial attacks. Subsequently, we provide a comprehensive analysis of the challenges associated with these defense mechanisms, including network properties and components to guide ablation studies and benchmarking metrics to evaluate network robustness.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2508.16225v1)

---


## DRespNeT: A UAV Dataset and YOLOv8\-DRN Model for Aerial Instance Segmentation of Building Access Points for Post\-Earthquake Search\-and\-Rescue Missions / 

发布日期：2025-08-22

作者：Aykut Sirma

摘要：Recent advancements in computer vision and deep learning have enhanced disaster\-response capabilities, particularly in the rapid assessment of earthquake\-affected urban environments. Timely identification of accessible entry points and structural obstacles is essential for effective search\-and\-rescue \(SAR\) operations. To address this need, we introduce DRespNeT, a high\-resolution dataset specifically developed for aerial instance segmentation of post\-earthquake structural environments. Unlike existing datasets, which rely heavily on satellite imagery or coarse semantic labeling, DRespNeT provides detailed polygon\-level instance segmentation annotations derived from high\-definition \(1080p\) aerial footage captured in disaster zones, including the 2023 Turkiye earthquake and other impacted regions. The dataset comprises 28 operationally critical classes, including structurally compromised buildings, access points such as doors, windows, and gaps, multiple debris levels, rescue personnel, vehicles, and civilian visibility. A distinctive feature of DRespNeT is its fine\-grained annotation detail, enabling differentiation between accessible and obstructed areas, thereby improving operational planning and response efficiency. Performance evaluations using YOLO\-based instance segmentation models, specifically YOLOv8\-seg, demonstrate significant gains in real\-time situational awareness and decision\-making. Our optimized YOLOv8\-DRN model achieves 92.7% mAP50 with an inference speed of 27 FPS on an RTX\-4090 GPU for multi\-target detection, meeting real\-time operational requirements. The dataset and models support SAR teams and robotic systems, providing a foundation for enhancing human\-robot collaboration, streamlining emergency response, and improving survivor outcomes.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2508.16016v1)

---


## A Novel Attention\-Augmented Wavelet YOLO System for Real\-time Brain Vessel Segmentation on Transcranial Color\-coded Doppler / 

发布日期：2025-08-19

作者：Wenxuan Zhang

摘要：The Circle of Willis \(CoW\), vital for ensuring consistent blood flow to the brain, is closely linked to ischemic stroke. Accurate assessment of the CoW is important for identifying individuals at risk and guiding appropriate clinical management. Among existing imaging methods, Transcranial Color\-coded Doppler \(TCCD\) offers unique advantages due to its radiation\-free nature, affordability, and accessibility. However, reliable TCCD assessments depend heavily on operator expertise for identifying anatomical landmarks and performing accurate angle correction, which limits its widespread adoption. To address this challenge, we propose an AI\-powered, real\-time CoW auto\-segmentation system capable of efficiently capturing cerebral arteries. No prior studies have explored AI\-driven cerebrovascular segmentation using TCCD. In this work, we introduce a novel Attention\-Augmented Wavelet YOLO \(AAW\-YOLO\) network tailored for TCCD data, designed to provide real\-time guidance for brain vessel segmentation in the CoW. We prospectively collected TCCD data comprising 738 annotated frames and 3,419 labeled artery instances to establish a high\-quality dataset for model training and evaluation. The proposed AAW\-YOLO demonstrated strong performance in segmenting both ipsilateral and contralateral CoW vessels, achieving an average Dice score of 0.901, IoU of 0.823, precision of 0.882, recall of 0.926, and mAP of 0.953, with a per\-frame inference speed of 14.199 ms. This system offers a practical solution to reduce reliance on operator experience in TCCD\-based cerebrovascular screening, with potential applications in routine clinical workflows and resource\-constrained settings. Future research will explore bilateral modeling and larger\-scale validation.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2508.13875v1)

---


## Mechanical Automation with Vision: A Design for Rubik's Cube Solver / 

发布日期：2025-08-17

作者：Abhinav Chalise

摘要：The core mechanical system is built around three stepper motors for physical manipulation, a microcontroller for hardware control, a camera and YOLO detection model for real\-time cube state detection. A significant software component is the development of a user\-friendly graphical user interface \(GUI\) designed in Unity. The initial state after detection from real\-time YOLOv8 model \(Precision 0.98443, Recall 0.98419, Box Loss 0.42051, Class Loss 0.2611\) is virtualized on GUI. To get the solution, the system employs the Kociemba's algorithm while physical manipulation with a single degree of freedom is done by combination of stepper motors' interaction with the cube achieving the average solving time of ~2.2 minutes.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2508.12469v1)

---


## TACR\-YOLO: A Real\-time Detection Framework for Abnormal Human Behaviors Enhanced with Coordinate and Task\-Aware Representations / 

发布日期：2025-08-15

作者：Xinyi Yin

摘要：Abnormal Human Behavior Detection \(AHBD\) under special scenarios is becoming increasingly crucial. While YOLO\-based detection methods excel in real\-time tasks, they remain hindered by challenges including small objects, task conflicts, and multi\-scale fusion in AHBD. To tackle them, we propose TACR\-YOLO, a new real\-time framework for AHBD. We introduce a Coordinate Attention Module to enhance small object detection, a Task\-Aware Attention Module to deal with classification\-regression conflicts, and a Strengthen Neck Network for refined multi\-scale fusion, respectively. In addition, we optimize Anchor Box sizes using K\-means clustering and deploy DIoU\-Loss to improve bounding box regression. The Personnel Anomalous Behavior Detection \(PABD\) dataset, which includes 8,529 samples across four behavior categories, is also presented. Extensive experimental results indicate that TACR\-YOLO achieves 91.92% mAP on PABD, with competitive speed and robustness. Ablation studies highlight the contribution of each improvement. This work provides new insights for abnormal behavior detection under special scenarios, advancing its progress.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2508.11478v1)

---

