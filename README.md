# 每日从arXiv中获取最新YOLO相关论文


## HOTSPOT\-YOLO: A Lightweight Deep Learning Attention\-Driven Model for Detecting Thermal Anomalies in Drone\-Based Solar Photovoltaic Inspections / 

发布日期：2025-08-26

作者：Mahmoud Dhimish

摘要：Thermal anomaly detection in solar photovoltaic \(PV\) systems is essential for ensuring operational efficiency and reducing maintenance costs. In this study, we developed and named HOTSPOT\-YOLO, a lightweight artificial intelligence \(AI\) model that integrates an efficient convolutional neural network backbone and attention mechanisms to improve object detection. This model is specifically designed for drone\-based thermal inspections of PV systems, addressing the unique challenges of detecting small and subtle thermal anomalies, such as hotspots and defective modules, while maintaining real\-time performance. Experimental results demonstrate a mean average precision of 90.8%, reflecting a significant improvement over baseline object detection models. With a reduced computational load and robustness under diverse environmental conditions, HOTSPOT\-YOLO offers a scalable and reliable solution for large\-scale PV inspections. This work highlights the integration of advanced AI techniques with practical engineering applications, revolutionizing automated fault detection in renewable energy systems.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2508.18912v1)

---


## LPLC: A Dataset for License Plate Legibility Classification / 

发布日期：2025-08-25

作者：Lucas Wojcik

摘要：Automatic License Plate Recognition \(ALPR\) faces a major challenge when dealing with illegible license plates \(LPs\). While reconstruction methods such as super\-resolution \(SR\) have emerged, the core issue of recognizing these low\-quality LPs remains unresolved. To optimize model performance and computational efficiency, image pre\-processing should be applied selectively to cases that require enhanced legibility. To support research in this area, we introduce a novel dataset comprising 10,210 images of vehicles with 12,687 annotated LPs for legibility classification \(the LPLC dataset\). The images span a wide range of vehicle types, lighting conditions, and camera/image quality levels. We adopt a fine\-grained annotation strategy that includes vehicle\- and LP\-level occlusions, four legibility categories \(perfect, good, poor, and illegible\), and character labels for three categories \(excluding illegible LPs\). As a benchmark, we propose a classification task using three image recognition networks to determine whether an LP image is good enough, requires super\-resolution, or is completely unrecoverable. The overall F1 score, which remained below 80% for all three baseline models \(ViT, ResNet, and YOLO\), together with the analyses of SR and LP recognition methods, highlights the difficulty of the task and reinforces the need for further research. The proposed dataset is publicly available at https://github.com/lmlwojcik/lplc\-dataset.

中文摘要：


代码链接：https://github.com/lmlwojcik/lplc-dataset.

论文链接：[阅读更多](http://arxiv.org/abs/2508.18425v1)

---


## Integration of Computer Vision with Adaptive Control for Autonomous Driving Using ADORE / 

发布日期：2025-08-25

作者：Abu Shad Ahammed

摘要：Ensuring safety in autonomous driving requires a seamless integration of perception and decision making under uncertain conditions. Although computer vision \(CV\) models such as YOLO achieve high accuracy in detecting traffic signs and obstacles, their performance degrades in drift scenarios caused by weather variations or unseen objects. This work presents a simulated autonomous driving system that combines a context aware CV model with adaptive control using the ADORE framework. The CARLA simulator was integrated with ADORE via the ROS bridge, allowing real\-time communication between perception, decision, and control modules. A simulated test case was designed in both clear and drift weather conditions to demonstrate the robust detection performance of the perception model while ADORE successfully adapted vehicle behavior to speed limits and obstacles with low response latency. The findings highlight the potential of coupling deep learning\-based perception with rule\-based adaptive decision making to improve automotive safety critical system.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2508.17985v1)

---


## Enhanced Drift\-Aware Computer Vision Architecture for Autonomous Driving / 

发布日期：2025-08-25

作者：Md Shahi Amran Hossain

摘要：The use of computer vision in automotive is a trending research in which safety and security are a primary concern. In particular, for autonomous driving, preventing road accidents requires highly accurate object detection under diverse conditions. To address this issue, recently the International Organization for Standardization \(ISO\) released the 8800 norm, providing structured frameworks for managing associated AI relevant risks. However, challenging scenarios such as adverse weather or low lighting often introduce data drift, leading to degraded model performance and potential safety violations. In this work, we present a novel hybrid computer vision architecture trained with thousands of synthetic image data from the road environment to improve robustness in unseen drifted environments. Our dual mode framework utilized YOLO version 8 for swift detection and incorporated a five\-layer CNN for verification. The system functioned in sequence and improved the detection accuracy by more than 90% when tested with drift\-augmented road images. The focus was to demonstrate how such a hybrid model can provide better road safety when working together in a hybrid structure.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2508.17975v1)

---


## A Synthetic Dataset for Manometry Recognition in Robotic Applications / 

发布日期：2025-08-24

作者：Pedro Antonio Rabelo Saraiva

摘要：This work addresses the challenges of data scarcity and high acquisition costs for training robust object detection models in complex industrial environments, such as offshore oil platforms. The practical and economic barriers to collecting real\-world data in these hazardous settings often hamper the development of autonomous inspection systems. To overcome this, in this work we propose and validate a hybrid data synthesis pipeline that combines procedural rendering with AI\-driven video generation. Our methodology leverages BlenderProc to create photorealistic images with precise annotations and controlled domain randomization, and integrates NVIDIA's Cosmos\-Predict2 world\-foundation model to synthesize physically plausible video sequences with temporal diversity, capturing rare viewpoints and adverse conditions. We demonstrate that a YOLO\-based detection network trained on a composite dataset, blending real images with our synthetic data, achieves superior performance compared to models trained exclusively on real\-world data. Notably, a 1:1 mixture of real and synthetic data yielded the highest accuracy, surpassing the real\-only baseline. These findings highlight the viability of a synthetic\-first approach as an efficient, cost\-effective, and safe alternative for developing reliable perception systems in safety\-critical and resource\-constrained industrial applications.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2508.17468v1)

---

