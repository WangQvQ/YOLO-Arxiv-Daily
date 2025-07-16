# 每日从arXiv中获取最新YOLO相关论文


## Automatic Road Subsurface Distress Recognition from Ground Penetrating Radar Images using Deep Learning\-based Cross\-verification / 

发布日期：2025-07-15

作者：Chang Peng

摘要：Ground penetrating radar \(GPR\) has become a rapid and non\-destructive solution for road subsurface distress \(RSD\) detection. However, RSD recognition from GPR images is labor\-intensive and heavily relies on inspectors' expertise. Deep learning offers the possibility for automatic RSD recognition, but its current performance is limited by two factors: Scarcity of high\-quality dataset for network training and insufficient capability of network to distinguish RSD. In this study, a rigorously validated 3D GPR dataset containing 2134 samples of diverse types was constructed through field scanning. Based on the finding that the YOLO model trained with one of the three scans of GPR images exhibits varying sensitivity to specific type of RSD, we proposed a novel cross\-verification strategy with outstanding accuracy in RSD recognition, achieving recall over 98.6% in field tests. The approach, integrated into an online RSD detection system, can reduce the labor of inspection by around 90%.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2507.11081v1)

---


## A Lightweight and Robust Framework for Real\-Time Colorectal Polyp Detection Using LOF\-Based Preprocessing and YOLO\-v11n / 

发布日期：2025-07-14

作者：Saadat Behzadi

摘要：Objectives: Timely and accurate detection of colorectal polyps plays a crucial role in diagnosing and preventing colorectal cancer, a major cause of mortality worldwide. This study introduces a new, lightweight, and efficient framework for polyp detection that combines the Local Outlier Factor \(LOF\) algorithm for filtering noisy data with the YOLO\-v11n deep learning model.   Study design: An experimental study leveraging deep learning and outlier removal techniques across multiple public datasets.   Methods: The proposed approach was tested on five diverse and publicly available datasets: CVC\-ColonDB, CVC\-ClinicDB, Kvasir\-SEG, ETIS, and EndoScene. Since these datasets originally lacked bounding box annotations, we converted their segmentation masks into suitable detection labels. To enhance the robustness and generalizability of our model, we apply 5\-fold cross\-validation and remove anomalous samples using the LOF method configured with 30 neighbors and a contamination ratio of 5%. Cleaned data are then fed into YOLO\-v11n, a fast and resource\-efficient object detection architecture optimized for real\-time applications. We train the model using a combination of modern augmentation strategies to improve detection accuracy under diverse conditions.   Results: Our approach significantly improves polyp localization performance, achieving a precision of 95.83%, recall of 91.85%, F1\-score of 93.48%, mAP@0.5 of 96.48%, and mAP@0.5:0.95 of 77.75%. Compared to previous YOLO\-based methods, our model demonstrates enhanced accuracy and efficiency.   Conclusions: These results suggest that the proposed method is well\-suited for real\-time colonoscopy support in clinical settings. Overall, the study underscores how crucial data preprocessing and model efficiency are when designing effective AI systems for medical imaging.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2507.10864v1)

---


## EA: An Event Autoencoder for High\-Speed Vision Sensing / 

发布日期：2025-07-09

作者：Riadul Islam

摘要：High\-speed vision sensing is essential for real\-time perception in applications such as robotics, autonomous vehicles, and industrial automation. Traditional frame\-based vision systems suffer from motion blur, high latency, and redundant data processing, limiting their performance in dynamic environments. Event cameras, which capture asynchronous brightness changes at the pixel level, offer a promising alternative but pose challenges in object detection due to sparse and noisy event streams. To address this, we propose an event autoencoder architecture that efficiently compresses and reconstructs event data while preserving critical spatial and temporal features. The proposed model employs convolutional encoding and incorporates adaptive threshold selection and a lightweight classifier to enhance recognition accuracy while reducing computational complexity. Experimental results on the existing Smart Event Face Dataset \(SEFD\) demonstrate that our approach achieves comparable accuracy to the YOLO\-v4 model while utilizing up to $35.5times$ fewer parameters. Implementations on embedded platforms, including Raspberry Pi 4B and NVIDIA Jetson Nano, show high frame rates ranging from 8 FPS up to 44.8 FPS. The proposed classifier exhibits up to 87.84x better FPS than the state\-of\-the\-art and significantly improves event\-based vision performance, making it ideal for low\-power, high\-speed applications in real\-time edge computing.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2507.06459v1)

---


## ECORE: Energy\-Conscious Optimized Routing for Deep Learning Models at the Edge / 

发布日期：2025-07-08

作者：Daghash K. Alqahtani

摘要：Edge computing enables data processing closer to the source, significantly reducing latency an essential requirement for real\-time vision\-based analytics such as object detection in surveillance and smart city environments. However, these tasks place substantial demands on resource constrained edge devices, making the joint optimization of energy consumption and detection accuracy critical. To address this challenge, we propose ECORE, a framework that integrates multiple dynamic routing strategies including estimation based techniques and a greedy selection algorithm to direct image processing requests to the most suitable edge device\-model pair. ECORE dynamically balances energy efficiency and detection performance based on object characteristics. We evaluate our approach through extensive experiments on real\-world datasets, comparing the proposed routers against widely used baseline techniques. The evaluation leverages established object detection models \(YOLO, SSD, EfficientDet\) and diverse edge platforms, including Jetson Orin Nano, Raspberry Pi 4 and 5, and TPU accelerators. Results demonstrate that our proposed context\-aware routing strategies can reduce energy consumption and latency by 45% and 49%, respectively, while incurring only a 2% loss in detection accuracy compared to accuracy\-centric methods.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2507.06011v2)

---


## YOLO\-APD: Enhancing YOLOv8 for Robust Pedestrian Detection on Complex Road Geometries / 

发布日期：2025-07-07

作者：Aquino Joctum

摘要：Autonomous vehicle perception systems require robust pedestrian detection, particularly on geometrically complex roadways like Type\-S curved surfaces, where standard RGB camera\-based methods face limitations. This paper introduces YOLO\-APD, a novel deep learning architecture enhancing the YOLOv8 framework specifically for this challenge. YOLO\-APD integrates several key architectural modifications: a parameter\-free SimAM attention mechanism, computationally efficient C3Ghost modules, a novel SimSPPF module for enhanced multi\-scale feature pooling, the Mish activation function for improved optimization, and an Intelligent Gather & Distribute \(IGD\) module for superior feature fusion in the network's neck. The concept of leveraging vehicle steering dynamics for adaptive region\-of\-interest processing is also presented. Comprehensive evaluations on a custom CARLA dataset simulating complex scenarios demonstrate that YOLO\-APD achieves state\-of\-the\-art detection accuracy, reaching 77.7% mAP@0.5:0.95 and exceptional pedestrian recall exceeding 96%, significantly outperforming baseline models, including YOLOv8. Furthermore, it maintains real\-time processing capabilities at 100 FPS, showcasing a superior balance between accuracy and efficiency. Ablation studies validate the synergistic contribution of each integrated component. Evaluation on the KITTI dataset confirms the architecture's potential while highlighting the need for domain adaptation. This research advances the development of highly accurate, efficient, and adaptable perception systems based on cost\-effective sensors, contributing to enhanced safety and reliability for autonomous navigation in challenging, less\-structured driving environments.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2507.05376v1)

---

