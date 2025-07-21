# 每日从arXiv中获取最新YOLO相关论文


## SOD\-YOLO: Enhancing YOLO\-Based Detection of Small Objects in UAV Imagery / 

发布日期：2025-07-17

作者：Peijun Wang

摘要：Small object detection remains a challenging problem in the field of object detection. To address this challenge, we propose an enhanced YOLOv8\-based model, SOD\-YOLO. This model integrates an ASF mechanism in the neck to enhance multi\-scale feature fusion, adds a Small Object Detection Layer \(named P2\) to provide higher\-resolution feature maps for better small object detection, and employs Soft\-NMS to refine confidence scores and retain true positives. Experimental results demonstrate that SOD\-YOLO significantly improves detection performance, achieving a 36.1% increase in mAP$\_\{50:95\}$ and 20.6% increase in mAP$\_\{50\}$ on the VisDrone2019\-DET dataset compared to the baseline model. These enhancements make SOD\-YOLO a practical and efficient solution for small object detection in UAV imagery. Our source code, hyper\-parameters, and model weights are available at https://github.com/iamwangxiaobai/SOD\-YOLO.

中文摘要：


代码链接：https://github.com/iamwangxiaobai/SOD-YOLO.

论文链接：[阅读更多](http://arxiv.org/abs/2507.12727v1)

---


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


## Landmark Detection for Medical Images using a General\-purpose Segmentation Model / 

发布日期：2025-07-13

作者：Ekaterina Stansfield

摘要：Radiographic images are a cornerstone of medical diagnostics in orthopaedics, with anatomical landmark detection serving as a crucial intermediate step for information extraction. General\-purpose foundational segmentation models, such as SAM \(Segment Anything Model\), do not support landmark segmentation out of the box and require prompts to function. However, in medical imaging, the prompts for landmarks are highly specific. Since SAM has not been trained to recognize such landmarks, it cannot generate accurate landmark segmentations for diagnostic purposes. Even MedSAM, a medically adapted variant of SAM, has been trained to identify larger anatomical structures, such as organs and their parts, and lacks the fine\-grained precision required for orthopaedic pelvic landmarks. To address this limitation, we propose leveraging another general\-purpose, non\-foundational model: YOLO. YOLO excels in object detection and can provide bounding boxes that serve as input prompts for SAM. While YOLO is efficient at detection, it is significantly outperformed by SAM in segmenting complex structures. In combination, these two models form a reliable pipeline capable of segmenting not only a small pilot set of eight anatomical landmarks but also an expanded set of 72 landmarks and 16 regions with complex outlines, such as the femoral cortical bone and the pelvic inlet. By using YOLO\-generated bounding boxes to guide SAM, we trained the hybrid model to accurately segment orthopaedic pelvic radiographs. Our results show that the proposed combination of YOLO and SAM yields excellent performance in detecting anatomical landmarks and intricate outlines in orthopaedic pelvic radiographs.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2507.11551v1)

---


## Butter: Frequency Consistency and Hierarchical Fusion for Autonomous Driving Object Detection / 

发布日期：2025-07-12

作者：Xiaojian Lin

摘要：Hierarchical feature representations play a pivotal role in computer vision, particularly in object detection for autonomous driving. Multi\-level semantic understanding is crucial for accurately identifying pedestrians, vehicles, and traffic signs in dynamic environments. However, existing architectures, such as YOLO and DETR, struggle to maintain feature consistency across different scales while balancing detection precision and computational efficiency. To address these challenges, we propose Butter, a novel object detection framework designed to enhance hierarchical feature representations for improving detection robustness. Specifically, Butter introduces two key innovations: Frequency\-Adaptive Feature Consistency Enhancement \(FAFCE\) Component, which refines multi\-scale feature consistency by leveraging adaptive frequency filtering to enhance structural and boundary precision, and Progressive Hierarchical Feature Fusion Network \(PHFFNet\) Module, which progressively integrates multi\-level features to mitigate semantic gaps and strengthen hierarchical feature learning. Through extensive experiments on BDD100K, KITTI, and Cityscapes, Butter demonstrates superior feature representation capabilities, leading to notable improvements in detection accuracy while reducing model complexity. By focusing on hierarchical feature refinement and integration, Butter provides an advanced approach to object detection that achieves a balance between accuracy, deployability, and computational efficiency in real\-time autonomous driving scenarios. Our model and implementation are publicly available at https://github.com/Aveiro\-Lin/Butter, facilitating further research and validation within the autonomous driving community.

中文摘要：


代码链接：https://github.com/Aveiro-Lin/Butter,

论文链接：[阅读更多](http://arxiv.org/abs/2507.13373v1)

---

