# 每日从arXiv中获取最新YOLO相关论文


## Grasp\-HGN: Grasping the Unexpected / 

发布日期：2025-08-11

作者：Mehrshad Zandigohar

摘要：For transradial amputees, robotic prosthetic hands promise to regain the capability to perform daily living activities. To advance next\-generation prosthetic hand control design, it is crucial to address current shortcomings in robustness to out of lab artifacts, and generalizability to new environments. Due to the fixed number of object to interact with in existing datasets, contrasted with the virtually infinite variety of objects encountered in the real world, current grasp models perform poorly on unseen objects, negatively affecting users' independence and quality of life.   To address this: \(i\) we define semantic projection, the ability of a model to generalize to unseen object types and show that conventional models like YOLO, despite 80% training accuracy, drop to 15% on unseen objects. \(ii\) we propose Grasp\-LLaVA, a Grasp Vision Language Model enabling human\-like reasoning to infer the suitable grasp type estimate based on the object's physical characteristics resulting in a significant 50.2% accuracy over unseen object types compared to 36.7% accuracy of an SOTA grasp estimation model.   Lastly, to bridge the performance\-latency gap, we propose Hybrid Grasp Network \(HGN\), an edge\-cloud deployment infrastructure enabling fast grasp estimation on edge and accurate cloud inference as a fail\-safe, effectively expanding the latency vs. accuracy Pareto. HGN with confidence calibration \(DC\) enables dynamic switching between edge and cloud models, improving semantic projection accuracy by 5.6% \(to 42.3%\) with 3.5x speedup over the unseen object types. Over a real\-world sample mix, it reaches 86% average accuracy \(12.2% gain over edge\-only\), and 2.2x faster inference than Grasp\-LLaVA alone.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2508.07648v1)

---


## YOLOv8\-Based Deep Learning Model for Automated Poultry Disease Detection and Health Monitoring paper / 

发布日期：2025-08-06

作者：Akhil Saketh Reddy Sabbella

摘要：In the poultry industry, detecting chicken illnesses is essential to avoid financial losses. Conventional techniques depend on manual observation, which is laborious and prone to mistakes. Using YOLO v8 a deep learning model for real\-time object recognition. This study suggests an AI based approach, by developing a system that analyzes high resolution chicken photos, YOLO v8 detects signs of illness, such as abnormalities in behavior and appearance. A sizable, annotated dataset has been used to train the algorithm, which provides accurate real\-time identification of infected chicken and prompt warnings to farm operators for prompt action. By facilitating early infection identification, eliminating the need for human inspection, and enhancing biosecurity in large\-scale farms, this AI technology improves chicken health management. The real\-time features of YOLO v8 provide a scalable and effective method for improving farm management techniques.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2508.04658v1)

---


## Deep learning framework for crater detection and identification on the Moon and Mars / 

发布日期：2025-08-05

作者：Yihan Ma

摘要：Impact craters are among the most prominent geomorphological features on planetary surfaces and are of substantial significance in planetary science research. Their spatial distribution and morphological characteristics provide critical information on planetary surface composition, geological history, and impact processes. In recent years, the rapid advancement of deep learning models has fostered significant interest in automated crater detection. In this paper, we apply advancements in deep learning models for impact crater detection and identification. We use novel models, including Convolutional Neural Networks \(CNNs\) and variants such as YOLO and ResNet. We present a framework that features a two\-stage approach where the first stage features crater identification using simple classic CNN, ResNet\-50 and YOLO. In the second stage, our framework employs YOLO\-based detection for crater localisation. Therefore, we detect and identify different types of craters and present a summary report with remote sensing data for a selected region. We consider selected regions for craters and identification from Mars and the Moon based on remote sensing data. Our results indicate that YOLO demonstrates the most balanced crater detection performance, while ResNet\-50 excels in identifying large craters with high precision.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2508.03920v1)

---


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

