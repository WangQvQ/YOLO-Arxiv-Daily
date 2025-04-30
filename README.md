# 每日从arXiv中获取最新YOLO相关论文


## FBRT\-YOLO: Faster and Better for Real\-Time Aerial Image Detection / 

发布日期：2025-04-29

作者：Yao Xiao

摘要：Embedded flight devices with visual capabilities have become essential for a wide range of applications. In aerial image detection, while many existing methods have partially addressed the issue of small target detection, challenges remain in optimizing small target detection and balancing detection accuracy with efficiency. These issues are key obstacles to the advancement of real\-time aerial image detection. In this paper, we propose a new family of real\-time detectors for aerial image detection, named FBRT\-YOLO, to address the imbalance between detection accuracy and efficiency. Our method comprises two lightweight modules: Feature Complementary Mapping Module \(FCM\) and Multi\-Kernel Perception Unit\(MKP\), designed to enhance object perception for small targets in aerial images. FCM focuses on alleviating the problem of information imbalance caused by the loss of small target information in deep networks. It aims to integrate spatial positional information of targets more deeply into the network,better aligning with semantic information in the deeper layers to improve the localization of small targets. We introduce MKP, which leverages convolutions with kernels of different sizes to enhance the relationships between targets of various scales and improve the perception of targets at different scales. Extensive experimental results on three major aerial image datasets, including Visdrone, UAVDT, and AI\-TOD,demonstrate that FBRT\-YOLO outperforms various real\-time detectors in terms of performance and speed.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2504.20670v1)

---


## MASF\-YOLO: An Improved YOLOv11 Network for Small Object Detection on Drone View / 

发布日期：2025-04-25

作者：Liugang Lu

摘要：With the rapid advancement of Unmanned Aerial Vehicle \(UAV\) and computer vision technologies, object detection from UAV perspectives has emerged as a prominent research area. However, challenges for detection brought by the extremely small proportion of target pixels, significant scale variations of objects, and complex background information in UAV images have greatly limited the practical applications of UAV. To address these challenges, we propose a novel object detection network Multi\-scale Context Aggregation and Scale\-adaptive Fusion YOLO \(MASF\-YOLO\), which is developed based on YOLOv11. Firstly, to tackle the difficulty of detecting small objects in UAV images, we design a Multi\-scale Feature Aggregation Module \(MFAM\), which significantly improves the detection accuracy of small objects through parallel multi\-scale convolutions and feature fusion. Secondly, to mitigate the interference of background noise, we propose an Improved Efficient Multi\-scale Attention Module \(IEMA\), which enhances the focus on target regions through feature grouping, parallel sub\-networks, and cross\-spatial learning. Thirdly, we introduce a Dimension\-Aware Selective Integration Module \(DASI\), which further enhances multi\-scale feature fusion capabilities by adaptively weighting and fusing low\-dimensional features and high\-dimensional features. Finally, we conducted extensive performance evaluations of our proposed method on the VisDrone2019 dataset. Compared to YOLOv11\-s, MASFYOLO\-s achieves improvements of 4.6% in mAP@0.5 and 3.5% in mAP@0.5:0.95 on the VisDrone2019 validation set. Remarkably, MASF\-YOLO\-s outperforms YOLOv11\-m while requiring only approximately 60% of its parameters and 65% of its computational cost. Furthermore, comparative experiments with state\-of\-the\-art detectors confirm that MASF\-YOLO\-s maintains a clear competitive advantage in both detection accuracy and model efficiency.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2504.18136v1)

---


## A Decade of You Only Look Once \(YOLO\) for Object Detection / 

发布日期：2025-04-24

作者：Leo Thomas Ramos

摘要：This review marks the tenth anniversary of You Only Look Once \(YOLO\), one of the most influential frameworks in real\-time object detection. Over the past decade, YOLO has evolved from a streamlined detector into a diverse family of architectures characterized by efficient design, modular scalability, and cross\-domain adaptability. The paper presents a technical overview of the main versions, highlights key architectural trends, and surveys the principal application areas in which YOLO has been adopted. It also addresses evaluation practices, ethical considerations, and potential future directions for the framework's continued development. The analysis aims to provide a comprehensive and critical perspective on YOLO's trajectory and ongoing transformation.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2504.18586v1)

---


## Vision Controlled Orthotic Hand Exoskeleton / 

发布日期：2025-04-22

作者：Connor Blais

摘要：This paper presents the design and implementation of an AI vision\-controlled orthotic hand exoskeleton to enhance rehabilitation and assistive functionality for individuals with hand mobility impairments. The system leverages a Google Coral Dev Board Micro with an Edge TPU to enable real\-time object detection using a customized MobileNet\_V2 model trained on a six\-class dataset. The exoskeleton autonomously detects objects, estimates proximity, and triggers pneumatic actuation for grasp\-and\-release tasks, eliminating the need for user\-specific calibration needed in traditional EMG\-based systems. The design prioritizes compactness, featuring an internal battery. It achieves an 8\-hour runtime with a 1300 mAh battery. Experimental results demonstrate a 51ms inference speed, a significant improvement over prior iterations, though challenges persist in model robustness under varying lighting conditions and object orientations. While the most recent YOLO model \(YOLOv11\) showed potential with 15.4 FPS performance, quantization issues hindered deployment. The prototype underscores the viability of vision\-controlled exoskeletons for real\-world assistive applications, balancing portability, efficiency, and real\-time responsiveness, while highlighting future directions for model optimization and hardware miniaturization.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2504.16319v1)

---


## ISTD\-YOLO: A Multi\-Scale Lightweight High\-Performance Infrared Small Target Detection Algorithm / 

发布日期：2025-04-19

作者：Shang Zhang

摘要：Aiming at the detection difficulties of infrared images such as complex background, low signal\-to\-noise ratio, small target size and weak brightness, a lightweight infrared small target detection algorithm ISTD\-YOLO based on improved YOLOv7 was proposed. Firstly, the YOLOv7 network structure was lightweight reconstructed, and a three\-scale lightweight network architecture was designed. Then, the ELAN\-W module of the model neck network is replaced by VoV\-GSCSP to reduce the computational cost and the complexity of the network structure. Secondly, a parameter\-free attention mechanism was introduced into the neck network to enhance the relevance of local con\-text information. Finally, the Normalized Wasserstein Distance \(NWD\) was used to optimize the commonly used IoU index to enhance the localization and detection accuracy of small targets. Experimental results show that compared with YOLOv7 and the current mainstream algorithms, ISTD\-YOLO can effectively improve the detection effect, and all indicators are effectively improved, which can achieve high\-quality detection of infrared small targets.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2504.14289v1)

---

