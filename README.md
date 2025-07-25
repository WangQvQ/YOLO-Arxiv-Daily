# 每日从arXiv中获取最新YOLO相关论文


## Real\-Time Object Detection and Classification using YOLO for Edge FPGAs / 

发布日期：2025-07-24

作者：Rashed Al Amin

摘要：Object detection and classification are crucial tasks across various application domains, particularly in the development of safe and reliable Advanced Driver Assistance Systems \(ADAS\). Existing deep learning\-based methods such as Convolutional Neural Networks \(CNNs\), Single Shot Detectors \(SSDs\), and You Only Look Once \(YOLO\) have demonstrated high performance in terms of accuracy and computational speed when deployed on Field\-Programmable Gate Arrays \(FPGAs\). However, despite these advances, state\-of\-the\-art YOLO\-based object detection and classification systems continue to face challenges in achieving resource efficiency suitable for edge FPGA platforms. To address this limitation, this paper presents a resource\-efficient real\-time object detection and classification system based on YOLOv5 optimized for FPGA deployment. The proposed system is trained on the COCO and GTSRD datasets and implemented on the Xilinx Kria KV260 FPGA board. Experimental results demonstrate a classification accuracy of 99%, with a power consumption of 3.5W and a processing speed of 9 frames per second \(FPS\). These findings highlight the effectiveness of the proposed approach in enabling real\-time, resource\-efficient object detection and classification for edge computing applications.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2507.18174v1)

---


## Bearded Dragon Activity Recognition Pipeline: An AI\-Based Approach to Behavioural Monitoring / 

发布日期：2025-07-23

作者：Arsen Yermukan

摘要：Traditional monitoring of bearded dragon \(Pogona Viticeps\) behaviour is time\-consuming and prone to errors. This project introduces an automated system for real\-time video analysis, using You Only Look Once \(YOLO\) object detection models to identify two key behaviours: basking and hunting. We trained five YOLO variants \(v5, v7, v8, v11, v12\) on a custom, publicly available dataset of 1200 images, encompassing bearded dragons \(600\), heating lamps \(500\), and crickets \(100\). YOLOv8s was selected as the optimal model due to its superior balance of accuracy \(mAP@0.5:0.95 = 0.855\) and speed. The system processes video footage by extracting per\-frame object coordinates, applying temporal interpolation for continuity, and using rule\-based logic to classify specific behaviours. Basking detection proved reliable. However, hunting detection was less accurate, primarily due to weak cricket detection \(mAP@0.5 = 0.392\). Future improvements will focus on enhancing cricket detection through expanded datasets or specialised small\-object detectors. This automated system offers a scalable solution for monitoring reptile behaviour in controlled environments, significantly improving research efficiency and data quality.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2507.17987v1)

---


## BetterCheck: Towards Safeguarding VLMs for Automotive Perception Systems / 

发布日期：2025-07-23

作者：Malsha Ashani Mahawatta Dona

摘要：Large language models \(LLMs\) are growingly extended to process multimodal data such as text and video simultaneously. Their remarkable performance in understanding what is shown in images is surpassing specialized neural networks \(NNs\) such as Yolo that is supporting only a well\-formed but very limited vocabulary, ie., objects that they are able to detect. When being non\-restricted, LLMs and in particular state\-of\-the\-art vision language models \(VLMs\) show impressive performance to describe even complex traffic situations. This is making them potentially suitable components for automotive perception systems to support the understanding of complex traffic situations or edge case situation. However, LLMs and VLMs are prone to hallucination, which mean to either potentially not seeing traffic agents such as vulnerable road users who are present in a situation, or to seeing traffic agents who are not there in reality. While the latter is unwanted making an ADAS or autonomous driving systems \(ADS\) to unnecessarily slow down, the former could lead to disastrous decisions from an ADS. In our work, we are systematically assessing the performance of 3 state\-of\-the\-art VLMs on a diverse subset of traffic situations sampled from the Waymo Open Dataset to support safety guardrails for capturing such hallucinations in VLM\-supported perception systems. We observe that both, proprietary and open VLMs exhibit remarkable image understanding capabilities even paying thorough attention to fine details sometimes difficult to spot for us humans. However, they are also still prone to making up elements in their descriptions to date requiring hallucination detection strategies such as BetterCheck that we propose in our work.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2507.17722v1)

---


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

