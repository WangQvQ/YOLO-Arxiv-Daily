# 每日从arXiv中获取最新YOLO相关论文


## Anomaly\-Aware YOLO: A Frugal yet Robust Approach to Infrared Small Target Detection / 

发布日期：2025-10-06

作者：Alina Ciocarlan

摘要：Infrared Small Target Detection \(IRSTD\) is a challenging task in defense applications, where complex backgrounds and tiny target sizes often result in numerous false alarms using conventional object detectors. To overcome this limitation, we propose Anomaly\-Aware YOLO \(AA\-YOLO\), which integrates a statistical anomaly detection test into its detection head. By treating small targets as unexpected patterns against the background, AA\-YOLO effectively controls the false alarm rate. Our approach not only achieves competitive performance on several IRSTD benchmarks, but also demonstrates remarkable robustness in scenarios with limited training data, noise, and domain shifts. Furthermore, since only the detection head is modified, our design is highly generic and has been successfully applied across various YOLO backbones, including lightweight models. It also provides promising results when integrated into an instance segmentation YOLO. This versatility makes AA\-YOLO an attractive solution for real\-world deployments where resources are constrained. The code will be publicly released.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2510.04741v1)

---


## Bio\-Inspired Robotic Houbara: From Development to Field Deployment for Behavioral Studies / 

发布日期：2025-10-06

作者：Lyes Saad Saoud

摘要：Biomimetic intelligence and robotics are transforming field ecology by enabling lifelike robotic surrogates that interact naturally with animals under real world conditions. Studying avian behavior in the wild remains challenging due to the need for highly realistic morphology, durable outdoor operation, and intelligent perception that can adapt to uncontrolled environments. We present a next generation bio inspired robotic platform that replicates the morphology and visual appearance of the female Houbara bustard to support controlled ethological studies and conservation oriented field research. The system introduces a fully digitally replicable fabrication workflow that combines high resolution structured light 3D scanning, parametric CAD modelling, articulated 3D printing, and photorealistic UV textured vinyl finishing to achieve anatomically accurate and durable robotic surrogates. A six wheeled rocker bogie chassis ensures stable mobility on sand and irregular terrain, while an embedded NVIDIA Jetson module enables real time RGB and thermal perception, lightweight YOLO based detection, and an autonomous visual servoing loop that aligns the robot's head toward detected targets without human intervention. A lightweight thermal visible fusion module enhances perception in low light conditions. Field trials in desert aviaries demonstrated reliable real time operation at 15 to 22 FPS with latency under 100 ms and confirmed that the platform elicits natural recognition and interactive responses from live Houbara bustards under harsh outdoor conditions. This integrated framework advances biomimetic field robotics by uniting reproducible digital fabrication, embodied visual intelligence, and ecological validation, providing a transferable blueprint for animal robot interaction research, conservation robotics, and public engagement.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2510.04692v1)

---


## Road Damage and Manhole Detection using Deep Learning for Smart Cities: A Polygonal Annotation Approach / 

发布日期：2025-10-04

作者：Rasel Hossen

摘要：Urban safety and infrastructure maintenance are critical components of smart city development. Manual monitoring of road damages is time\-consuming, highly costly, and error\-prone. This paper presents a deep learning approach for automated road damage and manhole detection using the YOLOv9 algorithm with polygonal annotations. Unlike traditional bounding box annotation, we employ polygonal annotations for more precise localization of road defects. We develop a novel dataset comprising more than one thousand images which are mostly collected from Dhaka, Bangladesh. This dataset is used to train a YOLO\-based model for three classes, namely Broken, Not Broken, and Manhole. We achieve 78.1% overall image\-level accuracy. The YOLOv9 model demonstrates strong performance for Broken \(86.7% F1\-score\) and Not Broken \(89.2% F1\-score\) classes, with challenges in Manhole detection \(18.2% F1\-score\) due to class imbalance. Our approach offers an efficient and scalable solution for monitoring urban infrastructure in developing countries.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2510.03797v1)

---


## Real\-Time Threaded Houbara Detection and Segmentation for Wildlife Conservation using Mobile Platforms / 

发布日期：2025-10-03

作者：Lyes Saad Saoud

摘要：Real\-time animal detection and segmentation in natural environments are vital for wildlife conservation, enabling non\-invasive monitoring through remote camera streams. However, these tasks remain challenging due to limited computational resources and the cryptic appearance of many species. We propose a mobile\-optimized two\-stage deep learning framework that integrates a Threading Detection Model \(TDM\) to parallelize YOLOv10\-based detection and MobileSAM\-based segmentation. Unlike prior YOLO\+SAM pipelines, our approach improves real\-time performance by reducing latency through threading. YOLOv10 handles detection while MobileSAM performs lightweight segmentation, both executed concurrently for efficient resource use. On the cryptic Houbara Bustard, a conservation\-priority species, our model achieves mAP50 of 0.9627, mAP75 of 0.7731, mAP95 of 0.7178, and a MobileSAM mIoU of 0.7421. YOLOv10 operates at 43.7 ms per frame, confirming real\-time readiness. We introduce a curated Houbara dataset of 40,000 annotated images to support model training and evaluation across diverse conditions. The code and dataset used in this study are publicly available on GitHub at https://github.com/LyesSaadSaoud/mobile\-houbara\-detseg. For interactive demos and additional resources, visit https://lyessaadsaoud.github.io/LyesSaadSaoud\-Threaded\-YOLO\-SAM\-Houbara.

中文摘要：


代码链接：https://github.com/LyesSaadSaoud/mobile-houbara-detseg.，https://lyessaadsaoud.github.io/LyesSaadSaoud-Threaded-YOLO-SAM-Houbara.

论文链接：[阅读更多](http://arxiv.org/abs/2510.03501v1)

---


## Automated Defect Detection for Mass\-Produced Electronic Components Based on YOLO Object Detection Models / 

发布日期：2025-10-02

作者：Wei\-Lung Mao

摘要：Since the defect detection of conventional industry components is time\-consuming and labor\-intensive, it leads to a significant burden on quality inspection personnel and makes it difficult to manage product quality. In this paper, we propose an automated defect detection system for the dual in\-line package \(DIP\) that is widely used in industry, using digital camera optics and a deep learning \(DL\)\-based model. The two most common defect categories of DIP are examined: \(1\) surface defects, and \(2\) pin\-leg defects. However, the lack of defective component images leads to a challenge for detection tasks. To solve this problem, the ConSinGAN is used to generate a suitable\-sized dataset for training and testing. Four varieties of the YOLO model are investigated \(v3, v4, v7, and v9\), both in isolation and with the ConSinGAN augmentation. The proposed YOLOv7 with ConSinGAN is superior to the other YOLO versions in accuracy of 95.50%, detection time of 285 ms, and is far superior to threshold\-based approaches. In addition, the supervisory control and data acquisition \(SCADA\) system is developed, and the associated sensor architecture is described. The proposed automated defect detection can be easily established with numerous types of defects or insufficient defect data.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2510.01914v2)

---

