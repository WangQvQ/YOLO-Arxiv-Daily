# 每日从arXiv中获取最新YOLO相关论文


## DriveIndia: An Object Detection Dataset for Diverse Indian Traffic Scenes / 

发布日期：2025-07-26

作者：Rishav Kumar

摘要：We introduce textbf\{DriveIndia\}, a large\-scale object detection dataset purpose\-built to capture the complexity and unpredictability of Indian traffic environments. The dataset contains textbf\{66,986 high\-resolution images\} annotated in YOLO format across textbf\{24 traffic\-relevant object categories\}, encompassing diverse conditions such as varied weather \(fog, rain\), illumination changes, heterogeneous road infrastructure, and dense, mixed traffic patterns and collected over textbf\{120\+ hours\} and covering textbf\{3,400\+ kilometers\} across urban, rural, and highway routes. DriveIndia offers a comprehensive benchmark for real\-world autonomous driving challenges. We provide baseline results using state\-of\-the\-art textbf\{YOLO family models\}, with the top\-performing variant achieving a $mAP\_\{50\}$ of textbf\{78.7%\}. Designed to support research in robust, generalizable object detection under uncertain road conditions, DriveIndia will be publicly available via the TiHAN\-IIT Hyderabad dataset repository \(https://tihan.iith.ac.in/tiand\-datasets/\).

中文摘要：


代码链接：https://tihan.iith.ac.in/tiand-datasets/).

论文链接：[阅读更多](http://arxiv.org/abs/2507.19912v1)

---


## Underwater Waste Detection Using Deep Learning A Performance Comparison of YOLOv7 to 10 and Faster RCNN / 

发布日期：2025-07-25

作者：UMMPK Nawarathne

摘要：Underwater pollution is one of today's most significant environmental concerns, with vast volumes of garbage found in seas, rivers, and landscapes around the world. Accurate detection of these waste materials is crucial for successful waste management, environmental monitoring, and mitigation strategies. In this study, we investigated the performance of five cutting\-edge object recognition algorithms, namely YOLO \(You Only Look Once\) models, including YOLOv7, YOLOv8, YOLOv9, YOLOv10, and Faster Region\-Convolutional Neural Network \(R\-CNN\), to identify which model was most effective at recognizing materials in underwater situations. The models were thoroughly trained and tested on a large dataset containing fifteen different classes under diverse conditions, such as low visibility and variable depths. From the above\-mentioned models, YOLOv8 outperformed the others, with a mean Average Precision \(mAP\) of 80.9%, indicating a significant performance. This increased performance is attributed to YOLOv8's architecture, which incorporates advanced features such as improved anchor\-free mechanisms and self\-supervised learning, allowing for more precise and efficient recognition of items in a variety of settings. These findings highlight the YOLOv8 model's potential as an effective tool in the global fight against pollution, improving both the detection capabilities and scalability of underwater cleanup operations.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2507.18967v1)

---


## YOLO for Knowledge Extraction from Vehicle Images: A Baseline Study / 

发布日期：2025-07-25

作者：Saraa Al\-Saddik

摘要：Accurate identification of vehicle attributes such as make, colour, and shape is critical for law enforcement and intelligence applications. This study evaluates the effectiveness of three state\-of\-the\-art deep learning approaches YOLO\-v11, YOLO\-World, and YOLO\-Classification on a real\-world vehicle image dataset. This dataset was collected under challenging and unconstrained conditions by NSW Police Highway Patrol Vehicles. A multi\-view inference \(MVI\) approach was deployed to enhance the performance of the models' predictions. To conduct the analyses, datasets with 100,000 plus images were created for each of the three metadata prediction tasks, specifically make, shape and colour. The models were tested on a separate dataset with 29,937 images belonging to 1809 number plates. Different sets of experiments have been investigated by varying the models sizes. A classification accuracy of 93.70%, 82.86%, 85.19%, and 94.86% was achieved with the best performing make, shape, colour, and colour\-binary models respectively. It was concluded that there is a need to use MVI to get usable models within such complex real\-world datasets. Our findings indicated that the object detection models YOLO\-v11 and YOLO\-World outperformed classification\-only models in make and shape extraction. Moreover, smaller YOLO variants perform comparably to larger counterparts, offering substantial efficiency benefits for real\-time predictions. This work provides a robust baseline for extracting vehicle metadata in real\-world scenarios. Such models can be used in filtering and sorting user queries, minimising the time required to search large vehicle images datasets.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2507.18966v1)

---


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

