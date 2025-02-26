# 每日从arXiv中获取最新YOLO相关论文


## Automatic Vehicle Detection using DETR: A Transformer\-Based Approach for Navigating Treacherous Roads / 

发布日期：2025-02-25

作者：Istiaq Ahmed Fahad

摘要：Automatic Vehicle Detection \(AVD\) in diverse driving environments presents unique challenges due to varying lighting conditions, road types, and vehicle types. Traditional methods, such as YOLO and Faster R\-CNN, often struggle to cope with these complexities. As computer vision evolves, combining Convolutional Neural Networks \(CNNs\) with Transformer\-based approaches offers promising opportunities for improving detection accuracy and efficiency. This study is the first to experiment with Detection Transformer \(DETR\) for automatic vehicle detection in complex and varied settings. We employ a Collaborative Hybrid Assignments Training scheme, Co\-DETR, to enhance feature learning and attention mechanisms in DETR. By leveraging versatile label assignment strategies and introducing multiple parallel auxiliary heads, we provide more effective supervision during training and extract positive coordinates to boost training efficiency. Through extensive experiments on DETR variants and YOLO models, conducted using the BadODD dataset, we demonstrate the advantages of our approach. Our method achieves superior results, and improved accuracy in diverse conditions, making it practical for real\-world deployment. This work significantly advances autonomous navigation technology and opens new research avenues in object detection for autonomous vehicles. By integrating the strengths of CNNs and Transformers, we highlight the potential of DETR for robust and efficient vehicle detection in challenging driving environments.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2502.17843v1)

---


## Experimental validation of UAV search and detection system in real wilderness environment / 

发布日期：2025-02-24

作者：Stella Dumenčić

摘要：Search and rescue \(SAR\) missions require reliable search methods to locate survivors, especially in challenging or inaccessible environments. This is why introducing unmanned aerial vehicles \(UAVs\) can be of great help to enhance the efficiency of SAR missions while simultaneously increasing the safety of everyone involved in the mission. Motivated by this, we design and experiment with autonomous UAV search for humans in a Mediterranean karst environment. The UAVs are directed using Heat equation\-driven area coverage \(HEDAC\) ergodic control method according to known probability density and detection function. The implemented sensing framework consists of a probabilistic search model, motion control system, and computer vision object detection. It enables calculation of the probability of the target being detected in the SAR mission, and this paper focuses on experimental validation of proposed probabilistic framework and UAV control. The uniform probability density to ensure the even probability of finding the targets in the desired search area is achieved by assigning suitably thought\-out tasks to 78 volunteers. The detection model is based on YOLO and trained with a previously collected ortho\-photo image database. The experimental search is carefully planned and conducted, while as many parameters as possible are recorded. The thorough analysis consists of the motion control system, object detection, and the search validation. The assessment of the detection and search performance provides strong indication that the designed detection model in the UAV control algorithm is aligned with real\-world results.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2502.17372v1)

---


## Efficient Semantic\-aware Encryption for Secure Communications in Intelligent Connected Vehicles / 

发布日期：2025-02-23

作者：Bizhu Wang

摘要：Semantic communication \(SemCom\) significantly improves inter\-vehicle interactions in intelligent connected vehicles \(ICVs\) within limited wireless spectrum. However, the open nature of wireless communications introduces eavesdropping risks. To mitigate this, we propose the Efficient Semantic\-aware Encryption \(ESAE\) mechanism, integrating cryptography into SemCom to secure semantic transmission without complex key management. ESAE leverages semantic reciprocity between source and reconstructed information from past communications to independently generate session keys at both ends, reducing key transmission costs and associated security risks. Additionally, ESAE introduces a semantic\-aware key pre\-processing method \(SA\-KP\) using the YOLO\-v10 model to extract consistent semantics from bit\-level diverse yet semantically identical content, ensuring key consistency. Experimental results validate ESAE's effectiveness and feasibility under various wireless conditions, with key performance factors discussed.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2502.16400v1)

---


## Soybean pod and seed counting in both outdoor fields and indoor laboratories using unions of deep neural networks / 

发布日期：2025-02-21

作者：Tianyou Jiang

摘要：Automatic counting soybean pods and seeds in outdoor fields allows for rapid yield estimation before harvesting, while indoor laboratory counting offers greater accuracy. Both methods can significantly accelerate the breeding process. However, it remains challenging for accurately counting pods and seeds in outdoor fields, and there are still no accurate enough tools for counting pods and seeds in laboratories. In this study, we developed efficient deep learning models for counting soybean pods and seeds in both outdoor fields and indoor laboratories. For outdoor fields, annotating not only visible seeds but also occluded seeds makes YOLO have the ability to estimate the number of soybean seeds that are occluded. Moreover, we enhanced YOLO architecture by integrating it with HQ\-SAM \(YOLO\-SAM\), and domain adaptation techniques \(YOLO\-DA\), to improve model robustness and generalization across soybean images taken in outdoor fields. Testing on soybean images from the outdoor field, we achieved a mean absolute error \(MAE\) of 6.13 for pod counting and 10.05 for seed counting. For the indoor setting, we utilized Mask\-RCNN supplemented with a Swin Transformer module \(Mask\-RCNN\-Swin\), models were trained exclusively on synthetic training images generated from a small set of labeled data. This approach resulted in near\-perfect accuracy, with an MAE of 1.07 for pod counting and 1.33 for seed counting across actual laboratory images from two distinct studies.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2502.15286v1)

---


## ODVerse33: Is the New YOLO Version Always Better? A Multi Domain benchmark from YOLO v5 to v11 / 

发布日期：2025-02-20

作者：Tianyou Jiang

摘要：You Look Only Once \(YOLO\) models have been widely used for building real\-time object detectors across various domains. With the increasing frequency of new YOLO versions being released, key questions arise. Are the newer versions always better than their previous versions? What are the core innovations in each YOLO version and how do these changes translate into real\-world performance gains? In this paper, we summarize the key innovations from YOLOv1 to YOLOv11, introduce a comprehensive benchmark called ODverse33, which includes 33 datasets spanning 11 diverse domains \(Autonomous driving, Agricultural, Underwater, Medical, Videogame, Industrial, Aerial, Wildlife, Retail, Microscopic, and Security\), and explore the practical impact of model improvements in real\-world, multi\-domain applications through extensive experimental results. We hope this study can provide some guidance to the extensive users of object detection models and give some references for future real\-time object detector development.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2502.14314v1)

---

