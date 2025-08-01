# 每日从arXiv中获取最新YOLO相关论文


## Towards Field\-Ready AI\-based Malaria Diagnosis: A Continual Learning Approach / 

发布日期：2025-07-31

作者：Louise Guillon

摘要：Malaria remains a major global health challenge, particularly in low\-resource settings where access to expert microscopy may be limited. Deep learning\-based computer\-aided diagnosis \(CAD\) systems have been developed and demonstrate promising performance on thin blood smear images. However, their clinical deployment may be hindered by limited generalization across sites with varying conditions. Yet very few practical solutions have been proposed. In this work, we investigate continual learning \(CL\) as a strategy to enhance the robustness of malaria CAD models to domain shifts. We frame the problem as a domain\-incremental learning scenario, where a YOLO\-based object detector must adapt to new acquisition sites while retaining performance on previously seen domains. We evaluate four CL strategies, two rehearsal\-based and two regularization\-based methods, on real\-life conditions thanks to a multi\-site clinical dataset of thin blood smear images. Our results suggest that CL, and rehearsal\-based methods in particular, can significantly improve performance. These findings highlight the potential of continual learning to support the development of deployable, field\-ready CAD tools for malaria.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2507.23648v1)

---


## Contrastive Learning\-Driven Traffic Sign Perception: Multi\-Modal Fusion of Text and Vision / 

发布日期：2025-07-31

作者：Qiang Lu

摘要：Traffic sign recognition, as a core component of autonomous driving perception systems, directly influences vehicle environmental awareness and driving safety. Current technologies face two significant challenges: first, the traffic sign dataset exhibits a pronounced long\-tail distribution, resulting in a substantial decline in recognition performance of traditional convolutional networks when processing low\-frequency and out\-of\-distribution classes; second, traffic signs in real\-world scenarios are predominantly small targets with significant scale variations, making it difficult to extract multi\-scale features.To overcome these issues, we propose a novel two\-stage framework combining open\-vocabulary detection and cross\-modal learning. For traffic sign detection, our NanoVerse YOLO model integrates a reparameterizable vision\-language path aggregation network \(RepVL\-PAN\) and an SPD\-Conv module to specifically enhance feature extraction for small, multi\-scale targets. For traffic sign classification, we designed a Traffic Sign Recognition Multimodal Contrastive Learning model \(TSR\-MCL\). By contrasting visual features from a Vision Transformer with semantic features from a rule\-based BERT, TSR\-MCL learns robust, frequency\-independent representations, effectively mitigating class confusion caused by data imbalance. On the TT100K dataset, our method achieves a state\-of\-the\-art 78.4% mAP in the long\-tail detection task for all\-class recognition. The model also obtains 91.8% accuracy and 88.9% recall, significantly outperforming mainstream algorithms and demonstrating superior accuracy and generalization in complex, open\-world scenarios.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2507.23331v1)

---


## YOLO\-ROC: A High\-Precision and Ultra\-Lightweight Model for Real\-Time Road Damage Detection / 

发布日期：2025-07-31

作者：Zicheng Lin

摘要：Road damage detection is a critical task for ensuring traffic safety and maintaining infrastructure integrity. While deep learning\-based detection methods are now widely adopted, they still face two core challenges: first, the inadequate multi\-scale feature extraction capabilities of existing networks for diverse targets like cracks and potholes, leading to high miss rates for small\-scale damage; and second, the substantial parameter counts and computational demands of mainstream models, which hinder their deployment for efficient, real\-time detection in practical applications. To address these issues, this paper proposes a high\-precision and lightweight model, YOLO \- Road Orthogonal Compact \(YOLO\-ROC\). We designed a Bidirectional Multi\-scale Spatial Pyramid Pooling Fast \(BMS\-SPPF\) module to enhance multi\-scale feature extraction and implemented a hierarchical channel compression strategy to reduce computational complexity. The BMS\-SPPF module leverages a bidirectional spatial\-channel attention mechanism to improve the detection of small targets. Concurrently, the channel compression strategy reduces the parameter count from 3.01M to 0.89M and GFLOPs from 8.1 to 2.6. Experiments on the RDD2022\_China\_Drone dataset demonstrate that YOLO\-ROC achieves a mAP50 of 67.6%, surpassing the baseline YOLOv8n by 2.11%. Notably, the mAP50 for the small\-target D40 category improved by 16.8%, and the final model size is only 2.0 MB. Furthermore, the model exhibits excellent generalization performance on the RDD2022\_China\_Motorbike dataset.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2507.23225v1)

---


## DriveIndia: An Object Detection Dataset for Diverse Indian Traffic Scenes / 

发布日期：2025-07-26

作者：Rishav Kumar

摘要：We introduce DriveIndia, a large\-scale object detection dataset purpose\-built to capture the complexity and unpredictability of Indian traffic environments. The dataset contains 66,986 high\-resolution images annotated in YOLO format across 24 traffic\-relevant object categories, encompassing diverse conditions such as varied weather \(fog, rain\), illumination changes, heterogeneous road infrastructure, and dense, mixed traffic patterns and collected over 120\+ hours and covering 3,400\+ kilometers across urban, rural, and highway routes. DriveIndia offers a comprehensive benchmark for real\-world autonomous driving challenges. We provide baseline results using state\-of\-the\-art YOLO family models, with the top\-performing variant achieving a mAP50 of 78.7%. Designed to support research in robust, generalizable object detection under uncertain road conditions, DriveIndia will be publicly available via the TiHAN\-IIT Hyderabad dataset repository \(https://tihan.iith.ac.in/tiand\-datasets/\).

中文摘要：


代码链接：https://tihan.iith.ac.in/tiand-datasets/).

论文链接：[阅读更多](http://arxiv.org/abs/2507.19912v2)

---


## Underwater Waste Detection Using Deep Learning A Performance Comparison of YOLOv7 to 10 and Faster RCNN / 

发布日期：2025-07-25

作者：UMMPK Nawarathne

摘要：Underwater pollution is one of today's most significant environmental concerns, with vast volumes of garbage found in seas, rivers, and landscapes around the world. Accurate detection of these waste materials is crucial for successful waste management, environmental monitoring, and mitigation strategies. In this study, we investigated the performance of five cutting\-edge object recognition algorithms, namely YOLO \(You Only Look Once\) models, including YOLOv7, YOLOv8, YOLOv9, YOLOv10, and Faster Region\-Convolutional Neural Network \(R\-CNN\), to identify which model was most effective at recognizing materials in underwater situations. The models were thoroughly trained and tested on a large dataset containing fifteen different classes under diverse conditions, such as low visibility and variable depths. From the above\-mentioned models, YOLOv8 outperformed the others, with a mean Average Precision \(mAP\) of 80.9%, indicating a significant performance. This increased performance is attributed to YOLOv8's architecture, which incorporates advanced features such as improved anchor\-free mechanisms and self\-supervised learning, allowing for more precise and efficient recognition of items in a variety of settings. These findings highlight the YOLOv8 model's potential as an effective tool in the global fight against pollution, improving both the detection capabilities and scalability of underwater cleanup operations.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2507.18967v1)

---

