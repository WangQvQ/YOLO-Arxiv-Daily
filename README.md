# 每日从arXiv中获取最新YOLO相关论文


## hYOLO Model: Enhancing Object Classification with Hierarchical Context in YOLOv8 / 

发布日期：2025-10-27

作者：Veska Tsenkova

摘要：Current convolution neural network \(CNN\) classification methods are predominantly focused on flat classification which aims solely to identify a specified object within an image. However, real\-world objects often possess a natural hierarchical organization that can significantly help classification tasks. Capturing the presence of relations between objects enables better contextual understanding as well as control over the severity of mistakes. Considering these aspects, this paper proposes an end\-to\-end hierarchical model for image detection and classification built upon the YOLO model family. A novel hierarchical architecture, a modified loss function, and a performance metric tailored to the hierarchical nature of the model are introduced. The proposed model is trained and evaluated on two different hierarchical categorizations of the same dataset: a systematic categorization that disregards visual similarities between objects and a categorization accounting for common visual characteristics across classes. The results illustrate how the suggested methodology addresses the inherent hierarchical structure present in real\-world objects, which conventional flat classification algorithms often overlook.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2510.23278v1)

---


## An Intelligent Water\-Saving Irrigation System Based on Multi\-Sensor Fusion and Visual Servoing Control / 

发布日期：2025-10-27

作者：ZhengKai Huang

摘要：This paper introduces an intelligent water\-saving irrigation system designed to address critical challenges in precision agriculture, such as inefficient water use and poor terrain adaptability. The system integrates advanced computer vision, robotic control, and real\-time stabilization technologies via a multi\-sensor fusion approach. A lightweight YOLO model, deployed on an embedded vision processor \(K210\), enables real\-time plant container detection with over 96% accuracy under varying lighting conditions. A simplified hand\-eye calibration algorithm\-designed for 'handheld camera' robot arm configurations\-ensures that the end effector can be precisely positioned, with a success rate exceeding 90%. The active leveling system, driven by the STM32F103ZET6 main control chip and JY901S inertial measurement data, can stabilize the irrigation platform on slopes up to 10 degrees, with a response time of 1.8 seconds. Experimental results across three simulated agricultural environments \(standard greenhouse, hilly terrain, complex lighting\) demonstrate a 30\-50% reduction in water consumption compared to conventional flood irrigation, with water use efficiency exceeding 92% in all test cases.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2510.23003v1)

---


## Human\-Centric Anomaly Detection in Surveillance Videos Using YOLO\-World and Spatio\-Temporal Deep Learning / 

发布日期：2025-10-24

作者：Mohammad Ali Etemadi Naeen

摘要：Anomaly detection in surveillance videos remains a challenging task due to the diversity of abnormal events, class imbalance, and scene\-dependent visual clutter. To address these issues, we propose a robust deep learning framework that integrates human\-centric preprocessing with spatio\-temporal modeling for multi\-class anomaly classification. Our pipeline begins by applying YOLO\-World \- an open\-vocabulary vision\-language detector \- to identify human instances in raw video clips, followed by ByteTrack for consistent identity\-aware tracking. Background regions outside detected bounding boxes are suppressed via Gaussian blurring, effectively reducing scene\-specific distractions and focusing the model on behaviorally relevant foreground content. The refined frames are then processed by an ImageNet\-pretrained InceptionV3 network for spatial feature extraction, and temporal dynamics are captured using a bidirectional LSTM \(BiLSTM\) for sequence\-level classification. Evaluated on a five\-class subset of the UCF\-Crime dataset \(Normal, Burglary, Fighting, Arson, Explosion\), our method achieves a mean test accuracy of 92.41% across three independent trials, with per\-class F1\-scores consistently exceeding 0.85. Comprehensive evaluation metrics \- including confusion matrices, ROC curves, and macro/weighted averages \- demonstrate strong generalization and resilience to class imbalance. The results confirm that foreground\-focused preprocessing significantly enhances anomaly discrimination in real\-world surveillance scenarios.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2510.22056v1)

---


## Deep learning\-based automated damage detection in concrete structures using images from earthquake events / 

发布日期：2025-10-24

作者：Abdullah Turer

摘要：Timely assessment of integrity of structures after seismic events is crucial for public safety and emergency response. This study focuses on assessing the structural damage conditions using deep learning methods to detect exposed steel reinforcement in concrete buildings and bridges after large earthquakes. Steel bars are typically exposed after concrete spalling or large flexural or shear cracks. The amount and distribution of exposed steel reinforcement is an indication of structural damage and degradation. To automatically detect exposed steel bars, new datasets of images collected after the 2023 Turkey Earthquakes were labeled to represent a wide variety of damaged concrete structures. The proposed method builds upon a deep learning framework, enhanced with fine\-tuning, data augmentation, and testing on public datasets. An automated classification framework is developed that can be used to identify inside/outside buildings and structural components. Then, a YOLOv11 \(You Only Look Once\) model is trained to detect cracking and spalling damage and exposed bars. Another YOLO model is finetuned to distinguish different categories of structural damage levels. All these trained models are used to create a hybrid framework to automatically and reliably determine the damage levels from input images. This research demonstrates that rapid and automated damage detection following disasters is achievable across diverse damage contexts by utilizing image data collection, annotation, and deep learning approaches.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2510.21063v1)

---


## Big Data, Tiny Targets: An Exploratory Study in Machine Learning\-enhanced Detection of Microplastic from Filters / 

发布日期：2025-10-20

作者：Paul\-Tiberiu Miclea

摘要：Microplastics \(MPs\) are ubiquitous pollutants with demonstrated potential to impact ecosystems and human health. Their microscopic size complicates detection, classification, and removal, especially in biological and environmental samples. While techniques like optical microscopy, Scanning Electron Microscopy \(SEM\), and Atomic Force Microscopy \(AFM\) provide a sound basis for detection, applying these approaches requires usually manual analysis and prevents efficient use in large screening studies. To this end, machine learning \(ML\) has emerged as a powerful tool in advancing microplastic detection. In this exploratory study, we investigate potential, limitations and future directions of advancing the detection and quantification of MP particles and fibres using a combination of SEM imaging and machine learning\-based object detection. For simplicity, we focus on a filtration scenario where image backgrounds exhibit a symmetric and repetitive pattern. Our findings indicate differences in the quality of YOLO models for the given task and the relevance of optimizing preprocessing. At the same time, we identify open challenges, such as limited amounts of expert\-labeled data necessary for reliable training of ML models.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2510.18089v1)

---

