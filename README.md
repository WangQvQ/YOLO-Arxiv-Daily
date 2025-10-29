# 每日从arXiv中获取最新YOLO相关论文


## Delving into Cascaded Instability: A Lipschitz Continuity View on Image Restoration and Object Detection Synergy / 

发布日期：2025-10-28

作者：Qing Zhao

摘要：To improve detection robustness in adverse conditions \(e.g., haze and low light\), image restoration is commonly applied as a pre\-processing step to enhance image quality for the detector. However, the functional mismatch between restoration and detection networks can introduce instability and hinder effective integration \-\- an issue that remains underexplored. We revisit this limitation through the lens of Lipschitz continuity, analyzing the functional differences between restoration and detection networks in both the input space and the parameter space. Our analysis shows that restoration networks perform smooth, continuous transformations, while object detectors operate with discontinuous decision boundaries, making them highly sensitive to minor perturbations. This mismatch introduces instability in traditional cascade frameworks, where even imperceptible noise from restoration is amplified during detection, disrupting gradient flow and hindering optimization. To address this, we propose Lipschitz\-regularized object detection \(LROD\), a simple yet effective framework that integrates image restoration directly into the detector's feature learning, harmonizing the Lipschitz continuity of both tasks during training. We implement this framework as Lipschitz\-regularized YOLO \(LR\-YOLO\), extending seamlessly to existing YOLO detectors. Extensive experiments on haze and low\-light benchmarks demonstrate that LR\-YOLO consistently improves detection stability, optimization smoothness, and overall accuracy.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2510.24232v1)

---


## CSST Slitless Spectra: Target Detection and Classification with YOLO / 

发布日期：2025-10-28

作者：Yingying Zhou

摘要：Addressing the spatial uncertainty and spectral blending challenges in CSST slitless spectroscopy, we present a deep learning\-driven, end\-to\-end framework based on the You Only Look Once \(YOLO\) models. This approach directly detects, classifies, and analyzes spectral traces from raw 2D images, bypassing traditional, error\-accumulating pipelines. YOLOv5 effectively detects both compact zero\-order and extended first\-order traces even in highly crowded fields. Building on this, YOLO11 integrates source classification \(star/galaxy\) and discrete astrophysical parameter estimation \(e.g., redshift bins\), showcasing complete spectral trace analysis without other manual preprocessing. Our framework processes large images rapidly, learning spectral\-spatial features holistically to minimize errors. We achieve high trace detection precision \(YOLOv5\) and demonstrate successful quasar identification and binned redshift estimation \(YOLO11\). This study establishes machine learning as a paradigm shift in slitless spectroscopy, unifying detection, classification, and preliminary parameter estimation in a scalable system. Future research will concentrate on direct, continuous prediction of astrophysical parameters from raw spectral traces.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2510.24087v1)

---


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

