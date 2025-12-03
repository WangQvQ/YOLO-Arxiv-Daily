# 每日从arXiv中获取最新YOLO相关论文


## Edge\-Native, Behavior\-Adaptive Drone System for Wildlife Monitoring / 

发布日期：2025-12-01

作者：Jenna Kline

摘要：Wildlife monitoring with drones must balance competing demands: approaching close enough to capture behaviorally\-relevant video while avoiding stress responses that compromise animal welfare and data validity. Human operators face a fundamental attentional bottleneck: they cannot simultaneously control drone operations and monitor vigilance states across entire animal groups. By the time elevated vigilance becomes obvious, an adverse flee response by the animals may be unavoidable. To solve this challenge, we present an edge\-native, behavior\-adaptive drone system for wildlife monitoring. This configurable decision\-support system augments operator expertise with automated group\-level vigilance monitoring. Our system continuously tracks individual behaviors using YOLOv11m detection and YOLO\-Behavior classification, aggregates vigilance states into a real\-time group stress metric, and provides graduated alerts \(alert vigilance to flee response\) with operator\-tunable thresholds for context\-specific calibration. We derive service\-level objectives \(SLOs\) from video frame rates and behavioral dynamics: to monitor 30fps video streams in real\-time, our system must complete detection and classification within 33ms per frame. Our edge\-native pipeline achieves 23.8ms total inference on GPU\-accelerated hardware, meeting this constraint with a substantial margin. Retrospective analysis of seven wildlife monitoring missions demonstrates detection capability and quantifies the cost of reactive control: manual piloting results in 14 seconds average adverse behavior duration with 71.9% usable frames. Our analysis reveals operators could have received actionable alerts 51s before animals fled in 57% of missions. Simulating 5\-second operator intervention yields a projected performance of 82.8% usable frames with 1\-second adverse behavior duration,a 93% reduction compared to manual piloting.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2512.02285v1)

---


## Real\-Time On\-the\-Go Annotation Framework Using YOLO for Automated Dataset Generation / 

发布日期：2025-12-01

作者：Mohamed Abdallah Salem

摘要：Efficient and accurate annotation of datasets remains a significant challenge for deploying object detection models such as You Only Look Once \(YOLO\) in real\-world applications, particularly in agriculture where rapid decision\-making is critical. Traditional annotation techniques are labor\-intensive, requiring extensive manual labeling post data collection. This paper presents a novel real\-time annotation approach leveraging YOLO models deployed on edge devices, enabling immediate labeling during image capture. To comprehensively evaluate the efficiency and accuracy of our proposed system, we conducted an extensive comparative analysis using three prominent YOLO architectures \(YOLOv5, YOLOv8, YOLOv12\) under various configurations: single\-class versus multi\-class annotation and pretrained versus scratch\-based training. Our analysis includes detailed statistical tests and learning dynamics, demonstrating significant advantages of pretrained and single\-class configurations in terms of model convergence, performance, and robustness. Results strongly validate the feasibility and effectiveness of our real\-time annotation framework, highlighting its capability to drastically reduce dataset preparation time while maintaining high annotation quality.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2512.01165v1)

---


## Identifying bars in galaxies using machine learning / 

发布日期：2025-11-28

作者：Rajit Shrivastava

摘要：This thesis presents an innovative framework for the automated detection and characterization of galactic bars, pivotal structures in spiral galaxies, using the YOLO\-OBB \(You Only Look Once with Oriented Bounding Boxes\) model. Traditional methods for identifying bars are often labor\-intensive and subjective, limiting their scalability for large astronomical surveys. To address this, a synthetic dataset of 1,000 barred spiral galaxy images was generated, incorporating realistic components such as disks, bars, bulges, spiral arms, stars, and observational noise, modeled through Gaussian, Ferrers, and Sersic functions. The YOLO\-OBB model, trained on this dataset for six epochs, achieved robust validation metrics, including a precision of 0.93745, recall of 0.85, and mean Average Precision \(mAP50\) of 0.94173. Applied to 10 real galaxy images, the model extracted physical parameters, such as bar lengths ranging from 2.27 to 9.70 kpc and orientations from 13.41$^circ$ to 134.11$^circ$, with detection confidences between 0.26 and 0.68. These measurements, validated through pixel\-to\-kiloparsec conversions, align with established bar sizes, demonstrating the model's reliability. The methodology's scalability and interpretability enable efficient analysis of complex galaxy morphologies, particularly for dwarf galaxies and varied orientations. Future research aims to expand the dataset to 5,000 galaxies and integrate the Tremaine\-Weinberg method to measure bar pattern speeds, enhancing insights into galaxy dynamics and evolution. This work advances automated morphological analysis, offering a transformative tool for large\-scale astronomical studies.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2511.23383v1)

---


## Hierarchical Feature Integration for Multi\-Signal Automatic Modulation Recognition / 

发布日期：2025-11-28

作者：Yunpeng Qu

摘要：Automatic modulation recognition \(AMR\) is a crucial step in wireless communication systems, which identifies the modulation scheme from detected signals to provide key information for further processing. However, previous work has mainly focused on the identification of a single signal, overlooking the phenomenon of multiple signal superposition in practical channels and the signal detection procedures that must be conducted beforehand. Considering the susceptibility of radio frequency \(RF\) signals to noise interference and significant spectral variations, we propose a novel Hierarchical Feature Integration \(HIFI\)\-YOLO framework for multi\-signal joint detection and modulation recognition. Our HIFI\-YOLO framework, with its unique design of hierarchical feature integration, effectively enhances the representation capability of features in different modules, thereby improving detection performance. We construct a large\-scale AMR dataset specifically tailored for scenarios of the coexistence or overlapping of multiple signals transmitted through channels with realistic propagation conditions, consisting of diverse digital and analog modulation schemes. Extensive experiments on our dataset demonstrate the excellent performance of HIFI\-YOLO in multi\-signal detection and modulation recognition as a joint approach.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2511.23258v1)

---


## Analysis of Incursive Breast Cancer in Mammograms Using YOLO, Explainability, and Domain Adaptation / 

发布日期：2025-11-28

作者：Jayan Adhikari

摘要：Deep learning models for breast cancer detection from mammographic images have significant reliability problems when presented with Out\-of\-Distribution \(OOD\) inputs such as other imaging modalities \(CT, MRI, X\-ray\) or equipment variations, leading to unreliable detection and misdiagnosis. The current research mitigates the fundamental OOD issue through a comprehensive approach integrating ResNet50\-based OOD filtering with YOLO architectures \(YOLOv8, YOLOv11, YOLOv12\) for accurate detection of breast cancer. Our strategy establishes an in\-domain gallery via cosine similarity to rigidly reject non\-mammographic inputs prior to processing, ensuring that only domain\-associated images supply the detection pipeline. The OOD detection component achieves 99.77% general accuracy with immaculate 100% accuracy on OOD test sets, effectively eliminating irrelevant imaging modalities. ResNet50 was selected as the optimum backbone after 12 CNN architecture searches. The joint framework unites OOD robustness with high detection performance \(mAP@0.5: 0.947\) and enhanced interpretability through Grad\-CAM visualizations. Experimental validation establishes that OOD filtering significantly improves system reliability by preventing false alarms on out\-of\-distribution inputs while maintaining higher detection accuracy on mammographic data. The present study offers a fundamental foundation for the deployment of reliable AI\-based breast cancer detection systems in diverse clinical environments with inherent data heterogeneity.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2512.00129v1)

---

