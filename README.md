# 每日从arXiv中获取最新YOLO相关论文


## Quantization Robustness to Input Degradations for Object Detection / 

发布日期：2025-08-27

作者：Toghrul Karimov

摘要：Post\-training quantization \(PTQ\) is crucial for deploying efficient object detection models, like YOLO, on resource\-constrained devices. However, the impact of reduced precision on model robustness to real\-world input degradations such as noise, blur, and compression artifacts is a significant concern. This paper presents a comprehensive empirical study evaluating the robustness of YOLO models \(nano to extra\-large scales\) across multiple precision formats: FP32, FP16 \(TensorRT\), Dynamic UINT8 \(ONNX\), and Static INT8 \(TensorRT\). We introduce and evaluate a degradation\-aware calibration strategy for Static INT8 PTQ, where the TensorRT calibration process is exposed to a mix of clean and synthetically degraded images. Models were benchmarked on the COCO dataset under seven distinct degradation conditions \(including various types and levels of noise, blur, low contrast, and JPEG compression\) and a mixed\-degradation scenario. Results indicate that while Static INT8 TensorRT engines offer substantial speedups \(~1.5\-3.3x\) with a moderate accuracy drop \(~3\-7% mAP50\-95\) on clean data, the proposed degradation\-aware calibration did not yield consistent, broad improvements in robustness over standard clean\-data calibration across most models and degradations. A notable exception was observed for larger model scales under specific noise conditions, suggesting model capacity may influence the efficacy of this calibration approach. These findings highlight the challenges in enhancing PTQ robustness and provide insights for deploying quantized detectors in uncontrolled environments. All code and evaluation tables are available at https://github.com/AllanK24/QRID.

中文摘要：


代码链接：https://github.com/AllanK24/QRID.

论文链接：[阅读更多](http://arxiv.org/abs/2508.19600v1)

---


## Spatial\-temporal risk field\-based coupled dynamic\-static driving risk assessment and trajectory planning in weaving segments / 

发布日期：2025-08-27

作者：Guodong Ma

摘要：In this paper, we first propose a spatial\-temporal coupled risk assessment paradigm by constructing a three\-dimensional spatial\-temporal risk field \(STRF\). Specifically, we introduce spatial\-temporal distances to quantify the impact of future trajectories of dynamic obstacles. We also incorporate a geometrically configured specialized field for the weaving segment to constrain vehicle movement directionally. To enhance the STRF's accuracy, we further developed a parameter calibration method using real\-world aerial video data, leveraging YOLO\-based machine vision and dynamic risk balance theory. A comparative analysis with the traditional risk field demonstrates the STRF's superior situational awareness of anticipatory risk. Building on these results, we final design a STRF\-based CAV trajectory planning method in weaving segments. We integrate spatial\-temporal risk occupancy maps, dynamic iterative sampling, and quadratic programming to enhance safety, comfort, and efficiency. By incorporating both dynamic and static risk factors during the sampling phase, our method ensures robust safety performance. Additionally, the proposed method simultaneously optimizes path and speed using a parallel computing approach, reducing computation time. Real\-world cases show that, compared to the dynamic planning \+ quadratic programming schemes, and real human driving trajectories, our method significantly improves safety, reduces lane\-change completion time, and minimizes speed fluctuations.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2508.19513v1)

---


## Weed Detection in Challenging Field Conditions: A Semi\-Supervised Framework for Overcoming Shadow Bias and Data Scarcity / 

发布日期：2025-08-27

作者：Alzayat Saleh

摘要：The automated management of invasive weeds is critical for sustainable agriculture, yet the performance of deep learning models in real\-world fields is often compromised by two factors: challenging environmental conditions and the high cost of data annotation. This study tackles both issues through a diagnostic\-driven, semi\-supervised framework. Using a unique dataset of approximately 975 labeled and 10,000 unlabeled images of Guinea Grass in sugarcane, we first establish strong supervised baselines for classification \(ResNet\) and detection \(YOLO, RF\-DETR\), achieving F1 scores up to 0.90 and mAP50 scores exceeding 0.82. Crucially, this foundational analysis, aided by interpretability tools, uncovered a pervasive "shadow bias," where models learned to misidentify shadows as vegetation. This diagnostic insight motivated our primary contribution: a semi\-supervised pipeline that leverages unlabeled data to enhance model robustness. By training models on a more diverse set of visual information through pseudo\-labeling, this framework not only helps mitigate the shadow bias but also provides a tangible boost in recall, a critical metric for minimizing weed escapes in automated spraying systems. To validate our methodology, we demonstrate its effectiveness in a low\-data regime on a public crop\-weed benchmark. Our work provides a clear and field\-tested framework for developing, diagnosing, and improving robust computer vision systems for the complex realities of precision agriculture.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2508.19511v1)

---


## HOTSPOT\-YOLO: A Lightweight Deep Learning Attention\-Driven Model for Detecting Thermal Anomalies in Drone\-Based Solar Photovoltaic Inspections / 

发布日期：2025-08-26

作者：Mahmoud Dhimish

摘要：Thermal anomaly detection in solar photovoltaic \(PV\) systems is essential for ensuring operational efficiency and reducing maintenance costs. In this study, we developed and named HOTSPOT\-YOLO, a lightweight artificial intelligence \(AI\) model that integrates an efficient convolutional neural network backbone and attention mechanisms to improve object detection. This model is specifically designed for drone\-based thermal inspections of PV systems, addressing the unique challenges of detecting small and subtle thermal anomalies, such as hotspots and defective modules, while maintaining real\-time performance. Experimental results demonstrate a mean average precision of 90.8%, reflecting a significant improvement over baseline object detection models. With a reduced computational load and robustness under diverse environmental conditions, HOTSPOT\-YOLO offers a scalable and reliable solution for large\-scale PV inspections. This work highlights the integration of advanced AI techniques with practical engineering applications, revolutionizing automated fault detection in renewable energy systems.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2508.18912v1)

---


## LPLC: A Dataset for License Plate Legibility Classification / 

发布日期：2025-08-25

作者：Lucas Wojcik

摘要：Automatic License Plate Recognition \(ALPR\) faces a major challenge when dealing with illegible license plates \(LPs\). While reconstruction methods such as super\-resolution \(SR\) have emerged, the core issue of recognizing these low\-quality LPs remains unresolved. To optimize model performance and computational efficiency, image pre\-processing should be applied selectively to cases that require enhanced legibility. To support research in this area, we introduce a novel dataset comprising 10,210 images of vehicles with 12,687 annotated LPs for legibility classification \(the LPLC dataset\). The images span a wide range of vehicle types, lighting conditions, and camera/image quality levels. We adopt a fine\-grained annotation strategy that includes vehicle\- and LP\-level occlusions, four legibility categories \(perfect, good, poor, and illegible\), and character labels for three categories \(excluding illegible LPs\). As a benchmark, we propose a classification task using three image recognition networks to determine whether an LP image is good enough, requires super\-resolution, or is completely unrecoverable. The overall F1 score, which remained below 80% for all three baseline models \(ViT, ResNet, and YOLO\), together with the analyses of SR and LP recognition methods, highlights the difficulty of the task and reinforces the need for further research. The proposed dataset is publicly available at https://github.com/lmlwojcik/lplc\-dataset.

中文摘要：


代码链接：https://github.com/lmlwojcik/lplc-dataset.

论文链接：[阅读更多](http://arxiv.org/abs/2508.18425v1)

---

