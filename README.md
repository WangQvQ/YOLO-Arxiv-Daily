# 每日从arXiv中获取最新YOLO相关论文


## Hardware optimization on Android for inference of AI models / 

发布日期：2025-11-17

作者：Iulius Gherasim

摘要：The pervasive integration of Artificial Intelligence models into contemporary mobile computing is notable across numerous use cases, from virtual assistants to advanced image processing. Optimizing the mobile user experience involves minimal latency and high responsiveness from deployed AI models with challenges from execution strategies that fully leverage real time constraints to the exploitation of heterogeneous hardware architecture. In this paper, we research and propose the optimal execution configurations for AI models on an Android system, focusing on two critical tasks: object detection \(YOLO family\) and image classification \(ResNet\). These configurations evaluate various model quantization schemes and the utilization of on device accelerators, specifically the GPU and NPU. Our core objective is to empirically determine the combination that achieves the best trade\-off between minimal accuracy degradation and maximal inference speed\-up.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2511.13453v1)

---


## YOLO Meets Mixture\-of\-Experts: Adaptive Expert Routing for Robust Object Detection / 

发布日期：2025-11-17

作者：Ori Meiraz

摘要：This paper presents a novel Mixture\-of\-Experts framework for object detection, incorporating adaptive routing among multiple YOLOv9\-T experts to enable dynamic feature specialization and achieve higher mean Average Precision \(mAP\) and Average Recall \(AR\) compared to a single YOLOv9\-T model.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2511.13344v1)

---


## MCAQ\-YOLO: Morphological Complexity\-Aware Quantization for Efficient Object Detection with Curriculum Learning / 

发布日期：2025-11-17

作者：Yoonjae Seo

摘要：Most neural network quantization methods apply uniform bit precision across spatial regions, ignoring the heterogeneous structural and textural complexity of visual data. This paper introduces MCAQ\-YOLO, a morphological complexity\-aware quantization framework for object detection. The framework employs five morphological metrics \- fractal dimension, texture entropy, gradient variance, edge density, and contour complexity \- to characterize local visual morphology and guide spatially adaptive bit allocation. By correlating these metrics with quantization sensitivity, MCAQ\-YOLO dynamically adjusts bit precision according to spatial complexity. In addition, a curriculum\-based quantization\-aware training scheme progressively increases quantization difficulty to stabilize optimization and accelerate convergence. Experimental results demonstrate a strong correlation between morphological complexity and quantization sensitivity and show that MCAQ\-YOLO achieves superior detection accuracy and convergence efficiency compared with uniform quantization. On a safety equipment dataset, MCAQ\-YOLO attains 85.6 percent mAP@0.5 with an average of 4.2 bits and a 7.6x compression ratio, yielding 3.5 percentage points higher mAP than uniform 4\-bit quantization while introducing only 1.8 ms of additional runtime overhead per image. Cross\-dataset validation on COCO and Pascal VOC further confirms consistent performance gains, indicating that morphology\-driven spatial quantization can enhance efficiency and robustness for computationally constrained, safety\-critical visual recognition tasks.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2511.12976v1)

---


## Facial Expression Recognition with YOLOv11 and YOLOv12: A Comparative Study / 

发布日期：2025-11-14

作者：Umma Aymon

摘要：Facial Expression Recognition remains a challenging task, especially in unconstrained, real\-world environments. This study investigates the performance of two lightweight models, YOLOv11n and YOLOv12n, which are the nano variants of the latest official YOLO series, within a unified detection and classification framework for FER. Two benchmark classification datasets, FER2013 and KDEF, are converted into object detection format and model performance is evaluated using mAP 0.5, precision, recall, and confusion matrices. Results show that YOLOv12n achieves the highest overall performance on the clean KDEF dataset with a mAP 0.5 of 95.6, and also outperforms YOLOv11n on the FER2013 dataset in terms of mAP 63.8, reflecting stronger sensitivity to varied expressions. In contrast, YOLOv11n demonstrates higher precision 65.2 on FER2013, indicating fewer false positives and better reliability in noisy, real\-world conditions. On FER2013, both models show more confusion between visually similar expressions, while clearer class separation is observed on the cleaner KDEF dataset. These findings underscore the trade\-off between sensitivity and precision, illustrating how lightweight YOLO models can effectively balance performance and efficiency. The results demonstrate adaptability across both controlled and real\-world conditions, establishing these models as strong candidates for real\-time, resource\-constrained emotion\-aware AI applications.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2511.10940v1)

---


## YOLO\-Drone: An Efficient Object Detection Approach Using the GhostHead Network for Drone Images / 

发布日期：2025-11-14

作者：Hyun\-Ki Jung

摘要：Object detection using images or videos captured by drones is a promising technology with significant potential across various industries. However, a major challenge is that drone images are typically taken from high altitudes, making object identification difficult. This paper proposes an effective solution to address this issue. The base model used in the experiments is YOLOv11, the latest object detection model, with a specific implementation based on YOLOv11n. The experimental data were sourced from the widely used and reliable VisDrone dataset, a standard benchmark in drone\-based object detection. This paper introduces an enhancement to the Head network of the YOLOv11 algorithm, called the GhostHead Network. The model incorporating this improvement is named YOLO\-Drone. Experimental results demonstrate that YOLO\-Drone achieves significant improvements in key detection accuracy metrics, including Precision, Recall, F1\-Score, and mAP \(0.5\), compared to the original YOLOv11. Specifically, the proposed model recorded a 0.4% increase in Precision, a 0.6% increase in Recall, a 0.5% increase in F1\-Score, and a 0.5% increase in mAP \(0.5\). Additionally, the Inference Speed metric, which measures image processing speed, also showed a notable improvement. These results indicate that YOLO\-Drone is a high\-performance model with enhanced accuracy and speed compared to YOLOv11. To further validate its reliability, comparative experiments were conducted against other high\-performance object detection models, including YOLOv8, YOLOv9, and YOLOv10. The results confirmed that the proposed model outperformed YOLOv8 by 0.1% in mAP \(0.5\) and surpassed YOLOv9 and YOLOv10 by 0.3% and 0.6%, respectively.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2511.10905v1)

---

