# 每日从arXiv中获取最新YOLO相关论文


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


## Benchmarking of Different YOLO Models for CAPTCHAs Detection and Classification / 

发布日期：2025-02-19

作者：Mikołaj Wysocki

摘要：This paper provides an analysis and comparison of the YOLOv5, YOLOv8 and YOLOv10 models for webpage CAPTCHAs detection using the datasets collected from the web and darknet as well as synthetized data of webpages. The study examines the nano \(n\), small \(s\), and medium \(m\) variants of YOLO architectures and use metrics such as Precision, Recall, F1 score, mAP@50 and inference speed to determine the real\-life utility. Additionally, the possibility of tuning the trained model to detect new CAPTCHA patterns efficiently was examined as it is a crucial part of real\-life applications. The image slicing method was proposed as a way to improve the metrics of detection on oversized input images which can be a common scenario in webpages analysis. Models in version nano achieved the best results in terms of speed, while more complexed architectures scored better in terms of other metrics.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2502.13740v1)

---


## YOLOv12: Attention\-Centric Real\-Time Object Detectors / 

发布日期：2025-02-18

作者：Yunjie Tian

摘要：Enhancing the network architecture of the YOLO framework has been crucial for a long time, but has focused on CNN\-based improvements despite the proven superiority of attention mechanisms in modeling capabilities. This is because attention\-based models cannot match the speed of CNN\-based models. This paper proposes an attention\-centric YOLO framework, namely YOLOv12, that matches the speed of previous CNN\-based ones while harnessing the performance benefits of attention mechanisms. YOLOv12 surpasses all popular real\-time object detectors in accuracy with competitive speed. For example, YOLOv12\-N achieves 40.6% mAP with an inference latency of 1.64 ms on a T4 GPU, outperforming advanced YOLOv10\-N / YOLOv11\-N by 2.1%/1.2% mAP with a comparable speed. This advantage extends to other model scales. YOLOv12 also surpasses end\-to\-end real\-time detectors that improve DETR, such as RT\-DETR / RT\-DETRv2: YOLOv12\-S beats RT\-DETR\-R18 / RT\-DETRv2\-R18 while running 42% faster, using only 36% of the computation and 45% of the parameters. More comparisons are shown in Figure 1.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2502.12524v1)

---


## Acute Lymphoblastic Leukemia Diagnosis Employing YOLOv11, YOLOv8, ResNet50, and Inception\-ResNet\-v2 Deep Learning Models / 

发布日期：2025-02-13

作者：Alaa Awad

摘要：Thousands of individuals succumb annually to leukemia alone. As artificial intelligence\-driven technologies continue to evolve and advance, the question of their applicability and reliability remains unresolved. This study aims to utilize image processing and deep learning methodologies to achieve state\-of\-the\-art results for the detection of Acute Lymphoblastic Leukemia \(ALL\) using data that best represents real\-world scenarios. ALL is one of several types of blood cancer, and it is an aggressive form of leukemia. In this investigation, we examine the most recent advancements in ALL detection, as well as the latest iteration of the YOLO series and its performance. We address the question of whether white blood cells are malignant or benign. Additionally, the proposed models can identify different ALL stages, including early stages. Furthermore, these models can detect hematogones despite their frequent misclassification as ALL. By utilizing advanced deep learning models, namely, YOLOv8, YOLOv11, ResNet50 and Inception\-ResNet\-v2, the study achieves accuracy rates as high as 99.7%, demonstrating the effectiveness of these algorithms across multiple datasets and various real\-world situations.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2502.09804v1)

---

