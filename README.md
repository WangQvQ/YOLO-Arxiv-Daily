# 每日从arXiv中获取最新YOLO相关论文


## Assessing the Capability of YOLO\- and Transformer\-based Object Detectors for Real\-time Weed Detection / 

发布日期：2025-01-29

作者：Alicia Allmendinger

摘要：Spot spraying represents an efficient and sustainable method for reducing the amount of pesticides, particularly herbicides, used in agricultural fields. To achieve this, it is of utmost importance to reliably differentiate between crops and weeds, and even between individual weed species in situ and under real\-time conditions. To assess suitability for real\-time application, different object detection models that are currently state\-of\-the\-art are compared. All available models of YOLOv8, YOLOv9, YOLOv10, and RT\-DETR are trained and evaluated with images from a real field situation. The images are separated into two distinct datasets: In the initial data set, each species of plants is trained individually; in the subsequent dataset, a distinction is made between monocotyledonous weeds, dicotyledonous weeds, and three chosen crops. The results demonstrate that while all models perform equally well in the metrics evaluated, the YOLOv9 models, particularly the YOLOv9s and YOLOv9e, stand out in terms of their strong recall scores \(66.58 % and 72.36 %\), as well as mAP50 \(73.52 % and 79.86 %\), and mAP50\-95 \(43.82 % and 47.00 %\) in dataset 2. However, the RT\-DETR models, especially RT\-DETR\-l, excel in precision with reaching 82.44 % on dataset 1 and 81.46 % in dataset 2, making them particularly suitable for scenarios where minimizing false positives is critical. In particular, the smallest variants of the YOLO models \(YOLOv8n, YOLOv9t, and YOLOv10n\) achieve substantially faster inference times down to 7.58 ms for dataset 2 on the NVIDIA GeForce RTX 4090 GPU for analyzing one frame, while maintaining competitive accuracy, highlighting their potential for deployment in resource\-constrained embedded computing devices as typically used in productive setups.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2501.17387v1)

---


## Efficient Object Detection of Marine Debris using Pruned YOLO Model / 

发布日期：2025-01-27

作者：Abi Aryaza

摘要：Marine debris poses significant harm to marine life due to substances like microplastics, polychlorinated biphenyls, and pesticides, which damage habitats and poison organisms. Human\-based solutions, such as diving, are increasingly ineffective in addressing this issue. Autonomous underwater vehicles \(AUVs\) are being developed for efficient sea garbage collection, with the choice of object detection architecture being critical. This research employs the YOLOv4 model for real\-time detection of marine debris using the Trash\-ICRA 19 dataset, consisting of 7683 images at 480x320 pixels. Various modifications\-pretrained models, training from scratch, mosaic augmentation, layer freezing, YOLOv4\-tiny, and channel pruning\-are compared to enhance architecture efficiency. Channel pruning significantly improves detection speed, increasing the base YOLOv4 frame rate from 15.19 FPS to 19.4 FPS, with only a 1.2% drop in mean Average Precision, from 97.6% to 96.4%.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2501.16571v1)

---


## Explainable YOLO\-Based Dyslexia Detection in Synthetic Handwriting Data / 

发布日期：2025-01-25

作者：Nora Fink

摘要：Dyslexia affects reading and writing skills across many languages. This work describes a new application of YOLO\-based object detection to isolate and label handwriting patterns \(Normal, Reversal, Corrected\) within synthetic images that resemble real words. Individual letters are first collected, preprocessed into 32x32 samples, then assembled into larger synthetic 'words' to simulate realistic handwriting. Our YOLOv11 framework simultaneously localizes each letter and classifies it into one of three categories, reflecting key dyslexia traits. Empirically, we achieve near\-perfect performance, with precision, recall, and F1 metrics typically exceeding 0.999. This surpasses earlier single\-letter approaches that rely on conventional CNNs or transfer\-learning classifiers \(for example, MobileNet\-based methods in Robaa et al. arXiv:2410.19821\). Unlike simpler pipelines that consider each letter in isolation, our solution processes complete word images, resulting in more authentic representations of handwriting. Although relying on synthetic data raises concerns about domain gaps, these experiments highlight the promise of YOLO\-based detection for faster and more interpretable dyslexia screening. Future work will expand to real\-world handwriting, other languages, and deeper explainability methods to build confidence among educators, clinicians, and families.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2501.15263v1)

---


## Comprehensive Evaluation of Cloaking Backdoor Attacks on Object Detector in Real\-World / 

发布日期：2025-01-25

作者：Hua Ma

摘要：The exploration of backdoor vulnerabilities in object detectors, particularly in real\-world scenarios, remains limited. A significant challenge lies in the absence of a natural physical backdoor dataset, and constructing such a dataset is both time\- and labor\-intensive. In this work, we address this gap by creating a large\-scale dataset comprising approximately 11,800 images/frames with annotations featuring natural objects \(e.g., T\-shirts and hats\) as triggers to incur cloaking adversarial effects in diverse real\-world scenarios. This dataset is tailored for the study of physical backdoors in object detectors. Leveraging this dataset, we conduct a comprehensive evaluation of an insidious cloaking backdoor effect against object detectors, wherein the bounding box around a person vanishes when the individual is near a natural object \(e.g., a commonly available T\-shirt\) in front of the detector. Our evaluations encompass three prevalent attack surfaces: data outsourcing, model outsourcing, and the use of pretrained models. The cloaking effect is successfully implanted in object detectors across all three attack surfaces. We extensively evaluate four popular object detection algorithms \(anchor\-based Yolo\-V3, Yolo\-V4, Faster R\-CNN, and anchor\-free CenterNet\) using 19 videos \(totaling approximately 11,800 frames\) in real\-world scenarios. Our results demonstrate that the backdoor attack exhibits remarkable robustness against various factors, including movement, distance, angle, non\-rigid deformation, and lighting. In data and model outsourcing scenarios, the attack success rate \(ASR\) in most videos reaches 100% or near it, while the clean data accuracy of the backdoored model remains indistinguishable from that of the clean model, making it impossible to detect backdoor behavior through a validation set.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2501.15101v1)

---


## Effective Defect Detection Using Instance Segmentation for NDI / 

发布日期：2025-01-24

作者：Ashiqur Rahman

摘要：Ultrasonic testing is a common Non\-Destructive Inspection \(NDI\) method used in aerospace manufacturing. However, the complexity and size of the ultrasonic scans make it challenging to identify defects through visual inspection or machine learning models. Using computer vision techniques to identify defects from ultrasonic scans is an evolving research area. In this study, we used instance segmentation to identify the presence of defects in the ultrasonic scan images of composite panels that are representative of real components manufactured in aerospace. We used two models based on Mask\-RCNN \(Detectron 2\) and YOLO 11 respectively. Additionally, we implemented a simple statistical pre\-processing technique that reduces the burden of requiring custom\-tailored pre\-processing techniques. Our study demonstrates the feasibility and effectiveness of using instance segmentation in the NDI pipeline by significantly reducing data pre\-processing time, inspection time, and overall costs.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2501.14149v1)

---

