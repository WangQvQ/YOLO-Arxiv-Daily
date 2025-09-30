# 每日从arXiv中获取最新YOLO相关论文


## Enhanced Fracture Diagnosis Based on Critical Regional and Scale Aware in YOLO / 

发布日期：2025-09-27

作者：Yuyang Sun

摘要：Fracture detection plays a critical role in medical imaging analysis, traditional fracture diagnosis relies on visual assessment by experienced physicians, however the speed and accuracy of this approach are constrained by the expertise. With the rapid advancements in artificial intelligence, deep learning models based on the YOLO framework have been widely employed for fracture detection, demonstrating significant potential in improving diagnostic efficiency and accuracy. This study proposes an improved YOLO\-based model, termed Fracture\-YOLO, which integrates novel Critical\-Region\-Selector Attention \(CRSelector\) and Scale\-Aware \(ScA\) heads to further enhance detection performance. Specifically, the CRSelector module utilizes global texture information to focus on critical features of fracture regions. Meanwhile, the ScA module dynamically adjusts the weights of features at different scales, enhancing the model's capacity to identify fracture targets at multiple scales. Experimental results demonstrate that, compared to the baseline model, Fracture\-YOLO achieves a significant improvement in detection precision, with mAP50 and mAP50\-95 increasing by 4 and 3, surpassing the baseline model and achieving state\-of\-the\-art \(SOTA\) performance.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2509.23408v1)

---


## TRAX: TRacking Axles for Accurate Axle Count Estimation / 

发布日期：2025-09-27

作者：Avinash Rai

摘要：Accurate counting of vehicle axles is essential for traffic control, toll collection, and infrastructure development. We present an end\-to\-end, video\-based pipeline for axle counting that tackles limitations of previous works in dense environments. Our system leverages a combination of YOLO\-OBB to detect and categorize vehicles, and YOLO to detect tires. Detected tires are intelligently associated to their respective parent vehicles, enabling accurate axle prediction even in complex scenarios. However, there are a few challenges in detection when it comes to scenarios with longer and occluded vehicles. We mitigate vehicular occlusions and partial detections for longer vehicles by proposing a novel TRAX \(Tire and Axle Tracking\) Algorithm to successfully track axle\-related features between frames. Our method stands out by significantly reducing false positives and improving the accuracy of axle\-counting for long vehicles, demonstrating strong robustness in real\-world traffic videos. This work represents a significant step toward scalable, AI\-driven axle counting systems, paving the way for machine vision to replace legacy roadside infrastructure.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2509.23171v1)

---


## TY\-RIST: Tactical YOLO Tricks for Real\-time Infrared Small Target Detection / 

发布日期：2025-09-26

作者：Abdulkarim Atrash

摘要：Infrared small target detection \(IRSTD\) is critical for defense and surveillance but remains challenging due to \(1\) target loss from minimal features, \(2\) false alarms in cluttered environments, \(3\) missed detections from low saliency, and \(4\) high computational costs. To address these issues, we propose TY\-RIST, an optimized YOLOv12n architecture that integrates \(1\) a stride\-aware backbone with fine\-grained receptive fields, \(2\) a high\-resolution detection head, \(3\) cascaded coordinate attention blocks, and \(4\) a branch pruning strategy that reduces computational cost by about 25.5% while marginally improving accuracy and enabling real\-time inference. We also incorporate the Normalized Gaussian Wasserstein Distance \(NWD\) to enhance regression stability. Extensive experiments on four benchmarks and across 20 different models demonstrate state\-of\-the\-art performance, improving mAP at 0.5 IoU by \+7.9%, Precision by \+3%, and Recall by \+10.2%, while achieving up to 123 FPS on a single GPU. Cross\-dataset validation on a fifth dataset further confirms strong generalization capability. Additional results and resources are available at https://www.github.com/moured/TY\-RIST

中文摘要：


代码链接：https://www.github.com/moured/TY-RIST

论文链接：[阅读更多](http://arxiv.org/abs/2509.22909v1)

---


## HierLight\-YOLO: A Hierarchical and Lightweight Object Detection Network for UAV Photography / 

发布日期：2025-09-26

作者：Defan Chen

摘要：The real\-time detection of small objects in complex scenes, such as the unmanned aerial vehicle \(UAV\) photography captured by drones, has dual challenges of detecting small targets \(<32 pixels\) and maintaining real\-time efficiency on resource\-constrained platforms. While YOLO\-series detectors have achieved remarkable success in real\-time large object detection, they suffer from significantly higher false negative rates for drone\-based detection where small objects dominate, compared to large object scenarios. This paper proposes HierLight\-YOLO, a hierarchical feature fusion and lightweight model that enhances the real\-time detection of small objects, based on the YOLOv8 architecture. We propose the Hierarchical Extended Path Aggregation Network \(HEPAN\), a multi\-scale feature fusion method through hierarchical cross\-level connections, enhancing the small object detection accuracy. HierLight\-YOLO includes two innovative lightweight modules: Inverted Residual Depthwise Convolution Block \(IRDCB\) and Lightweight Downsample \(LDown\) module, which significantly reduce the model's parameters and computational complexity without sacrificing detection capabilities. Small object detection head is designed to further enhance spatial resolution and feature fusion to tackle the tiny object \(4 pixels\) detection. Comparison experiments and ablation studies on the VisDrone2019 benchmark demonstrate state\-of\-the\-art performance of HierLight\-YOLO.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2509.22365v1)

---


## MS\-YOLO: Infrared Object Detection for Edge Deployment via MobileNetV4 and SlideLoss / 

发布日期：2025-09-25

作者：Jiali Zhang

摘要：Infrared imaging has emerged as a robust solution for urban object detection under low\-light and adverse weather conditions, offering significant advantages over traditional visible\-light cameras. However, challenges such as class imbalance, thermal noise, and computational constraints can significantly hinder model performance in practical settings. To address these issues, we evaluate multiple YOLO variants on the FLIR ADAS V2 dataset, ultimately selecting YOLOv8 as our baseline due to its balanced accuracy and efficiency. Building on this foundation, we present texttt\{MS\-YOLO\} \(textbf\{M\}obileNetv4 and textbf\{S\}lideLoss based on YOLO\), which replaces YOLOv8's CSPDarknet backbone with the more efficient MobileNetV4, reducing computational overhead by textbf\{1.5%\} while sustaining high accuracy. In addition, we introduce emph\{SlideLoss\}, a novel loss function that dynamically emphasizes under\-represented and occluded samples, boosting precision without sacrificing recall. Experiments on the FLIR ADAS V2 benchmark show that texttt\{MS\-YOLO\} attains competitive mAP and superior precision while operating at only textbf\{6.7 GFLOPs\}. These results demonstrate that texttt\{MS\-YOLO\} effectively addresses the dual challenge of maintaining high detection quality while minimizing computational costs, making it well\-suited for real\-time edge deployment in urban environments.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2509.21696v1)

---

