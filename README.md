# 每日从arXiv中获取最新YOLO相关论文


## DEAL\-YOLO: Drone\-based Efficient Animal Localization using YOLO / 

发布日期：2025-03-06

作者：Aditya Prashant Naidu

摘要：Although advances in deep learning and aerial surveillance technology are improving wildlife conservation efforts, complex and erratic environmental conditions still pose a problem, requiring innovative solutions for cost\-effective small animal detection. This work introduces DEAL\-YOLO, a novel approach that improves small object detection in Unmanned Aerial Vehicle \(UAV\) images by using multi\-objective loss functions like Wise IoU \(WIoU\) and Normalized Wasserstein Distance \(NWD\), which prioritize pixels near the centre of the bounding box, ensuring smoother localization and reducing abrupt deviations. Additionally, the model is optimized through efficient feature extraction with Linear Deformable \(LD\) convolutions, enhancing accuracy while maintaining computational efficiency. The Scaled Sequence Feature Fusion \(SSFF\) module enhances object detection by effectively capturing inter\-scale relationships, improving feature representation, and boosting metrics through optimized multiscale fusion. Comparison with baseline models reveals high efficacy with up to 69.5% fewer parameters compared to vanilla Yolov8\-N, highlighting the robustness of the proposed modifications. Through this approach, our paper aims to facilitate the detection of endangered species, animal population analysis, habitat monitoring, biodiversity research, and various other applications that enrich wildlife conservation efforts. DEAL\-YOLO employs a two\-stage inference paradigm for object detection, refining selected regions to improve localization and confidence. This approach enhances performance, especially for small instances with low objectness scores.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2503.04698v1)

---


## Teach YOLO to Remember: A Self\-Distillation Approach for Continual Object Detection / 

发布日期：2025-03-06

作者：Riccardo De Monte

摘要：Real\-time object detectors like YOLO achieve exceptional performance when trained on large datasets for multiple epochs. However, in real\-world scenarios where data arrives incrementally, neural networks suffer from catastrophic forgetting, leading to a loss of previously learned knowledge. To address this, prior research has explored strategies for Class Incremental Learning \(CIL\) in Continual Learning for Object Detection \(CLOD\), with most approaches focusing on two\-stage object detectors. However, existing work suggests that Learning without Forgetting \(LwF\) may be ineffective for one\-stage anchor\-free detectors like YOLO due to noisy regression outputs, which risk transferring corrupted knowledge. In this work, we introduce YOLO LwF, a self\-distillation approach tailored for YOLO\-based continual object detection. We demonstrate that when coupled with a replay memory, YOLO LwF significantly mitigates forgetting. Compared to previous approaches, it achieves state\-of\-the\-art performance, improving mAP by \+2.1% and \+2.9% on the VOC and COCO benchmarks, respectively.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2503.04688v1)

---


## A lightweight model FDM\-YOLO for small target improvement based on YOLOv8 / 

发布日期：2025-03-06

作者：Xuerui Zhang

摘要：Small targets are particularly difficult to detect due to their low pixel count, complex backgrounds, and varying shooting angles, which make it hard for models to extract effective features. While some large\-scale models offer high accuracy, their long inference times make them unsuitable for real\-time deployment on edge devices. On the other hand, models designed for low computational power often suffer from poor detection accuracy. This paper focuses on small target detection and explores methods for object detection under low computational constraints. Building on the YOLOv8 model, we propose a new network architecture called FDM\-YOLO. Our research includes the following key contributions: We introduce FDM\-YOLO by analyzing the output of the YOLOv8 detection head. We add a highresolution layer and remove the large target detection layer to better handle small targets. Based on PConv, we propose a lightweight network structure called Fast\-C2f, which is integrated into the PAN module of the model. To mitigate the accuracy loss caused by model lightweighting, we employ dynamic upsampling \(Dysample\) and a lightweight EMA attention mechanism.The FDM\-YOLO model was validated on the Visdrone dataset, achieving a 38% reduction in parameter count and improving the Map0.5 score from 38.4% to 42.5%, all while maintaining nearly the same inference speed. This demonstrates the effectiveness of our approach in balancing accuracy and efficiency for edge device deployment.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2503.04452v1)

---


## Robust Computer\-Vision based Construction Site Detection for Assistive\-Technology Applications / 

发布日期：2025-03-06

作者：Junchi Feng

摘要：Navigating urban environments poses significant challenges for people with disabilities, particularly those with blindness and low vision. Environments with dynamic and unpredictable elements like construction sites are especially challenging. Construction sites introduce hazards like uneven surfaces, obstructive barriers, hazardous materials, and excessive noise, and they can alter routing, complicating safe mobility. Existing assistive technologies are limited, as navigation apps do not account for construction sites during trip planning, and detection tools that attempt hazard recognition struggle to address the extreme variability of construction paraphernalia. This study introduces a novel computer vision\-based system that integrates open\-vocabulary object detection, a YOLO\-based scaffolding\-pole detection model, and an optical character recognition \(OCR\) module to comprehensively identify and interpret construction site elements for assistive navigation. In static testing across seven construction sites, the system achieved an overall accuracy of 88.56%, reliably detecting objects from 2m to 10m within a 0$^circ$ \-\- 75$^circ$ angular offset. At closer distances \(2\-\-4m\), the detection rate was 100% at all tested angles. At

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2503.04139v1)

---


## YOLO\-PRO: Enhancing Instance\-Specific Object Detection with Full\-Channel Global Self\-Attention / 

发布日期：2025-03-04

作者：Lin Huang

摘要：This paper addresses the inherent limitations of conventional bottleneck structures \(diminished instance discriminability due to overemphasis on batch statistics\) and decoupled heads \(computational redundancy\) in object detection frameworks by proposing two novel modules: the Instance\-Specific Bottleneck with full\-channel global self\-attention \(ISB\) and the Instance\-Specific Asymmetric Decoupled Head \(ISADH\). The ISB module innovatively reconstructs feature maps to establish an efficient full\-channel global attention mechanism through synergistic fusion of batch\-statistical and instance\-specific features. Complementing this, the ISADH module pioneers an asymmetric decoupled architecture enabling hierarchical multi\-dimensional feature integration via dual\-stream batch\-instance representation fusion. Extensive experiments on the MS\-COCO benchmark demonstrate that the coordinated deployment of ISB and ISADH in the YOLO\-PRO framework achieves state\-of\-the\-art performance across all computational scales. Specifically, YOLO\-PRO surpasses YOLOv8 by 1.0\-1.6% AP \(N/S/M/L/X scales\) and outperforms YOLO11 by 0.1\-0.5% AP in critical M/L/X groups, while maintaining competitive computational efficiency. This work provides practical insights for developing high\-precision detectors deployable on edge devices.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2503.02348v1)

---

