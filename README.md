# 每日从arXiv中获取最新YOLO相关论文


## Vision Controlled Orthotic Hand Exoskeleton / 

发布日期：2025-04-22

作者：Connor Blais

摘要：This paper presents the design and implementation of an AI vision\-controlled orthotic hand exoskeleton to enhance rehabilitation and assistive functionality for individuals with hand mobility impairments. The system leverages a Google Coral Dev Board Micro with an Edge TPU to enable real\-time object detection using a customized MobileNet\_V2 model trained on a six\-class dataset. The exoskeleton autonomously detects objects, estimates proximity, and triggers pneumatic actuation for grasp\-and\-release tasks, eliminating the need for user\-specific calibration needed in traditional EMG\-based systems. The design prioritizes compactness, featuring an internal battery. It achieves an 8\-hour runtime with a 1300 mAh battery. Experimental results demonstrate a 51ms inference speed, a significant improvement over prior iterations, though challenges persist in model robustness under varying lighting conditions and object orientations. While the most recent YOLO model \(YOLOv11\) showed potential with 15.4 FPS performance, quantization issues hindered deployment. The prototype underscores the viability of vision\-controlled exoskeletons for real\-world assistive applications, balancing portability, efficiency, and real\-time responsiveness, while highlighting future directions for model optimization and hardware miniaturization.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2504.16319v1)

---


## ISTD\-YOLO: A Multi\-Scale Lightweight High\-Performance Infrared Small Target Detection Algorithm / 

发布日期：2025-04-19

作者：Shang Zhang

摘要：Aiming at the detection difficulties of infrared images such as complex background, low signal\-to\-noise ratio, small target size and weak brightness, a lightweight infrared small target detection algorithm ISTD\-YOLO based on improved YOLOv7 was proposed. Firstly, the YOLOv7 network structure was lightweight reconstructed, and a three\-scale lightweight network architecture was designed. Then, the ELAN\-W module of the model neck network is replaced by VoV\-GSCSP to reduce the computational cost and the complexity of the network structure. Secondly, a parameter\-free attention mechanism was introduced into the neck network to enhance the relevance of local con\-text information. Finally, the Normalized Wasserstein Distance \(NWD\) was used to optimize the commonly used IoU index to enhance the localization and detection accuracy of small targets. Experimental results show that compared with YOLOv7 and the current mainstream algorithms, ISTD\-YOLO can effectively improve the detection effect, and all indicators are effectively improved, which can achieve high\-quality detection of infrared small targets.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2504.14289v1)

---


## RF\-DETR Object Detection vs YOLOv12 : A Study of Transformer\-based and CNN\-based Architectures for Single\-Class and Multi\-Class Greenfruit Detection in Complex Orchard Environments Under Label Ambiguity / 

发布日期：2025-04-17

作者：Ranjan Sapkota

摘要：This study conducts a detailed comparison of RF\-DETR object detection base model and YOLOv12 object detection model configurations for detecting greenfruits in a complex orchard environment marked by label ambiguity, occlusions, and background blending. A custom dataset was developed featuring both single\-class \(greenfruit\) and multi\-class \(occluded and non\-occluded greenfruits\) annotations to assess model performance under dynamic real\-world conditions. RF\-DETR object detection model, utilizing a DINOv2 backbone and deformable attention, excelled in global context modeling, effectively identifying partially occluded or ambiguous greenfruits. In contrast, YOLOv12 leveraged CNN\-based attention for enhanced local feature extraction, optimizing it for computational efficiency and edge deployment. RF\-DETR achieved the highest mean Average Precision \(mAP50\) of 0.9464 in single\-class detection, proving its superior ability to localize greenfruits in cluttered scenes. Although YOLOv12N recorded the highest mAP@50:95 of 0.7620, RF\-DETR consistently outperformed in complex spatial scenarios. For multi\-class detection, RF\-DETR led with an mAP@50 of 0.8298, showing its capability to differentiate between occluded and non\-occluded fruits, while YOLOv12L scored highest in mAP@50:95 with 0.6622, indicating better classification in detailed occlusion contexts. Training dynamics analysis highlighted RF\-DETR's swift convergence, particularly in single\-class settings where it plateaued within 10 epochs, demonstrating the efficiency of transformer\-based architectures in adapting to dynamic visual data. These findings validate RF\-DETR's effectiveness for precision agricultural applications, with YOLOv12 suited for fast\-response scenarios. >Index Terms: RF\-DETR object detection, YOLOv12, YOLOv13, YOLOv14, YOLOv15, YOLOE, YOLO World, YOLO, You Only Look Once, Roboflow, Detection Transformers, CNNs

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2504.13099v1)

---


## A Review of YOLOv12: Attention\-Based Enhancements vs. Previous Versions / 

发布日期：2025-04-16

作者：Rahima Khanam

摘要：The YOLO \(You Only Look Once\) series has been a leading framework in real\-time object detection, consistently improving the balance between speed and accuracy. However, integrating attention mechanisms into YOLO has been challenging due to their high computational overhead. YOLOv12 introduces a novel approach that successfully incorporates attention\-based enhancements while preserving real\-time performance. This paper provides a comprehensive review of YOLOv12's architectural innovations, including Area Attention for computationally efficient self\-attention, Residual Efficient Layer Aggregation Networks for improved feature aggregation, and FlashAttention for optimized memory access. Additionally, we benchmark YOLOv12 against prior YOLO versions and competing object detectors, analyzing its improvements in accuracy, inference speed, and computational efficiency. Through this analysis, we demonstrate how YOLOv12 advances real\-time object detection by refining the latency\-accuracy trade\-off and optimizing computational resources.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2504.11995v1)

---


## CFIS\-YOLO: A Lightweight Multi\-Scale Fusion Network for Edge\-Deployable Wood Defect Detection / 

发布日期：2025-04-15

作者：Jincheng Kang

摘要：Wood defect detection is critical for ensuring quality control in the wood processing industry. However, current industrial applications face two major challenges: traditional methods are costly, subjective, and labor\-intensive, while mainstream deep learning models often struggle to balance detection accuracy and computational efficiency for edge deployment. To address these issues, this study proposes CFIS\-YOLO, a lightweight object detection model optimized for edge devices. The model introduces an enhanced C2f structure, a dynamic feature recombination module, and a novel loss function that incorporates auxiliary bounding boxes and angular constraints. These innovations improve multi\-scale feature fusion and small object localization while significantly reducing computational overhead. Evaluated on a public wood defect dataset, CFIS\-YOLO achieves a mean Average Precision \(mAP@0.5\) of 77.5%, outperforming the baseline YOLOv10s by 4 percentage points. On SOPHON BM1684X edge devices, CFIS\-YOLO delivers 135 FPS, reduces power consumption to 17.3% of the original implementation, and incurs only a 0.5 percentage point drop in mAP. These results demonstrate that CFIS\-YOLO is a practical and effective solution for real\-world wood defect detection in resource\-constrained environments.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2504.11305v1)

---

