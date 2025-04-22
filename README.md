# 每日从arXiv中获取最新YOLO相关论文


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


## YOLO\-RS: Remote Sensing Enhanced Crop Detection Methods / 

发布日期：2025-04-15

作者：Linlin Xiao

摘要：With the rapid development of remote sensing technology, crop classification and health detection based on deep learning have gradually become a research hotspot. However, the existing target detection methods show poor performance when dealing with small targets in remote sensing images, especially in the case of complex background and image mixing, which is difficult to meet the practical application requirementsite. To address this problem, a novel target detection model YOLO\-RS is proposed in this paper. The model is based on the latest Yolov11 which significantly enhances the detection of small targets by introducing the Context Anchor Attention \(CAA\) mechanism and an efficient multi\-field multi\-scale feature fusion network. YOLO\-RS adopts a bidirectional feature fusion strategy in the feature fusion process, which effectively enhances the model's performance in the detection of small targets. Small target detection. Meanwhile, the ACmix module at the end of the model backbone network solves the category imbalance problem by adaptively adjusting the contrast and sample mixing, thus enhancing the detection accuracy in complex scenes. In the experiments on the PDT remote sensing crop health detection dataset and the CWC crop classification dataset, YOLO\-RS improves both the recall and the mean average precision \(mAP\) by about 2\-3% or so compared with the existing state\-of\-the\-art methods, while the F1\-score is also significantly improved. Moreover, the computational complexity of the model only increases by about 5.2 GFLOPs, indicating its significant advantages in both performance and efficiency. The experimental results validate the effectiveness and application potential of YOLO\-RS in the task of detecting small targets in remote sensing images.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2504.11165v1)

---

