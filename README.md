# 每日从arXiv中获取最新YOLO相关论文


## YOLO Network For Defect Detection In Optical lenses / 

发布日期：2025-02-11

作者：Habib Yaseen

摘要：Mass\-produced optical lenses often exhibit defects that alter their scattering properties and compromise quality standards. Manual inspection is usually adopted to detect defects, but it is not recommended due to low accuracy, high error rate and limited scalability. To address these challenges, this study presents an automated defect detection system based on the YOLOv8 deep learning model. A custom dataset of optical lenses, annotated with defect and lens regions, was created to train the model. Experimental results obtained in this study reveal that the system can be used to efficiently and accurately detect defects in optical lenses. The proposed system can be utilized in real\-time industrial environments to enhance quality control processes by enabling reliable and scalable defect detection in optical lens manufacturing.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2502.07592v1)

---


## Data Warehouse Design for Multiple Source Forest Inventory Management and Image Processing / 

发布日期：2025-02-10

作者：Kristina Cormier

摘要：This research developed a prototype data warehouse to integrate multi\-source forestry data for long\-term monitoring, management, and sustainability. The data warehouse is intended to accommodate all types of imagery from various platforms, LiDAR point clouds, survey records, and paper documents, with the capability to transform these datasets into machine learning \(ML\) and deep learning classification and segmentation models. In this study, we pioneered the integration of unmanned aerial vehicle \(UAV\) imagery and paper records, testing the merged data on the YOLOv11 model. Paper records improved ground truth, and preliminary results demonstrated notable performance improvements.   This research aims to implement a data warehouse \(DW\) to manage data for a YOLO \(You Only Look Once\) model, which identifies objects in images. It does this by integrating advanced data processing pipelines. Data are also stored and easily accessible for future use, including comparing current and historical data to understand growth or declining patterns. In addition, the design is used to optimize resource usage. It also scales easily, not affecting other parts of the data warehouse when adding dimension tables or other fields to the fact table. DW performance and estimations for growing workloads are also explored in this paper.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2502.07015v1)

---


## Drone Detection and Tracking with YOLO and a Rule\-based Method / 

发布日期：2025-02-07

作者：Purbaditya Bhattacharya

摘要：Drones or unmanned aerial vehicles are traditionally used for military missions, warfare, and espionage. However, the usage of drones has significantly increased due to multiple industrial applications involving security and inspection, transportation, research purposes, and recreational drone flying. Such an increased volume of drone activity in public spaces requires regulatory actions for purposes of privacy protection and safety. Hence, detection of illegal drone activities such as boundary encroachment becomes a necessity. Such detection tasks are usually automated and performed by deep learning models which are trained on annotated image datasets. This paper builds on a previous work and extends an already published open source dataset. A description and analysis of the entire dataset is provided. The dataset is used to train the YOLOv7 deep learning model and some of its minor variants and the results are provided. Since the detection models are based on a single image input, a simple cross\-correlation based tracker is used to reduce detection drops and improve tracking performance in videos. Finally, the entire drone detection system is summarized.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2502.05292v1)

---


## MHAF\-YOLO: Multi\-Branch Heterogeneous Auxiliary Fusion YOLO for accurate object detection / 

发布日期：2025-02-07

作者：Zhiqiang Yang

摘要：Due to the effective multi\-scale feature fusion capabilities of the Path Aggregation FPN \(PAFPN\), it has become a widely adopted component in YOLO\-based detectors. However, PAFPN struggles to integrate high\-level semantic cues with low\-level spatial details, limiting its performance in real\-world applications, especially with significant scale variations. In this paper, we propose MHAF\-YOLO, a novel detection framework featuring a versatile neck design called the Multi\-Branch Auxiliary FPN \(MAFPN\), which consists of two key modules: the Superficial Assisted Fusion \(SAF\) and Advanced Assisted Fusion \(AAF\). The SAF bridges the backbone and the neck by fusing shallow features, effectively transferring crucial low\-level spatial information with high fidelity. Meanwhile, the AAF integrates multi\-scale feature information at deeper neck layers, delivering richer gradient information to the output layer and further enhancing the model learning capacity. To complement MAFPN, we introduce the Global Heterogeneous Flexible Kernel Selection \(GHFKS\) mechanism and the Reparameterized Heterogeneous Multi\-Scale \(RepHMS\) module to enhance feature fusion. RepHMS is globally integrated into the network, utilizing GHFKS to select larger convolutional kernels for various feature layers, expanding the vertical receptive field and capturing contextual information across spatial hierarchies. Locally, it optimizes convolution by processing both large and small kernels within the same layer, broadening the lateral receptive field and preserving crucial details for detecting smaller targets. The source code of this work is available at: https://github.com/yang0201/MHAF\-YOLO.

中文摘要：


代码链接：https://github.com/yang0201/MHAF-YOLO.

论文链接：[阅读更多](http://arxiv.org/abs/2502.04656v1)

---


## Brain Tumor Identification using Improved YOLOv8 / 

发布日期：2025-02-06

作者：Rupesh Dulal

摘要：Identifying the extent of brain tumors is a significant challenge in brain cancer treatment. The main difficulty is in the approximate detection of tumor size. Magnetic resonance imaging \(MRI\) has become a critical diagnostic tool. However, manually detecting the boundaries of brain tumors from MRI scans is a labor\-intensive task that requires extensive expertise. Deep learning and computer\-aided detection techniques have led to notable advances in machine learning for this purpose. In this paper, we propose a modified You Only Look Once \(YOLOv8\) model to accurately detect the tumors within the MRI images. The proposed model replaced the Non\-Maximum Suppression \(NMS\) algorithm with a Real\-Time Detection Transformer \(RT\- DETR\) in the detection head. NMS filters out redundant or overlapping bounding boxes in the detected tumors, but they are hand\-designed and pre\-set. RT\-DETR removes hand\-designed components. The second improvement was made by replacing the normal convolution block with ghost convolution. Ghost Convolution reduces computational and memory costs while maintaining high accuracy and enabling faster inference, making it ideal for resource\-constrained environments and real\-time applications. The third improvement was made by introducing a vision transformer block in the backbone of YOLOv8 to extract context\-aware features. We used a publicly available dataset of brain tumors in the proposed model. The proposed model performed better than the original YOLOv8 model and also performed better than other object detectors \(Faster R\- CNN, Mask R\-CNN, YOLO, YOLOv3, YOLOv4, YOLOv5, SSD, RetinaNet, EfficientDet, and DETR\). The proposed model achieved 0.91 mAP \(mean Average Precision\)@0.5.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2502.03746v1)

---

