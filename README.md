# 每日从arXiv中获取最新YOLO相关论文


## Enhancing Small Object Detection with YOLO: A Novel Framework for Improved Accuracy and Efficiency / 

发布日期：2025-12-08

作者：Mahila Moghadami

摘要：This paper investigates and develops methods for detecting small objects in large\-scale aerial images. Current approaches for detecting small objects in aerial images often involve image cropping and modifications to detector network architectures. Techniques such as sliding window cropping and architectural enhancements, including higher\-resolution feature maps and attention mechanisms, are commonly employed. Given the growing importance of aerial imagery in various critical and industrial applications, the need for robust frameworks for small object detection becomes imperative. To address this need, we adopted the base SW\-YOLO approach to enhance speed and accuracy in small object detection by refining cropping dimensions and overlap in sliding window usage and subsequently enhanced it through architectural modifications. we propose a novel model by modifying the base model architecture, including advanced feature extraction modules in the neck for feature map enhancement, integrating CBAM in the backbone to preserve spatial and channel information, and introducing a new head to boost small object detection accuracy. Finally, we compared our method with SAHI, one of the most powerful frameworks for processing large\-scale images, and CZDet, which is also based on image cropping, achieving significant improvements in accuracy. The proposed model achieves significant accuracy gains on the VisDrone2019 dataset, outperforming baseline YOLOv5L detection by a substantial margin. Specifically, the final proposed model elevates the mAP .5.5 accuracy on the VisDrone2019 dataset from the base accuracy of 35.5 achieved by the YOLOv5L detector to 61.2. Notably, the accuracy of CZDet, which is another classic method applied to this dataset, is 58.36. This research demonstrates a significant improvement, achieving an increase in accuracy from 35.5 to 61.2.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2512.07379v1)

---


## Hierarchical Image\-Guided 3D Point Cloud Segmentation in Industrial Scenes via Multi\-View Bayesian Fusion / 

发布日期：2025-12-07

作者：Yu Zhu

摘要：Reliable 3D segmentation is critical for understanding complex scenes with dense layouts and multi\-scale objects, as commonly seen in industrial environments. In such scenarios, heavy occlusion weakens geometric boundaries between objects, and large differences in object scale will cause end\-to\-end models fail to capture both coarse and fine details accurately. Existing 3D point\-based methods require costly annotations, while image\-guided methods often suffer from semantic inconsistencies across views. To address these challenges, we propose a hierarchical image\-guided 3D segmentation framework that progressively refines segmentation from instance\-level to part\-level. Instance segmentation involves rendering a top\-view image and projecting SAM\-generated masks prompted by YOLO\-World back onto the 3D point cloud. Part\-level segmentation is subsequently performed by rendering multi\-view images of each instance obtained from the previous stage and applying the same 2D segmentation and back\-projection process at each view, followed by Bayesian updating fusion to ensure semantic consistency across views. Experiments on real\-world factory data demonstrate that our method effectively handles occlusion and structural complexity, achieving consistently high per\-class mIoU scores. Additional evaluations on public dataset confirm the generalization ability of our framework, highlighting its robustness, annotation efficiency, and adaptability to diverse 3D environments.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2512.06882v1)

---


## An Integrated System for WEEE Sorting Employing X\-ray Imaging, AI\-based Object Detection and Segmentation, and Delta Robot Manipulation / 

发布日期：2025-12-05

作者：Panagiotis Giannikos

摘要：Battery recycling is becoming increasingly critical due to the rapid growth in battery usage and the limited availability of natural resources. Moreover, as battery energy densities continue to rise, improper handling during recycling poses significant safety hazards, including potential fires at recycling facilities. Numerous systems have been proposed for battery detection and removal from WEEE recycling lines, including X\-ray and RGB\-based visual inspection methods, typically driven by AI\-powered object detection models \(e.g., Mask R\-CNN, YOLO, ResNets\). Despite advances in optimizing detection techniques and model modifications, a fully autonomous solution capable of accurately identifying and sorting batteries across diverse WEEEs types has yet to be realized. In response to these challenges, we present our novel approach which integrates a specialized X\-ray transmission dual energy imaging subsystem with advanced pre\-processing algorithms, enabling high\-contrast image reconstruction for effective differentiation of dense and thin materials in WEEE. Devices move along a conveyor belt through a high\-resolution X\-ray imaging system, where YOLO and U\-Net models precisely detect and segment battery\-containing items. An intelligent tracking and position estimation algorithm then guides a Delta robot equipped with a suction gripper to selectively extract and properly discard the targeted devices. The approach is validated in a photorealistic simulation environment developed in NVIDIA Isaac Sim and on the real setup.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2512.05599v1)

---


## YOLO and SGBM Integration for Autonomous Tree Branch Detection and Depth Estimation in Radiata Pine Pruning Applications / 

发布日期：2025-12-05

作者：Yida Lin

摘要：Manual pruning of radiata pine trees poses significant safety risks due to extreme working heights and challenging terrain. This paper presents a computer vision framework that integrates YOLO object detection with Semi\-Global Block Matching \(SGBM\) stereo vision for autonomous drone\-based pruning operations. Our system achieves precise branch detection and depth estimation using only stereo camera input, eliminating the need for expensive LiDAR sensors. Experimental evaluation demonstrates YOLO's superior performance over Mask R\-CNN, achieving 82.0% mAPmask50\-95 for branch segmentation. The integrated system accurately localizes branches within a 2 m operational range, with processing times under one second per frame. These results establish the feasibility of cost\-effective autonomous pruning systems that enhance worker safety and operational efficiency in commercial forestry.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2512.05412v1)

---


## MT\-Depth: Multi\-task Instance feature analysis for the Depth Completion / 

发布日期：2025-12-04

作者：Abdul Haseeb Nizamani

摘要：Depth completion plays a vital role in 3D perception systems, especially in scenarios where sparse depth data must be densified for tasks such as autonomous driving, robotics, and augmented reality. While many existing approaches rely on semantic segmentation to guide depth completion, they often overlook the benefits of object\-level understanding. In this work, we introduce an instance\-aware depth completion framework that explicitly integrates binary instance masks as spatial priors to refine depth predictions. Our model combines four main components: a frozen YOLO V11 instance segmentation branch, a U\-Net\-based depth completion backbone, a cross\-attention fusion module, and an attention\-guided prediction head. The instance segmentation branch generates per\-image foreground masks that guide the depth branch via cross\-attention, allowing the network to focus on object\-centric regions during refinement. We validate our method on the Virtual KITTI 2 dataset, showing that it achieves lower RMSE compared to both a U\-Net\-only baseline and previous semantic\-guided methods, while maintaining competitive MAE. Qualitative and quantitative results demonstrate that the proposed model effectively enhances depth accuracy near object boundaries, occlusions, and thin structures. Our findings suggest that incorporating instance\-aware cues offers a promising direction for improving depth completion without relying on dense semantic labels.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2512.04734v1)

---

