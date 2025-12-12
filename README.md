# 每日从arXiv中获取最新YOLO相关论文


## Enhancing Large Language Models for End\-to\-End Circuit Analysis Problem Solving / 

发布日期：2025-12-10

作者：Liangliang Chen

摘要：Large language models \(LLMs\) have shown strong performance in data\-rich domains such as programming, but their reliability in engineering tasks remains limited. Circuit analysis \-\- requiring multimodal understanding and precise mathematical reasoning \-\- highlights these challenges. Although Gemini 2.5 Pro improves diagram interpretation and analog\-circuit reasoning, it still struggles to consistently produce correct solutions when given both text and circuit diagrams. At the same time, engineering education needs scalable AI tools capable of generating accurate solutions for tasks such as automated homework feedback and question\-answering. This paper presents an enhanced, end\-to\-end circuit problem solver built on Gemini 2.5 Pro. We first benchmark Gemini on a representative set of undergraduate circuit problems and identify two major failure modes: 1\) circuit\-recognition hallucinations, particularly incorrect source polarity detection, and 2\) reasoning\-process hallucinations, such as incorrect current directions. To address recognition errors, we integrate a fine\-tuned YOLO detector and OpenCV processing to isolate voltage and current sources, enabling Gemini to re\-identify source polarities from cropped images with near\-perfect accuracy. To reduce reasoning errors, we introduce an ngspice\-based verification loop in which Gemini generates a .cir file, ngspice simulates the circuit, and discrepancies trigger iterative regeneration with optional human\-in\-the\-loop review. Across 83 problems, the proposed pipeline achieves a 97.59% success rate \(81 correct solutions\), substantially outperforming Gemini 2.5 Pro's original 79.52% accuracy. This system extends LLM capabilities for multimodal engineering problem\-solving and supports the creation of high\-quality educational datasets and AI\-powered instructional tools.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2512.10159v1)

---


## LiM\-YOLO: Less is More with Pyramid Level Shift and Normalized Auxiliary Branch for Ship Detection in Optical Remote Sensing Imagery / 

发布日期：2025-12-10

作者：Seon\-Hoon Kim

摘要：Applying general\-purpose object detectors to ship detection in satellite imagery presents significant challenges due to the extreme scale disparity and morphological anisotropy of maritime targets. Standard architectures utilizing stride\-32 \(P5\) layers often fail to resolve narrow vessels, resulting in spatial feature dilution. In this work, we propose LiM\-YOLO, a specialized detector designed to resolve these domain\-specific conflicts. Based on a statistical analysis of ship scales, we introduce a Pyramid Level Shift Strategy that reconfigures the detection head to P2\-P4. This shift ensures compliance with Nyquist sampling criteria for small objects while eliminating the computational redundancy of deep layers. To further enhance training stability on high\-resolution inputs, we incorporate a Group Normalized Convolutional Block for Linear Projection \(GN\-CBLinear\), which mitigates gradient volatility in micro\-batch settings. Validated on SODA\-A, DOTA\-v1.5, FAIR1M\-v2.0, and ShipRSImageNet\-V1, LiM\-YOLO demonstrates superior detection accuracy and efficiency compared to state\-of\-the\-art models. The code is available at https://github.com/egshkim/LiM\-YOLO.

中文摘要：


代码链接：https://github.com/egshkim/LiM-YOLO.

论文链接：[阅读更多](http://arxiv.org/abs/2512.09700v1)

---


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

