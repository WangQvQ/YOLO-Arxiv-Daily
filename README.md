# 每日从arXiv中获取最新YOLO相关论文


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


## Real\-Time Control and Automation Framework for Acousto\-Holographic Microscopy / 

发布日期：2025-12-03

作者：Hasan Berkay Abdioğlu

摘要：Manual operation of microscopes for repetitive tasks in cell biology is a significant bottleneck, consuming invaluable expert time, and introducing human error. Automation is essential, and while Digital Holographic Microscopy \(DHM\) offers powerful, label\-free quantitative phase imaging \(QPI\), its inherently noisy and low\-contrast holograms make robust autofocus and object detection challenging. We present the design, integration, and validation of a fully automated closed\-loop DHM system engineered for high\-throughput mechanical characterization of biological cells. The system integrates automated serpentine scanning, real\-time YOLO\-based object detection, and a high\-performance, multi\-threaded software architecture using pinned memory and SPSC queues. This design enables the GPU\-accelerated reconstruction pipeline to run fully in parallel with the 50 fps data acquisition, adding no sequential overhead. A key contribution is the validation of a robust, multi\-stage holographic autofocus strategy; we demonstrate that a selected metric \(based on a low\-pass filter and standard deviation\) provides reliable focusing for noisy holograms where conventional methods \(e.g., Tenengrad, Laplacian\) fail entirely. Performance analysis of the complete system identifies the 2.23\-second autofocus operation\-not reconstruction\-as the primary throughput bottleneck, resulting in a 9.62\-second analysis time per object. This work delivers a complete functional platform for autonomous DHM screening and provides a clear, data\-driven path for future optimization, proposing a hybrid brightfield imaging modality to address current bottlenecks.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2512.03539v1)

---


## AfroBeats Dance Movement Analysis Using Computer Vision: A Proof\-of\-Concept Framework Combining YOLO and Segment Anything Model / 

发布日期：2025-12-03

作者：Kwaku Opoku\-Ware

摘要：This paper presents a preliminary investigation into automated dance movement analysis using contemporary computer vision techniques. We propose a proof\-of\-concept framework that integrates YOLOv8 and v11 for dancer detection with the Segment Anything Model \(SAM\) for precise segmentation, enabling the tracking and quantification of dancer movements in video recordings without specialized equipment or markers. Our approach identifies dancers within video frames, counts discrete dance steps, calculates spatial coverage patterns, and measures rhythm consistency across performance sequences. Testing this framework on a single 49\-second recording of Ghanaian AfroBeats dance demonstrates technical feasibility, with the system achieving approximately 94% detection precision and 89% recall on manually inspected samples. The pixel\-level segmentation provided by SAM, achieving approximately 83% intersection\-over\-union with visual inspection, enables motion quantification that captures body configuration changes beyond what bounding\-box approaches can represent. Analysis of this preliminary case study indicates that the dancer classified as primary by our system executed 23% more steps with 37% higher motion intensity and utilized 42% more performance space compared to dancers classified as secondary. However, this work represents an early\-stage investigation with substantial limitations including single\-video validation, absence of systematic ground truth annotations, and lack of comparison with existing pose estimation methods. We present this framework to demonstrate technical feasibility, identify promising directions for quantitative dance metrics, and establish a foundation for future systematic validation studies.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2512.03509v1)

---

