# 每日从arXiv中获取最新YOLO相关论文


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


## PatrolVision: Automated License Plate Recognition in the wild / 

发布日期：2025-04-15

作者：Anmol Singhal Navya Singhal

摘要：Adoption of AI driven techniques in public services remains low due to challenges related to accuracy and speed of information at population scale. Computer vision techniques for traffic monitoring have not gained much popularity despite their relative strength in areas such as autonomous driving. Despite large number of academic methods for Automatic License Plate Recognition \(ALPR\) systems, very few provide an end to end solution for patrolling in the city. This paper presents a novel prototype for a low power GPU based patrolling system to be deployed in an urban environment on surveillance vehicles for automated vehicle detection, recognition and tracking. In this work, we propose a complete ALPR system for Singapore license plates having both single and double line creating our own YOLO based network. We focus on unconstrained capture scenarios as would be the case in real world application, where the license plate \(LP\) might be considerably distorted due to oblique views. In this work, we first detect the license plate from the full image using RFB\-Net and rectify multiple distorted license plates in a single image. After that, the detected license plate image is fed to our network for character recognition. We evaluate the performance of our proposed system on a newly built dataset covering more than 16,000 images. The system was able to correctly detect license plates with 86% precision and recognize characters of a license plate in 67% of the test set, and 89% accuracy with one incorrect character \(partial match\). We also test latency of our system and achieve 64FPS on Tesla P4 GPU

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2504.10810v1)

---


## WildLive: Near Real\-time Visual Wildlife Tracking onboard UAVs / 

发布日期：2025-04-14

作者：Nguyen Ngoc Dat

摘要：Live tracking of wildlife via high\-resolution video processing directly onboard drones is widely unexplored and most existing solutions rely on streaming video to ground stations to support navigation. Yet, both autonomous animal\-reactive flight control beyond visual line of sight and/or mission\-specific individual and behaviour recognition tasks rely to some degree on this capability. In response, we introduce WildLive \-\- a near real\-time animal detection and tracking framework for high\-resolution imagery running directly onboard uncrewed aerial vehicles \(UAVs\). The system performs multi\-animal detection and tracking at 17fps\+ for HD and 7fps\+ on 4K video streams suitable for operation during higher altitude flights to minimise animal disturbance. Our system is optimised for Jetson Orin AGX onboard hardware. It integrates the efficiency of sparse optical flow tracking and mission\-specific sampling with device\-optimised and proven YOLO\-driven object detection and segmentation techniques. Essentially, computational resource is focused onto spatio\-temporal regions of high uncertainty to significantly improve UAV processing speeds without domain\-specific loss of accuracy. Alongside, we introduce our WildLive dataset, which comprises 200k\+ annotated animal instances across 19k\+ frames from 4K UAV videos collected at the Ol Pejeta Conservancy in Kenya. All frames contain ground truth bounding boxes, segmentation masks, as well as individual tracklets and tracking point trajectories. We compare our system against current object tracking approaches including OC\-SORT, ByteTrack, and SORT. Our materials are available at: https://dat\-nguyenvn.github.io/WildLive/

中文摘要：


代码链接：https://dat-nguyenvn.github.io/WildLive/

论文链接：[阅读更多](http://arxiv.org/abs/2504.10165v2)

---

