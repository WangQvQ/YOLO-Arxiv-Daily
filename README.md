# 每日从arXiv中获取最新YOLO相关论文


## Enhanced Vehicle Speed Detection Considering Lane Recognition Using Drone Videos in California / 

发布日期：2025-06-12

作者：Amirali Ataee Naeini

摘要：The increase in vehicle numbers in California, driven by inadequate transportation systems and sparse speed cameras, necessitates effective vehicle speed detection. Detecting vehicle speeds per lane is critical for monitoring High\-Occupancy Vehicle \(HOV\) lane speeds, distinguishing between cars and heavy vehicles with differing speed limits, and enforcing lane restrictions for heavy vehicles. While prior works utilized YOLO \(You Only Look Once\) for vehicle speed detection, they often lacked accuracy, failed to identify vehicle lanes, and offered limited or less practical classification categories. This study introduces a fine\-tuned YOLOv11 model, trained on almost 800 bird's\-eye view images, to enhance vehicle speed detection accuracy which is much higher compare to the previous works. The proposed system identifies the lane for each vehicle and classifies vehicles into two categories: cars and heavy vehicles. Designed to meet the specific requirements of traffic monitoring and regulation, the model also evaluates the effects of factors such as drone height, distance of Region of Interest \(ROI\), and vehicle speed on detection accuracy and speed measurement. Drone footage collected from Northern California was used to assess the proposed system. The fine\-tuned YOLOv11 achieved its best performance with a mean absolute error \(MAE\) of 0.97 mph and mean squared error \(MSE\) of 0.94 $text\{mph\}^2$, demonstrating its efficacy in addressing challenges in vehicle speed detection and classification.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2506.11239v1)

---


## J\-DDL: Surface Damage Detection and Localization System for Fighter Aircraft / 

发布日期：2025-06-12

作者：Jin Huang

摘要：Ensuring the safety and extended operational life of fighter aircraft necessitates frequent and exhaustive inspections. While surface defect detection is feasible for human inspectors, manual methods face critical limitations in scalability, efficiency, and consistency due to the vast surface area, structural complexity, and operational demands of aircraft maintenance. We propose a smart surface damage detection and localization system for fighter aircraft, termed J\-DDL. J\-DDL integrates 2D images and 3D point clouds of the entire aircraft surface, captured using a combined system of laser scanners and cameras, to achieve precise damage detection and localization. Central to our system is a novel damage detection network built on the YOLO architecture, specifically optimized for identifying surface defects in 2D aircraft images. Key innovations include lightweight Fasternet blocks for efficient feature extraction, an optimized neck architecture incorporating Efficient Multiscale Attention \(EMA\) modules for superior feature aggregation, and the introduction of a novel loss function, Inner\-CIOU, to enhance detection accuracy. After detecting damage in 2D images, the system maps the identified anomalies onto corresponding 3D point clouds, enabling accurate 3D localization of defects across the aircraft surface. Our J\-DDL not only streamlines the inspection process but also ensures more comprehensive and detailed coverage of large and complex aircraft exteriors. To facilitate further advancements in this domain, we have developed the first publicly available dataset specifically focused on aircraft damage. Experimental evaluations validate the effectiveness of our framework, underscoring its potential to significantly advance automated aircraft inspection technologies.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2506.10505v1)

---


## BakuFlow: A Streamlining Semi\-Automatic Label Generation Tool / 

发布日期：2025-06-10

作者：Jerry Lin

摘要：Accurately labeling \(or annotation\) data is still a bottleneck in computer vision, especially for large\-scale tasks where manual labeling is time\-consuming and error\-prone. While tools like LabelImg can handle the labeling task, some of them still require annotators to manually label each image. In this paper, we introduce BakuFlow, a streamlining semi\-automatic label generation tool. Key features include \(1\) a live adjustable magnifier for pixel\-precise manual corrections, improving user experience; \(2\) an interactive data augmentation module to diversify training datasets; \(3\) label propagation for rapidly copying labeled objects between consecutive frames, greatly accelerating annotation of video data; and \(4\) an automatic labeling module powered by a modified YOLOE framework. Unlike the original YOLOE, our extension supports adding new object classes and any number of visual prompts per class during annotation, enabling flexible and scalable labeling for dynamic, real\-world datasets. These innovations make BakuFlow especially effective for object detection and tracking, substantially reducing labeling workload and improving efficiency in practical computer vision and industrial scenarios.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2506.09083v1)

---


## CBAM\-STN\-TPS\-YOLO: Enhancing Agricultural Object Detection through Spatially Adaptive Attention Mechanisms / 

发布日期：2025-06-09

作者：Satvik Praveen

摘要：Object detection is vital in precision agriculture for plant monitoring, disease detection, and yield estimation. However, models like YOLO struggle with occlusions, irregular structures, and background noise, reducing detection accuracy. While Spatial Transformer Networks \(STNs\) improve spatial invariance through learned transformations, affine mappings are insufficient for non\-rigid deformations such as bent leaves and overlaps.   We propose CBAM\-STN\-TPS\-YOLO, a model integrating Thin\-Plate Splines \(TPS\) into STNs for flexible, non\-rigid spatial transformations that better align features. Performance is further enhanced by the Convolutional Block Attention Module \(CBAM\), which suppresses background noise and emphasizes relevant spatial and channel\-wise features.   On the occlusion\-heavy Plant Growth and Phenotyping \(PGP\) dataset, our model outperforms STN\-YOLO in precision, recall, and mAP. It achieves a 12% reduction in false positives, highlighting the benefits of improved spatial flexibility and attention\-guided refinement. We also examine the impact of the TPS regularization parameter in balancing transformation smoothness and detection performance.   This lightweight model improves spatial awareness and supports real\-time edge deployment, making it ideal for smart farming applications requiring accurate and efficient monitoring.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2506.07357v1)

---


## Gen\-n\-Val: Agentic Image Data Generation and Validation / 

发布日期：2025-06-05

作者：Jing\-En Huang

摘要：Recently, Large Language Models \(LLMs\) and Vision Large Language Models \(VLLMs\) have demonstrated impressive performance as agents across various tasks while data scarcity and label noise remain significant challenges in computer vision tasks, such as object detection and instance segmentation. A common solution for resolving these issues is to generate synthetic data. However, current synthetic data generation methods struggle with issues, such as multiple objects per mask, inaccurate segmentation, and incorrect category labels, limiting their effectiveness. To address these issues, we introduce Gen\-n\-Val, a novel agentic data generation framework that leverages Layer Diffusion \(LD\), LLMs, and VLLMs to produce high\-quality, single\-object masks and diverse backgrounds. Gen\-n\-Val consists of two agents: \(1\) The LD prompt agent, an LLM, optimizes prompts for LD to generate high\-quality foreground instance images and segmentation masks. These optimized prompts ensure the generation of single\-object synthetic data with precise instance masks and clean backgrounds. \(2\) The data validation agent, a VLLM, which filters out low\-quality synthetic instance images. The system prompts for both agents are refined through TextGrad. Additionally, we use image harmonization to combine multiple instances within scenes. Compared to state\-of\-the\-art synthetic data approaches like MosaicFusion, our approach reduces invalid synthetic data from 50% to 7% and improves performance by 1% mAP on rare classes in COCO instance segmentation with YOLOv9c and YOLO11m. Furthermore, Gen\-n\-Val shows significant improvements \(7. 1% mAP\) over YOLO\-Worldv2\-M in open\-vocabulary object detection benchmarks with YOLO11m. Moreover, Gen\-n\-Val improves the performance of YOLOv9 and YOLO11 families in instance segmentation and object detection.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2506.04676v1)

---

