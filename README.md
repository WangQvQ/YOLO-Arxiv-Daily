# 每日从arXiv中获取最新YOLO相关论文


## The Urban Vision Hackathon Dataset and Models: Towards Image Annotations and Accurate Vision Models for Indian Traffic / 

发布日期：2025-11-04

作者：Akash Sharma

摘要：This report describes the UVH\-26 dataset, the first public release by AIM@IISc of a large\-scale dataset of annotated traffic\-camera images from India. The dataset comprises 26,646 high\-resolution \(1080p\) images sampled from 2800 Bengaluru's Safe\-City CCTV cameras over a 4\-week period, and subsequently annotated through a crowdsourced hackathon involving 565 college students from across India. In total, 1.8 million bounding boxes were labeled across 14 vehicle classes specific to India: Cycle, 2\-Wheeler \(Motorcycle\), 3\-Wheeler \(Auto\-rickshaw\), LCV \(Light Commercial Vehicles\), Van, Tempo\-traveller, Hatchback, Sedan, SUV, MUV, Mini\-bus, Bus, Truck and Other. Of these, 283k\-316k consensus ground truth bounding boxes and labels were derived for distinct objects in the 26k images using Majority Voting and STAPLE algorithms. Further, we train multiple contemporary detectors, including YOLO11\-S/X, RT\-DETR\-S/X, and DAMO\-YOLO\-T/L using these datasets, and report accuracy based on mAP50, mAP75 and mAP50:95. Models trained on UVH\-26 achieve 8.4\-31.5% improvements in mAP50:95 over equivalent baseline models trained on COCO dataset, with RT\-DETR\-X showing the best performance at 0.67 \(mAP50:95\) as compared to 0.40 for COCO\-trained weights for common classes \(Car, Bus, and Truck\). This demonstrates the benefits of domain\-specific training data for Indian traffic scenarios. The release package provides the 26k images with consensus annotations based on Majority Voting \(UVH\-26\-MV\) and STAPLE \(UVH\-26\-ST\) and the 6 fine\-tuned YOLO and DETR models on each of these datasets. By capturing the heterogeneity of Indian urban mobility directly from operational traffic\-camera streams, UVH\-26 addresses a critical gap in existing global benchmarks, and offers a foundation for advancing detection, classification, and deployment of intelligent transportation systems in emerging nations with complex traffic conditions.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2511.02563v1)

---


## ASTROFLOW: A Real\-Time End\-to\-End Pipeline for Radio Single\-Pulse Searches / 

发布日期：2025-11-04

作者：Guanhong Lin

摘要：Fast radio bursts \(FRBs\) are extremely bright, millisecond duration cosmic transients of unknown origin. The growing number of wide\-field and high\-time\-resolution radio surveys, particularly with next\-generation facilities such as the SKA and MeerKAT, will dramatically increase FRB discovery rates, but also produce data volumes that overwhelm conventional search pipelines. Real\-time detection thus demands software that is both algorithmically robust and computationally efficient. We present Astroflow, an end\-to\-end, GPU\-accelerated pipeline for single\-pulse detection in radio time\-frequency data. Built on a unified C\+\+/CUDA core with a Python interface, Astroflow integrates RFI excision, incoherent dedispersion, dynamic\-spectrum tiling, and a YOLO\-based deep detector. Through vectorized memory access, shared\-memory tiling, and OpenMP parallelism, it achieves 10x faster\-than\-real\-time processing on consumer GPUs for a typical 150 s, 2048\-channel observation, while preserving high sensitivity across a wide range of pulse widths and dispersion measures. These results establish the feasibility of a fully integrated, GPU\-accelerated single\-pulse search stack, capable of scaling to the data volumes expected from upcoming large\-scale surveys. Astroflow offers a reusable and deployable solution for real\-time transient discovery, and provides a framework that can be continuously refined with new data and models.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2511.02328v1)

---


## Autobiasing Event Cameras for Flickering Mitigation / 

发布日期：2025-11-04

作者：Mehdi Sefidgar Dilmaghani

摘要：Understanding and mitigating flicker effects caused by rapid variations in light intensity is critical for enhancing the performance of event cameras in diverse environments. This paper introduces an innovative autonomous mechanism for tuning the biases of event cameras, effectively addressing flicker across a wide frequency range \-25 Hz to 500 Hz. Unlike traditional methods that rely on additional hardware or software for flicker filtering, our approach leverages the event cameras inherent bias settings. Utilizing a simple Convolutional Neural Networks \-CNNs, the system identifies instances of flicker in a spatial space and dynamically adjusts specific biases to minimize its impact. The efficacy of this autobiasing system was robustly tested using a face detector framework under both well\-lit and low\-light conditions, as well as across various frequencies. The results demonstrated significant improvements: enhanced YOLO confidence metrics for face detection, and an increased percentage of frames capturing detected faces. Moreover, the average gradient, which serves as an indicator of flicker presence through edge detection, decreased by 38.2 percent in well\-lit conditions and by 53.6 percent in low\-light conditions. These findings underscore the potential of our approach to significantly improve the functionality of event cameras in a range of adverse lighting scenarios.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2511.02180v1)

---


## Mask\-to\-Height: A YOLOv11\-Based Architecture for Joint Building Instance Segmentation and Height Classification from Satellite Imagery / 

发布日期：2025-10-31

作者：Mahmoud El Hussieni

摘要：Accurate building instance segmentation and height classification are critical for urban planning, 3D city modeling, and infrastructure monitoring. This paper presents a detailed analysis of YOLOv11, the recent advancement in the YOLO series of deep learning models, focusing on its application to joint building extraction and discrete height classification from satellite imagery. YOLOv11 builds on the strengths of earlier YOLO models by introducing a more efficient architecture that better combines features at different scales, improves object localization accuracy, and enhances performance in complex urban scenes. Using the DFC2023 Track 2 dataset \-\- which includes over 125,000 annotated buildings across 12 cities \-\- we evaluate YOLOv11's performance using metrics such as precision, recall, F1 score, and mean average precision \(mAP\). Our findings demonstrate that YOLOv11 achieves strong instance segmentation performance with 60.4% mAP@50 and 38.3% mAP@50\-\-95 while maintaining robust classification accuracy across five predefined height tiers. The model excels in handling occlusions, complex building shapes, and class imbalance, particularly for rare high\-rise structures. Comparative analysis confirms that YOLOv11 outperforms earlier multitask frameworks in both detection accuracy and inference speed, making it well\-suited for real\-time, large\-scale urban mapping. This research highlights YOLOv11's potential to advance semantic urban reconstruction through streamlined categorical height modeling, offering actionable insights for future developments in remote sensing and geospatial intelligence.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2510.27224v1)

---


## maxVSTAR: Maximally Adaptive Vision\-Guided CSI Sensing with Closed\-Loop Edge Model Adaptation for Robust Human Activity Recognition / 

发布日期：2025-10-30

作者：Kexing Liu

摘要：WiFi Channel State Information \(CSI\)\-based human activity recognition \(HAR\) provides a privacy\-preserving, device\-free sensing solution for smart environments. However, its deployment on edge devices is severely constrained by domain shift, where recognition performance deteriorates under varying environmental and hardware conditions. This study presents maxVSTAR \(maximally adaptive Vision\-guided Sensing Technology for Activity Recognition\), a closed\-loop, vision\-guided model adaptation framework that autonomously mitigates domain shift for edge\-deployed CSI sensing systems. The proposed system integrates a cross\-modal teacher\-student architecture, where a high\-accuracy YOLO\-based vision model serves as a dynamic supervisory signal, delivering real\-time activity labels for the CSI data stream. These labels enable autonomous, online fine\-tuning of a lightweight CSI\-based HAR model, termed Sensing Technology for Activity Recognition \(STAR\), directly at the edge. This closed\-loop retraining mechanism allows STAR to continuously adapt to environmental changes without manual intervention. Extensive experiments demonstrate the effectiveness of maxVSTAR. When deployed on uncalibrated hardware, the baseline STAR model's recognition accuracy declined from 93.52% to 49.14%. Following a single vision\-guided adaptation cycle, maxVSTAR restored the accuracy to 81.51%. These results confirm the system's capacity for dynamic, self\-supervised model adaptation in privacy\-conscious IoT environments, establishing a scalable and practical paradigm for long\-term autonomous HAR using CSI sensing at the network edge.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2510.26146v1)

---

