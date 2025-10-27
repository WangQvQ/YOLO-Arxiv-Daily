# 每日从arXiv中获取最新YOLO相关论文


## Deep learning\-based automated damage detection in concrete structures using images from earthquake events / 

发布日期：2025-10-24

作者：Abdullah Turer

摘要：Timely assessment of integrity of structures after seismic events is crucial for public safety and emergency response. This study focuses on assessing the structural damage conditions using deep learning methods to detect exposed steel reinforcement in concrete buildings and bridges after large earthquakes. Steel bars are typically exposed after concrete spalling or large flexural or shear cracks. The amount and distribution of exposed steel reinforcement is an indication of structural damage and degradation. To automatically detect exposed steel bars, new datasets of images collected after the 2023 Turkey Earthquakes were labeled to represent a wide variety of damaged concrete structures. The proposed method builds upon a deep learning framework, enhanced with fine\-tuning, data augmentation, and testing on public datasets. An automated classification framework is developed that can be used to identify inside/outside buildings and structural components. Then, a YOLOv11 \(You Only Look Once\) model is trained to detect cracking and spalling damage and exposed bars. Another YOLO model is finetuned to distinguish different categories of structural damage levels. All these trained models are used to create a hybrid framework to automatically and reliably determine the damage levels from input images. This research demonstrates that rapid and automated damage detection following disasters is achievable across diverse damage contexts by utilizing image data collection, annotation, and deep learning approaches.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2510.21063v1)

---


## Big Data, Tiny Targets: An Exploratory Study in Machine Learning\-enhanced Detection of Microplastic from Filters / 

发布日期：2025-10-20

作者：Paul\-Tiberiu Miclea

摘要：Microplastics \(MPs\) are ubiquitous pollutants with demonstrated potential to impact ecosystems and human health. Their microscopic size complicates detection, classification, and removal, especially in biological and environmental samples. While techniques like optical microscopy, Scanning Electron Microscopy \(SEM\), and Atomic Force Microscopy \(AFM\) provide a sound basis for detection, applying these approaches requires usually manual analysis and prevents efficient use in large screening studies. To this end, machine learning \(ML\) has emerged as a powerful tool in advancing microplastic detection. In this exploratory study, we investigate potential, limitations and future directions of advancing the detection and quantification of MP particles and fibres using a combination of SEM imaging and machine learning\-based object detection. For simplicity, we focus on a filtration scenario where image backgrounds exhibit a symmetric and repetitive pattern. Our findings indicate differences in the quality of YOLO models for the given task and the relevance of optimizing preprocessing. At the same time, we identify open challenges, such as limited amounts of expert\-labeled data necessary for reliable training of ML models.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2510.18089v1)

---


## Machine Vision\-Based Surgical Lighting System:Design and Implementation / 

发布日期：2025-10-20

作者：Amir Gharghabi

摘要：Effortless and ergonomically designed surgical lighting is critical for precision and safety during procedures. However, traditional systems often rely on manual adjustments, leading to surgeon fatigue, neck strain, and inconsistent illumination due to drift and shadowing. To address these challenges, we propose a novel surgical lighting system that leverages the YOLOv11 object detection algorithm to identify a blue marker placed above the target surgical site. A high\-power LED light source is then directed to the identified location using two servomotors equipped with tilt\-pan brackets. The YOLO model achieves 96.7% mAP@50 on the validation set consisting of annotated images simulating surgical scenes with the blue spherical marker. By automating the lighting process, this machine vision\-based solution reduces physical strain on surgeons, improves consistency in illumination, and supports improved surgical outcomes.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2510.17287v1)

---


## Towards Intelligent Traffic Signaling in Dhaka City Based on Vehicle Detection and Congestion Optimization / 

发布日期：2025-10-18

作者：Kazi Ababil Azam

摘要：The vehicular density in urbanizing cities of developing countries such as Dhaka, Bangladesh result in a lot of traffic congestion, causing poor on\-road experiences. Traffic signaling is a key component in effective traffic management for such situations, but the advancements in intelligent traffic signaling have been exclusive to developed countries with structured traffic. The non\-lane\-based, heterogeneous traffic of Dhaka City requires a contextual approach. This study focuses on the development of an intelligent traffic signaling system feasible in the context of developing countries such as Bangladesh. We propose a pipeline leveraging Real Time Streaming Protocol \(RTSP\) feeds, a low resources system Raspberry Pi 4B processing, and a state of the art YOLO\-based object detection model trained on the Non\-lane\-based and Heterogeneous Traffic \(NHT\-1071\) dataset to detect and classify heterogeneous traffic. A multi\-objective optimization algorithm, NSGA\-II, then generates optimized signal timings, minimizing waiting time while maximizing vehicle throughput. We test our implementation in a five\-road intersection at Palashi, Dhaka, demonstrating the potential to significantly improve traffic management in similar situations. The developed testbed paves the way for more contextual and effective Intelligent Traffic Signaling \(ITS\) solutions for developing areas with complicated traffic dynamics such as Dhaka City.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2510.16622v1)

---


## iWatchRoadv2: Pothole Detection, Geospatial Mapping, and Intelligent Road Governance / 

发布日期：2025-10-18

作者：Rishi Raj Sahoo

摘要：Road potholes pose significant safety hazards and maintenance challenges, particularly on India's diverse and under\-maintained road networks. This paper presents iWatchRoadv2, a fully automated end\-to\-end platform for real\-time pothole detection, GPS\-based geotagging, and dynamic road health visualization using OpenStreetMap \(OSM\). We curated a self\-annotated dataset of over 7,000 dashcam frames capturing diverse Indian road conditions, weather patterns, and lighting scenarios, which we used to fine\-tune the Ultralytics YOLO model for accurate pothole detection. The system synchronizes OCR\-extracted video timestamps with external GPS logs to precisely geolocate each detected pothole, enriching detections with comprehensive metadata, including road segment attribution and contractor information managed through an optimized backend database. iWatchRoadv2 introduces intelligent governance features that enable authorities to link road segments with contract metadata through a secure login interface. The system automatically sends alerts to contractors and officials when road health deteriorates, supporting automated accountability and warranty enforcement. The intuitive web interface delivers actionable analytics to stakeholders and the public, facilitating evidence\-driven repair planning, budget allocation, and quality assessment. Our cost\-effective and scalable solution streamlines frame processing and storage while supporting seamless public engagement for urban and rural deployments. By automating the complete pothole monitoring lifecycle, from detection to repair verification, iWatchRoadv2 enables data\-driven smart city management, transparent governance, and sustainable improvements in road infrastructure maintenance. The platform and live demonstration are accessible at https://smlab.niser.ac.in/project/iwatchroad.

中文摘要：


代码链接：https://smlab.niser.ac.in/project/iwatchroad.

论文链接：[阅读更多](http://arxiv.org/abs/2510.16375v1)

---

