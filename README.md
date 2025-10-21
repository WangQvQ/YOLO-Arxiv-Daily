# 每日从arXiv中获取最新YOLO相关论文


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


## BoardVision: Deployment\-ready and Robust Motherboard Defect Detection with YOLO\+Faster\-RCNN Ensemble / 

发布日期：2025-10-16

作者：Brandon Hill

摘要：Motherboard defect detection is critical for ensuring reliability in high\-volume electronics manufacturing. While prior research in PCB inspection has largely targeted bare\-board or trace\-level defects, assembly\-level inspection of full motherboards inspection remains underexplored. In this work, we present BoardVision, a reproducible framework for detecting assembly\-level defects such as missing screws, loose fan wiring, and surface scratches. We benchmark two representative detectors \- YOLOv7 and Faster R\-CNN, under controlled conditions on the MiracleFactory motherboard dataset, providing the first systematic comparison in this domain. To mitigate the limitations of single models, where YOLO excels in precision but underperforms in recall and Faster R\-CNN shows the reverse, we propose a lightweight ensemble, Confidence\-Temporal Voting \(CTV Voter\), that balances precision and recall through interpretable rules. We further evaluate robustness under realistic perturbations including sharpness, brightness, and orientation changes, highlighting stability challenges often overlooked in motherboard defect detection. Finally, we release a deployable GUI\-driven inspection tool that bridges research evaluation with operator usability. Together, these contributions demonstrate how computer vision techniques can transition from benchmark results to practical quality assurance for assembly\-level motherboard manufacturing.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2510.14389v1)

---


## Efficient Few\-Shot Learning in Remote Sensing: Fusing Vision and Vision\-Language Models / 

发布日期：2025-10-15

作者：Jia Yun Chua

摘要：Remote sensing has become a vital tool across sectors such as urban planning, environmental monitoring, and disaster response. While the volume of data generated has increased significantly, traditional vision models are often constrained by the requirement for extensive domain\-specific labelled data and their limited ability to understand the context within complex environments. Vision Language Models offer a complementary approach by integrating visual and textual data; however, their application to remote sensing remains underexplored, particularly given their generalist nature. This work investigates the combination of vision models and VLMs to enhance image analysis in remote sensing, with a focus on aircraft detection and scene understanding. The integration of YOLO with VLMs such as LLaVA, ChatGPT, and Gemini aims to achieve more accurate and contextually aware image interpretation. Performance is evaluated on both labelled and unlabelled remote sensing data, as well as degraded image scenarios which are crucial for remote sensing. The findings show an average MAE improvement of 48.46% across models in the accuracy of aircraft detection and counting, especially in challenging conditions, in both raw and degraded scenarios. A 6.17% improvement in CLIPScore for comprehensive understanding of remote sensing images is obtained. The proposed approach combining traditional vision models and VLMs paves the way for more advanced and efficient remote sensing image analysis, especially in few\-shot learning scenarios.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2510.13993v1)

---

