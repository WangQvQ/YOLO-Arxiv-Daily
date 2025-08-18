# 每日从arXiv中获取最新YOLO相关论文


## TACR\-YOLO: A Real\-time Detection Framework for Abnormal Human Behaviors Enhanced with Coordinate and Task\-Aware Representations / 

发布日期：2025-08-15

作者：Xinyi Yin

摘要：Abnormal Human Behavior Detection \(AHBD\) under special scenarios is becoming increasingly crucial. While YOLO\-based detection methods excel in real\-time tasks, they remain hindered by challenges including small objects, task conflicts, and multi\-scale fusion in AHBD. To tackle them, we propose TACR\-YOLO, a new real\-time framework for AHBD. We introduce a Coordinate Attention Module to enhance small object detection, a Task\-Aware Attention Module to deal with classification\-regression conflicts, and a Strengthen Neck Network for refined multi\-scale fusion, respectively. In addition, we optimize Anchor Box sizes using K\-means clustering and deploy DIoU\-Loss to improve bounding box regression. The Personnel Anomalous Behavior Detection \(PABD\) dataset, which includes 8,529 samples across four behavior categories, is also presented. Extensive experimental results indicate that TACR\-YOLO achieves 91.92% mAP on PABD, with competitive speed and robustness. Ablation studies highlight the contribution of each improvement. This work provides new insights for abnormal behavior detection under special scenarios, advancing its progress.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2508.11478v1)

---


## Utilizing Vision\-Language Models as Action Models for Intent Recognition and Assistance / 

发布日期：2025-08-14

作者：Cesar Alan Contreras

摘要：Human\-robot collaboration requires robots to quickly infer user intent, provide transparent reasoning, and assist users in achieving their goals. Our recent work introduced GUIDER, our framework for inferring navigation and manipulation intents. We propose augmenting GUIDER with a vision\-language model \(VLM\) and a text\-only language model \(LLM\) to form a semantic prior that filters objects and locations based on the mission prompt. A vision pipeline \(YOLO for object detection and the Segment Anything Model for instance segmentation\) feeds candidate object crops into the VLM, which scores their relevance given an operator prompt; in addition, the list of detected object labels is ranked by a text\-only LLM. These scores weight the existing navigation and manipulation layers of GUIDER, selecting context\-relevant targets while suppressing unrelated objects. Once the combined belief exceeds a threshold, autonomy changes occur, enabling the robot to navigate to the desired area and retrieve the desired object, while adapting to any changes in the operator's intent. Future work will evaluate the system on Isaac Sim using a Franka Emika arm on a Ridgeback base, with a focus on real\-time assistance.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2508.11093v1)

---


## SynSpill: Improved Industrial Spill Detection With Synthetic Data / 

发布日期：2025-08-13

作者：Aaditya Baranwal

摘要：Large\-scale Vision\-Language Models \(VLMs\) have transformed general\-purpose visual recognition through strong zero\-shot capabilities. However, their performance degrades significantly in niche, safety\-critical domains such as industrial spill detection, where hazardous events are rare, sensitive, and difficult to annotate. This scarcity \-\- driven by privacy concerns, data sensitivity, and the infrequency of real incidents \-\- renders conventional fine\-tuning of detectors infeasible for most industrial settings.   We address this challenge by introducing a scalable framework centered on a high\-quality synthetic data generation pipeline. We demonstrate that this synthetic corpus enables effective Parameter\-Efficient Fine\-Tuning \(PEFT\) of VLMs and substantially boosts the performance of state\-of\-the\-art object detectors such as YOLO and DETR. Notably, in the absence of synthetic data \(SynSpill dataset\), VLMs still generalize better to unseen spill scenarios than these detectors. When SynSpill is used, both VLMs and detectors achieve marked improvements, with their performance becoming comparable.   Our results underscore that high\-fidelity synthetic data is a powerful means to bridge the domain gap in safety\-critical applications. The combination of synthetic generation and lightweight adaptation offers a cost\-effective, scalable pathway for deploying vision systems in industrial environments where real data is scarce/impractical to obtain.   Project Page: https://synspill.vercel.app

中文摘要：


代码链接：https://synspill.vercel.app

论文链接：[阅读更多](http://arxiv.org/abs/2508.10171v1)

---


## IPG: Incremental Patch Generation for Generalized Adversarial Patch Training / 

发布日期：2025-08-13

作者：Wonho Lee

摘要：The advent of adversarial patches poses a significant challenge to the robustness of AI models, particularly in the domain of computer vision tasks such as object detection. In contradistinction to traditional adversarial examples, these patches target specific regions of an image, resulting in the malfunction of AI models. This paper proposes Incremental Patch Generation \(IPG\), a method that generates adversarial patches up to 11.1 times more efficiently than existing approaches while maintaining comparable attack performance. The efficacy of IPG is demonstrated by experiments and ablation studies including YOLO's feature distribution visualization and adversarial training results, which show that it produces well\-generalized patches that effectively cover a broader range of model vulnerabilities. Furthermore, IPG\-generated datasets can serve as a robust knowledge foundation for constructing a robust model, enabling structured representation, advanced reasoning, and proactive defenses in AI security ecosystems. The findings of this study suggest that IPG has considerable potential for future utilization not only in adversarial patch defense but also in real\-world applications such as autonomous vehicles, security systems, and medical imaging, where AI models must remain resilient to adversarial attacks in dynamic and high\-stakes environments.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2508.10946v1)

---


## iWatchRoad: Scalable Detection and Geospatial Visualization of Potholes for Smart Cities / 

发布日期：2025-08-13

作者：Rishi Raj Sahoo

摘要：Potholes on the roads are a serious hazard and maintenance burden. This poses a significant threat to road safety and vehicle longevity, especially on the diverse and under\-maintained roads of India. In this paper, we present a complete end\-to\-end system called iWatchRoad for automated pothole detection, Global Positioning System \(GPS\) tagging, and real time mapping using OpenStreetMap \(OSM\). We curated a large, self\-annotated dataset of over 7,000 frames captured across various road types, lighting conditions, and weather scenarios unique to Indian environments, leveraging dashcam footage. This dataset is used to fine\-tune, Ultralytics You Only Look Once \(YOLO\) model to perform real time pothole detection, while a custom Optical Character Recognition \(OCR\) module was employed to extract timestamps directly from video frames. The timestamps are synchronized with GPS logs to geotag each detected potholes accurately. The processed data includes the potholes' details and frames as metadata is stored in a database and visualized via a user friendly web interface using OSM. iWatchRoad not only improves detection accuracy under challenging conditions but also provides government compatible outputs for road assessment and maintenance planning through the metadata visible on the website. Our solution is cost effective, hardware efficient, and scalable, offering a practical tool for urban and rural road management in developing regions, making the system automated. iWatchRoad is available at https://smlab.niser.ac.in/project/iwatchroad

中文摘要：


代码链接：https://smlab.niser.ac.in/project/iwatchroad

论文链接：[阅读更多](http://arxiv.org/abs/2508.10945v1)

---

