# 每日从arXiv中获取最新YOLO相关论文


## A Novel Attention\-Augmented Wavelet YOLO System for Real\-time Brain Vessel Segmentation on Transcranial Color\-coded Doppler / 

发布日期：2025-08-19

作者：Wenxuan Zhang

摘要：The Circle of Willis \(CoW\), vital for ensuring consistent blood flow to the brain, is closely linked to ischemic stroke. Accurate assessment of the CoW is important for identifying individuals at risk and guiding appropriate clinical management. Among existing imaging methods, Transcranial Color\-coded Doppler \(TCCD\) offers unique advantages due to its radiation\-free nature, affordability, and accessibility. However, reliable TCCD assessments depend heavily on operator expertise for identifying anatomical landmarks and performing accurate angle correction, which limits its widespread adoption. To address this challenge, we propose an AI\-powered, real\-time CoW auto\-segmentation system capable of efficiently capturing cerebral arteries. No prior studies have explored AI\-driven cerebrovascular segmentation using TCCD. In this work, we introduce a novel Attention\-Augmented Wavelet YOLO \(AAW\-YOLO\) network tailored for TCCD data, designed to provide real\-time guidance for brain vessel segmentation in the CoW. We prospectively collected TCCD data comprising 738 annotated frames and 3,419 labeled artery instances to establish a high\-quality dataset for model training and evaluation. The proposed AAW\-YOLO demonstrated strong performance in segmenting both ipsilateral and contralateral CoW vessels, achieving an average Dice score of 0.901, IoU of 0.823, precision of 0.882, recall of 0.926, and mAP of 0.953, with a per\-frame inference speed of 14.199 ms. This system offers a practical solution to reduce reliance on operator experience in TCCD\-based cerebrovascular screening, with potential applications in routine clinical workflows and resource\-constrained settings. Future research will explore bilateral modeling and larger\-scale validation.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2508.13875v1)

---


## Mechanical Automation with Vision: A Design for Rubik's Cube Solver / 

发布日期：2025-08-17

作者：Abhinav Chalise

摘要：The core mechanical system is built around three stepper motors for physical manipulation, a microcontroller for hardware control, a camera and YOLO detection model for real\-time cube state detection. A significant software component is the development of a user\-friendly graphical user interface \(GUI\) designed in Unity. The initial state after detection from real\-time YOLOv8 model \(Precision 0.98443, Recall 0.98419, Box Loss 0.42051, Class Loss 0.2611\) is virtualized on GUI. To get the solution, the system employs the Kociemba's algorithm while physical manipulation with a single degree of freedom is done by combination of stepper motors' interaction with the cube achieving the average solving time of ~2.2 minutes.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2508.12469v1)

---


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

