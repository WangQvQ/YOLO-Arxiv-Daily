# 每日从arXiv中获取最新YOLO相关论文


## LogSTOP: Temporal Scores over Prediction Sequences for Matching and Retrieval / 

发布日期：2025-10-07

作者：Avishree Khare

摘要：Neural models such as YOLO and HuBERT can be used to detect local properties such as objects \("car"\) and emotions \("angry"\) in individual frames of videos and audio clips respectively. The likelihood of these detections is indicated by scores in \[0, 1\]. Lifting these scores to temporal properties over sequences can be useful for several downstream applications such as query matching \(e.g., "does the speaker eventually sound happy in this audio clip?"\), and ranked retrieval \(e.g., "retrieve top 5 videos with a 10 second scene where a car is detected until a pedestrian is detected"\). In this work, we formalize this problem of assigning Scores for TempOral Properties \(STOPs\) over sequences, given potentially noisy score predictors for local properties. We then propose a scoring function called LogSTOP that can efficiently compute these scores for temporal properties represented in Linear Temporal Logic. Empirically, LogSTOP, with YOLO and HuBERT, outperforms Large Vision / Audio Language Models and other Temporal Logic\-based baselines by at least 16% on query matching with temporal properties over objects\-in\-videos and emotions\-in\-speech respectively. Similarly, on ranked retrieval with temporal properties over objects and actions in videos, LogSTOP with Grounding DINO and SlowR50 reports at least a 19% and 16% increase in mean average precision and recall over zero\-shot text\-to\-video retrieval baselines respectively.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2510.06512v1)

---


## Video\-LMM Post\-Training: A Deep Dive into Video Reasoning with Large Multimodal Models / 

发布日期：2025-10-06

作者：Yolo Yunlong Tang

摘要：Video understanding represents the most challenging frontier in computer vision, requiring models to reason about complex spatiotemporal relationships, long\-term dependencies, and multimodal evidence. The recent emergence of Video\-Large Multimodal Models \(Video\-LMMs\), which integrate visual encoders with powerful decoder\-based language models, has demonstrated remarkable capabilities in video understanding tasks. However, the critical phase that transforms these models from basic perception systems into sophisticated reasoning engines, post\-training, remains fragmented across the literature. This survey provides the first comprehensive examination of post\-training methodologies for Video\-LMMs, encompassing three fundamental pillars: supervised fine\-tuning \(SFT\) with chain\-of\-thought, reinforcement learning \(RL\) from verifiable objectives, and test\-time scaling \(TTS\) through enhanced inference computation. We present a structured taxonomy that clarifies the roles, interconnections, and video\-specific adaptations of these techniques, addressing unique challenges such as temporal localization, spatiotemporal grounding, long video efficiency, and multimodal evidence integration. Through systematic analysis of representative methods, we synthesize key design principles, insights, and evaluation protocols while identifying critical open challenges in reward design, scalability, and cost\-performance optimization. We further curate essential benchmarks, datasets, and metrics to facilitate rigorous assessment of post\-training effectiveness. This survey aims to provide researchers and practitioners with a unified framework for advancing Video\-LMM capabilities. Additional resources and updates are maintained at: https://github.com/yunlong10/Awesome\-Video\-LMM\-Post\-Training

中文摘要：


代码链接：https://github.com/yunlong10/Awesome-Video-LMM-Post-Training

论文链接：[阅读更多](http://arxiv.org/abs/2510.05034v2)

---


## Anomaly\-Aware YOLO: A Frugal yet Robust Approach to Infrared Small Target Detection / 

发布日期：2025-10-06

作者：Alina Ciocarlan

摘要：Infrared Small Target Detection \(IRSTD\) is a challenging task in defense applications, where complex backgrounds and tiny target sizes often result in numerous false alarms using conventional object detectors. To overcome this limitation, we propose Anomaly\-Aware YOLO \(AA\-YOLO\), which integrates a statistical anomaly detection test into its detection head. By treating small targets as unexpected patterns against the background, AA\-YOLO effectively controls the false alarm rate. Our approach not only achieves competitive performance on several IRSTD benchmarks, but also demonstrates remarkable robustness in scenarios with limited training data, noise, and domain shifts. Furthermore, since only the detection head is modified, our design is highly generic and has been successfully applied across various YOLO backbones, including lightweight models. It also provides promising results when integrated into an instance segmentation YOLO. This versatility makes AA\-YOLO an attractive solution for real\-world deployments where resources are constrained. The code will be publicly released.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2510.04741v1)

---


## Bio\-Inspired Robotic Houbara: From Development to Field Deployment for Behavioral Studies / 

发布日期：2025-10-06

作者：Lyes Saad Saoud

摘要：Biomimetic intelligence and robotics are transforming field ecology by enabling lifelike robotic surrogates that interact naturally with animals under real world conditions. Studying avian behavior in the wild remains challenging due to the need for highly realistic morphology, durable outdoor operation, and intelligent perception that can adapt to uncontrolled environments. We present a next generation bio inspired robotic platform that replicates the morphology and visual appearance of the female Houbara bustard to support controlled ethological studies and conservation oriented field research. The system introduces a fully digitally replicable fabrication workflow that combines high resolution structured light 3D scanning, parametric CAD modelling, articulated 3D printing, and photorealistic UV textured vinyl finishing to achieve anatomically accurate and durable robotic surrogates. A six wheeled rocker bogie chassis ensures stable mobility on sand and irregular terrain, while an embedded NVIDIA Jetson module enables real time RGB and thermal perception, lightweight YOLO based detection, and an autonomous visual servoing loop that aligns the robot's head toward detected targets without human intervention. A lightweight thermal visible fusion module enhances perception in low light conditions. Field trials in desert aviaries demonstrated reliable real time operation at 15 to 22 FPS with latency under 100 ms and confirmed that the platform elicits natural recognition and interactive responses from live Houbara bustards under harsh outdoor conditions. This integrated framework advances biomimetic field robotics by uniting reproducible digital fabrication, embodied visual intelligence, and ecological validation, providing a transferable blueprint for animal robot interaction research, conservation robotics, and public engagement.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2510.04692v1)

---


## Road Damage and Manhole Detection using Deep Learning for Smart Cities: A Polygonal Annotation Approach / 

发布日期：2025-10-04

作者：Rasel Hossen

摘要：Urban safety and infrastructure maintenance are critical components of smart city development. Manual monitoring of road damages is time\-consuming, highly costly, and error\-prone. This paper presents a deep learning approach for automated road damage and manhole detection using the YOLOv9 algorithm with polygonal annotations. Unlike traditional bounding box annotation, we employ polygonal annotations for more precise localization of road defects. We develop a novel dataset comprising more than one thousand images which are mostly collected from Dhaka, Bangladesh. This dataset is used to train a YOLO\-based model for three classes, namely Broken, Not Broken, and Manhole. We achieve 78.1% overall image\-level accuracy. The YOLOv9 model demonstrates strong performance for Broken \(86.7% F1\-score\) and Not Broken \(89.2% F1\-score\) classes, with challenges in Manhole detection \(18.2% F1\-score\) due to class imbalance. Our approach offers an efficient and scalable solution for monitoring urban infrastructure in developing countries.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2510.03797v1)

---

