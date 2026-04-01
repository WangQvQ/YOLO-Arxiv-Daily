# 每日从arXiv中获取最新YOLO相关论文


## AutoFormBench: Benchmark Dataset for Automating Form Understanding / 

发布日期：2026-03-31

作者：Gaurab Baral

摘要：Automated processing of structured documents such as government forms, healthcare records, and enterprise invoices remains a persistent challenge due to the high degree of layout variability encountered in real\-world settings. This paper introduces AutoFormBench, a benchmark dataset of 407 annotated real\-world forms spanning government, healthcare, and enterprise domains, designed to train and evaluate form element detection models. We present a systematic comparison of classical OpenCV approaches and four YOLO architectures \(YOLOv8, YOLOv11, YOLOv26\-s, and YOLOv26\-l\) for localizing and classifying fillable form elements. specifically checkboxes, input lines, and text boxes across diverse PDF document types. YOLOv11 demonstrates consistently superior performance in both F1 score and Jaccard accuracy across all element classes and tolerance levels.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2603.29832v1)

---


## Sim\-to\-Real Fruit Detection Using Synthetic Data: Quantitative Evaluation and Embedded Deployment with Isaac Sim / 

发布日期：2026-03-30

作者：Martina Hutter\-Mironovova

摘要：This study investigates the effectiveness of synthetic data for sim\-to\-real transfer in object detection under constrained data conditions and embedded deployment requirements. Synthetic datasets were generated in NVIDIA Isaac Sim and combined with limited real\-world fruit images to train YOLO\-based detection models under real\-only, synthetic\-only, and hybrid regimes. Performance was evaluated on two test datasets: an in\-domain dataset with conditions matching the training data and a domain shift dataset containing real fruit and different background conditions. Results show that models trained exclusively on real data achieve the highest accuracy, while synthetic\-only models exhibit reduced performance due to a domain gap. Hybrid training strategies significantly improve performance compared to synthetic\-only approaches and achieve results close to real\-only training while reducing the need for manual annotation. Under domain shift conditions, all models show performance degradation, with hybrid models providing improved robustness. The trained models were successfully deployed on a Jetson Orin NX using TensorRT optimization, achieving real\-time inference performance. The findings highlight that synthetic data is most effective when used in combination with real data and that deployment constraints must be considered alongside detection accuracy.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2603.28670v1)

---


## Human\-Centric Perception for Child Sexual Abuse Imagery / 

发布日期：2026-03-28

作者：Camila Laranjeira

摘要：Law enforcement agencies and non\-gonvernmental organizations handling reports of Child Sexual Abuse Imagery \(CSAI\) are overwhelmed by large volumes of data, requiring the aid of automation tools. However, defining sexual abuse in images of children is inherently challenging, encompassing sexually explicit activities and hints of sexuality conveyed by the individual's pose, or their attire. CSAI classification methods often rely on black\-box approaches, targeting broad and abstract concepts such as pornography. Thus, our work is an in\-depth exploration of tasks from the literature on Human\-Centric Perception, across the domains of safe images, adult pornography, and CSAI, focusing on targets that enable more objective and explainable pipelines for CSAI classification in the future. We introduce the Body\-Keypoint\-Part Dataset \(BKPD\), gathering images of people from varying age groups and sexual explicitness to approximate the domain of CSAI, along with manually curated hierarchically structured labels for skeletal keypoints and bounding boxes for person and body parts, including head, chest, hip, and hands. We propose two methods, namely BKP\-Association and YOLO\-BKP, for simultaneous pose estimation and detection, with targets associated per individual for a comprehensive decomposed representation of each person. Our methods are benchmarked on COCO\-Keypoints and COCO\-HumanParts, as well as our human\-centric dataset, achieving competitive results with models that jointly perform all tasks. Cross\-domain ablation studies on BKPD and a case study on RCPD highlight the challenges posed by sexually explicit domains. Our study addresses previously unexplored targets in the CSAI domain, paving the way for novel research opportunities.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2603.27290v1)

---


## Autonomous overtaking trajectory optimization using reinforcement learning and opponent pose estimation / 

发布日期：2026-03-28

作者：Matej Rene Cihlar

摘要：Vehicle overtaking is one of the most complex driving maneuvers for autonomous vehicles. To achieve optimal autonomous overtaking, driving systems rely on multiple sensors that enable safe trajectory optimization and overtaking efficiency. This paper presents a reinforcement learning mechanism for multi\-agent autonomous racing environments, enabling overtaking trajectory optimization, based on LiDAR and depth image data. The developed reinforcement learning agent uses pre\-generated raceline data and sensor inputs to compute the steering angle and linear velocity for optimal overtaking. The system uses LiDAR with a 2D detection algorithm and a depth camera with YOLO\-based object detection to identify the vehicle to be overtaken and its pose. The LiDAR and the depth camera detection data are fused using a UKF for improved opponent pose estimation and trajectory optimization for overtaking in racing scenarios. The results show that the proposed algorithm successfully performs overtaking maneuvers in both simulation and real\-world experiments, with pose estimation RMSE of \(0.0816, 0.0531\) m in \(x, y\).

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2603.27207v1)

---


## YOLO Object Detectors for Robotics \-\- a Comparative Study / 

发布日期：2026-03-27

作者：Patryk Niżeniec

摘要：YOLO object detectors recently became a key component of vision systems in many domains. The family of available YOLO models consists of multiple versions, each in various variants. The research reported in this paper aims to validate the applicability of members of this family to detect objects located within the robot workspace. In our experiments, we used our custom dataset and the COCO2017 dataset. To test the robustness of investigated detectors, the images of these datasets were subject to distortions. The results of our experiments, including variations of training/testing configurations and models, may support the choice of the appropriate YOLO version for robotic vision tasks.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2603.27029v1)

---

