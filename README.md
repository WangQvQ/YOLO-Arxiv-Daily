# 每日从arXiv中获取最新YOLO相关论文


## HierLight\-YOLO: A Hierarchical and Lightweight Object Detection Network for UAV Photography / 

发布日期：2025-09-26

作者：Defan Chen

摘要：The real\-time detection of small objects in complex scenes, such as the unmanned aerial vehicle \(UAV\) photography captured by drones, has dual challenges of detecting small targets \(<32 pixels\) and maintaining real\-time efficiency on resource\-constrained platforms. While YOLO\-series detectors have achieved remarkable success in real\-time large object detection, they suffer from significantly higher false negative rates for drone\-based detection where small objects dominate, compared to large object scenarios. This paper proposes HierLight\-YOLO, a hierarchical feature fusion and lightweight model that enhances the real\-time detection of small objects, based on the YOLOv8 architecture. We propose the Hierarchical Extended Path Aggregation Network \(HEPAN\), a multi\-scale feature fusion method through hierarchical cross\-level connections, enhancing the small object detection accuracy. HierLight\-YOLO includes two innovative lightweight modules: Inverted Residual Depthwise Convolution Block \(IRDCB\) and Lightweight Downsample \(LDown\) module, which significantly reduce the model's parameters and computational complexity without sacrificing detection capabilities. Small object detection head is designed to further enhance spatial resolution and feature fusion to tackle the tiny object \(4 pixels\) detection. Comparison experiments and ablation studies on the VisDrone2019 benchmark demonstrate state\-of\-the\-art performance of HierLight\-YOLO.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2509.22365v1)

---


## MS\-YOLO: Infrared Object Detection for Edge Deployment via MobileNetV4 and SlideLoss / 

发布日期：2025-09-25

作者：Jiali Zhang

摘要：Infrared imaging has emerged as a robust solution for urban object detection under low\-light and adverse weather conditions, offering significant advantages over traditional visible\-light cameras. However, challenges such as class imbalance, thermal noise, and computational constraints can significantly hinder model performance in practical settings. To address these issues, we evaluate multiple YOLO variants on the FLIR ADAS V2 dataset, ultimately selecting YOLOv8 as our baseline due to its balanced accuracy and efficiency. Building on this foundation, we present texttt\{MS\-YOLO\} \(textbf\{M\}obileNetv4 and textbf\{S\}lideLoss based on YOLO\), which replaces YOLOv8's CSPDarknet backbone with the more efficient MobileNetV4, reducing computational overhead by textbf\{1.5%\} while sustaining high accuracy. In addition, we introduce emph\{SlideLoss\}, a novel loss function that dynamically emphasizes under\-represented and occluded samples, boosting precision without sacrificing recall. Experiments on the FLIR ADAS V2 benchmark show that texttt\{MS\-YOLO\} attains competitive mAP and superior precision while operating at only textbf\{6.7 GFLOPs\}. These results demonstrate that texttt\{MS\-YOLO\} effectively addresses the dual challenge of maintaining high detection quality while minimizing computational costs, making it well\-suited for real\-time edge deployment in urban environments.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2509.21696v1)

---


## DroneFL: Federated Learning for Multi\-UAV Visual Target Tracking / 

发布日期：2025-09-25

作者：Xiaofan Yu

摘要：Multi\-robot target tracking is a fundamental problem that requires coordinated monitoring of dynamic entities in applications such as precision agriculture, environmental monitoring, disaster response, and security surveillance. While Federated Learning \(FL\) has the potential to enhance learning across multiple robots without centralized data aggregation, its use in multi\-Unmanned Aerial Vehicle \(UAV\) target tracking remains largely underexplored. Key challenges include limited onboard computational resources, significant data heterogeneity in FL due to varying targets and the fields of view, and the need for tight coupling between trajectory prediction and multi\-robot planning. In this paper, we introduce DroneFL, the first federated learning framework specifically designed for efficient multi\-UAV target tracking. We design a lightweight local model to predict target trajectories from sensor inputs, using a frozen YOLO backbone and a shallow transformer for efficient onboard training. The updated models are periodically aggregated in the cloud for global knowledge sharing. To alleviate the data heterogeneity that hinders FL convergence, DroneFL introduces a position\-invariant model architecture with altitude\-based adaptive instance normalization. Finally, we fuse predictions from multiple UAVs in the cloud and generate optimal trajectories that balance target prediction accuracy and overall tracking performance. Our results show that DroneFL reduces prediction error by 6%\-83% and tracking distance by 0.4%\-4.6% compared to a distributed non\-FL framework. In terms of efficiency, DroneFL runs in real time on a Raspberry Pi 5 and has on average just 1.56 KBps data rate to the cloud.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2509.21523v1)

---


## Real\-Time Object Detection Meets DINOv3 / 

发布日期：2025-09-25

作者：Shihua Huang

摘要：Benefiting from the simplicity and effectiveness of Dense O2O and MAL, DEIM has become the mainstream training framework for real\-time DETRs, significantly outperforming the YOLO series. In this work, we extend it with DINOv3 features, resulting in DEIMv2. DEIMv2 spans eight model sizes from X to Atto, covering GPU, edge, and mobile deployment. For the X, L, M, and S variants, we adopt DINOv3\-pretrained or distilled backbones and introduce a Spatial Tuning Adapter \(STA\), which efficiently converts DINOv3's single\-scale output into multi\-scale features and complements strong semantics with fine\-grained details to enhance detection. For ultra\-lightweight models \(Nano, Pico, Femto, and Atto\), we employ HGNetv2 with depth and width pruning to meet strict resource budgets. Together with a simplified decoder and an upgraded Dense O2O, this unified design enables DEIMv2 to achieve a superior performance\-cost trade\-off across diverse scenarios, establishing new state\-of\-the\-art results. Notably, our largest model, DEIMv2\-X, achieves 57.8 AP with only 50.3 million parameters, surpassing prior X\-scale models that require over 60 million parameters for just 56.5 AP. On the compact side, DEIMv2\-S is the first sub\-10 million model \(9.71 million\) to exceed the 50 AP milestone on COCO, reaching 50.9 AP. Even the ultra\-lightweight DEIMv2\-Pico, with just 1.5 million parameters, delivers 38.5 AP, matching YOLOv10\-Nano \(2.3 million\) with around 50 percent fewer parameters. Our code and pre\-trained models are available at https://github.com/Intellindust\-AI\-Lab/DEIMv2

中文摘要：


代码链接：https://github.com/Intellindust-AI-Lab/DEIMv2

论文链接：[阅读更多](http://arxiv.org/abs/2509.20787v2)

---


## Building Information Models to Robot\-Ready Site Digital Twins \(BIM2RDT\): An Agentic AI Safety\-First Framework / 

发布日期：2025-09-25

作者：Reza Akhavian

摘要：The adoption of cyber\-physical systems and jobsite intelligence that connects design models, real\-time site sensing, and autonomous field operations can dramatically enhance digital management in the construction industry. This paper introduces BIM2RDT \(Building Information Models to Robot\-Ready Site Digital Twins\), an agentic artificial intelligence \(AI\) framework designed to transform static Building Information Modeling \(BIM\) into dynamic, robot\-ready digital twins \(DTs\) that prioritize safety during execution. The framework bridges the gap between pre\-existing BIM data and real\-time site conditions by integrating three key data streams: geometric and semantic information from BIM models, activity data from IoT sensor networks, and visual\-spatial data collected by robots during site traversal. The methodology introduces Semantic\-Gravity ICP \(SG\-ICP\), a point cloud registration algorithm that leverages large language model \(LLM\) reasoning. Unlike traditional methods, SG\-ICP utilizes an LLM to infer object\-specific, plausible orientation priors based on BIM semantics, improving alignment accuracy by avoiding convergence on local minima. This creates a feedback loop where robot\-collected data updates the DT, which in turn optimizes paths for missions. The framework employs YOLOE object detection and Shi\-Tomasi corner detection to identify and track construction elements while using BIM geometry as a priori maps. The framework also integrates real\-time Hand\-Arm Vibration \(HAV\) monitoring, mapping sensor\-detected safety events to the digital twin using IFC standards for intervention. Experiments demonstrate SG\-ICP's superiority over standard ICP, achieving RMSE reductions of 64.3%\-\-88.3% in alignment across scenarios with occluded features, ensuring plausible orientations. HAV integration triggers warnings upon exceeding exposure limits, enhancing compliance with ISO 5349\-1.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2509.20705v1)

---

