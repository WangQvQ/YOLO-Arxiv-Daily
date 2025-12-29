# 每日从arXiv中获取最新YOLO相关论文


## Comparative Analysis of Deep Learning Models for Perception in Autonomous Vehicles / 

发布日期：2025-12-25

作者：Jalal Khan

摘要：Recently, a plethora of machine learning \(ML\) and deep learning \(DL\) algorithms have been proposed to achieve the efficiency, safety, and reliability of autonomous vehicles \(AVs\). The AVs use a perception system to detect, localize, and identify other vehicles, pedestrians, and road signs to perform safe navigation and decision\-making. In this paper, we compare the performance of DL models, including YOLO\-NAS and YOLOv8, for a detection\-based perception task. We capture a custom dataset and experiment with both DL models using our custom dataset. Our analysis reveals that the YOLOv8s model saves 75% of training time compared to the YOLO\-NAS model. In addition, the YOLOv8s model \(83%\) outperforms the YOLO\-NAS model \(81%\) when the target is to achieve the highest object detection accuracy. These comparative analyses of these new emerging DL models will allow the relevant research community to understand the models' performance under real\-world use case scenarios.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2512.21673v1)

---


## Lightweight framework for underground pipeline recognition and spatial localization based on multi\-view 2D GPR images / 

发布日期：2025-12-24

作者：Haotian Lv

摘要：To address the issues of weak correlation between multi\-view features, low recognition accuracy of small\-scale targets, and insufficient robustness in complex scenarios in underground pipeline detection using 3D GPR, this paper proposes a 3D pipeline intelligent detection framework. First, based on a B/C/D\-Scan three\-view joint analysis strategy, a three\-dimensional pipeline three\-view feature evaluation method is established by cross\-validating forward simulation results obtained using FDTD methods with actual measurement data. Second, the DCO\-YOLO framework is proposed, which integrates DySample, CGLU, and OutlookAttention cross\-dimensional correlation mechanisms into the original YOLOv11 algorithm, significantly improving the small\-scale pipeline edge feature extraction capability. Furthermore, a 3D\-DIoU spatial feature matching algorithm is proposed, which integrates three\-dimensional geometric constraints and center distance penalty terms to achieve automated association of multi\-view annotations. The three\-view fusion strategy resolves inherent ambiguities in single\-view detection. Experiments based on real urban underground pipeline data show that the proposed method achieves accuracy, recall, and mean average precision of 96.2%, 93.3%, and 96.7%, respectively, in complex multi\-pipeline scenarios, which are 2.0%, 2.1%, and 0.9% higher than the baseline model. Ablation experiments validated the synergistic optimization effect of the dynamic feature enhancement module and Grad\-CAM\+\+ heatmap visualization demonstrated that the improved model significantly enhanced its ability to focus on pipeline geometric features. This study integrates deep learning optimization strategies with the physical characteristics of 3D GPR, offering an efficient and reliable novel technical framework for the intelligent recognition and localization of underground pipelines.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2512.20866v1)

---


## Multi\-temporal Adaptive Red\-Green\-Blue and Long\-Wave Infrared Fusion for You Only Look Once\-Based Landmine Detection from Unmanned Aerial Systems / 

发布日期：2025-12-23

作者：James E. Gallagher

摘要：Landmines remain a persistent humanitarian threat, with 110 million actively deployed mines across 60 countries, claiming 26,000 casualties annually. This research evaluates adaptive Red\-Green\-Blue \(RGB\) and Long\-Wave Infrared \(LWIR\) fusion for Unmanned Aerial Systems \(UAS\)\-based detection of surface\-laid landmines, leveraging the thermal contrast between the ordnance and the surrounding soil to enhance feature extraction. Using You Only Look Once \(YOLO\) architectures \(v8, v10, v11\) across 114 test images, generating 35,640 model\-condition evaluations, YOLOv11 achieved optimal performance \(86.8% mAP\), with 10 to 30% thermal fusion at 5 to 10m altitude identified as the optimal detection parameters. A complementary architectural comparison revealed that while RF\-DETR achieved the highest accuracy \(69.2% mAP\), followed by Faster R\-CNN \(67.6%\), YOLOv11 \(64.2%\), and RetinaNet \(50.2%\), YOLOv11 trained 17.7 times faster than the transformer\-based RF\-DETR \(41 minutes versus 12 hours\), presenting a critical accuracy\-efficiency tradeoff for operational deployment. Aggregated multi\-temporal training datasets outperformed season\-specific approaches by 1.8 to 9.6%, suggesting that models benefit from exposure to diverse thermal conditions. Anti\-Tank \(AT\) mines achieved 61.9% detection accuracy, compared with 19.2% for Anti\-Personnel \(AP\) mines, reflecting both the size differential and thermal\-mass differences between these ordnance classes. As this research examined surface\-laid mines where thermal contrast is maximized, future research should quantify thermal contrast effects for mines buried at varying depths across heterogeneous soil types.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2512.20487v1)

---


## Drift\-Corrected Monocular VIO and Perception\-Aware Planning for Autonomous Drone Racing / 

发布日期：2025-12-23

作者：Maulana Bisyir Azhari

摘要：The Abu Dhabi Autonomous Racing League\(A2RL\) x Drone Champions League competition\(DCL\) requires teams to perform high\-speed autonomous drone racing using only a single camera and a low\-quality inertial measurement unit \-\- a minimal sensor set that mirrors expert human drone racing pilots. This sensor limitation makes the system susceptible to drift from Visual\-Inertial Odometry \(VIO\), particularly during long and fast flights with aggressive maneuvers. This paper presents the system developed for the championship, which achieved a competitive performance. Our approach corrected VIO drift by fusing its output with global position measurements derived from a YOLO\-based gate detector using a Kalman filter. A perception\-aware planner generated trajectories that balance speed with the need to keep gates visible for the perception system. The system demonstrated high performance, securing podium finishes across multiple categories: third place in the AI Grand Challenge with top speed of 43.2 km/h, second place in the AI Drag Race with over 59 km/h, and second place in the AI Multi\-Drone Race. We detail the complete architecture and present a performance analysis based on experimental data from the competition, contributing our insights on building a successful system for monocular vision\-based autonomous drone flight.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2512.20475v1)

---


## Retrieving Objects from 3D Scenes with Box\-Guided Open\-Vocabulary Instance Segmentation / 

发布日期：2025-12-22

作者：Khanh Nguyen

摘要：Locating and retrieving objects from scene\-level point clouds is a challenging problem with broad applications in robotics and augmented reality. This task is commonly formulated as open\-vocabulary 3D instance segmentation. Although recent methods demonstrate strong performance, they depend heavily on SAM and CLIP to generate and classify 3D instance masks from images accompanying the point cloud, leading to substantial computational overhead and slow processing that limit their deployment in real\-world settings. Open\-YOLO 3D alleviates this issue by using a real\-time 2D detector to classify class\-agnostic masks produced directly from the point cloud by a pretrained 3D segmenter, eliminating the need for SAM and CLIP and significantly reducing inference time. However, Open\-YOLO 3D often fails to generalize to object categories that appear infrequently in the 3D training data. In this paper, we propose a method that generates 3D instance masks for novel objects from RGB images guided by a 2D open\-vocabulary detector. Our approach inherits the 2D detector's ability to recognize novel objects while maintaining efficient classification, enabling fast and accurate retrieval of rare instances from open\-ended text queries. Our code will be made available at https://github.com/ndkhanh360/BoxOVIS.

中文摘要：


代码链接：https://github.com/ndkhanh360/BoxOVIS.

论文链接：[阅读更多](http://arxiv.org/abs/2512.19088v1)

---

