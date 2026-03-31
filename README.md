# 每日从arXiv中获取最新YOLO相关论文


## Sim\-to\-Real Fruit Detection Using Synthetic Data: Quantitative Evaluation and Embedded Deployment with Isaac Sim / 基于合成数据的模拟到真实水果检测：Isaac Sim的定量评估和嵌入式部署

发布日期：2026-03-30

作者：Martina Hutter\-Mironovova

摘要：This study investigates the effectiveness of synthetic data for sim\-to\-real transfer in object detection under constrained data conditions and embedded deployment requirements. Synthetic datasets were generated in NVIDIA Isaac Sim and combined with limited real\-world fruit images to train YOLO\-based detection models under real\-only, synthetic\-only, and hybrid regimes. Performance was evaluated on two test datasets: an in\-domain dataset with conditions matching the training data and a domain shift dataset containing real fruit and different background conditions. Results show that models trained exclusively on real data achieve the highest accuracy, while synthetic\-only models exhibit reduced performance due to a domain gap. Hybrid training strategies significantly improve performance compared to synthetic\-only approaches and achieve results close to real\-only training while reducing the need for manual annotation. Under domain shift conditions, all models show performance degradation, with hybrid models providing improved robustness. The trained models were successfully deployed on a Jetson Orin NX using TensorRT optimization, achieving real\-time inference performance. The findings highlight that synthetic data is most effective when used in combination with real data and that deployment constraints must be considered alongside detection accuracy.

中文摘要：本研究调查了在受限数据条件和嵌入式部署要求下，合成数据在目标检测中从模拟到真实传输的有效性。合成数据集是在NVIDIA Isaac Sim中生成的，并与有限的真实世界水果图像相结合，在仅真实、仅合成和混合状态下训练基于YOLO的检测模型。在两个测试数据集上评估了性能：一个是条件与训练数据匹配的域内数据集，另一个是包含真实水果和不同背景条件的域移位数据集。结果表明，仅在真实数据上训练的模型达到了最高的精度，而仅合成的模型由于域差距而表现出性能下降。与纯合成方法相比，混合训练策略显著提高了性能，并实现了接近纯真实训练的结果，同时减少了手动注释的需要。在域偏移条件下，所有模型都表现出性能下降，混合模型提供了改进的鲁棒性。使用TensorRT优化，训练好的模型成功部署在Jetson Orin NX上，实现了实时推理性能。研究结果强调，合成数据与真实数据结合使用时最有效，部署约束必须与检测精度一起考虑。


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2603.28670v1)

---


## Human\-Centric Perception for Child Sexual Abuse Imagery / 儿童性虐待意象的人类中心感知

发布日期：2026-03-28

作者：Camila Laranjeira

摘要：Law enforcement agencies and non\-gonvernmental organizations handling reports of Child Sexual Abuse Imagery \(CSAI\) are overwhelmed by large volumes of data, requiring the aid of automation tools. However, defining sexual abuse in images of children is inherently challenging, encompassing sexually explicit activities and hints of sexuality conveyed by the individual's pose, or their attire. CSAI classification methods often rely on black\-box approaches, targeting broad and abstract concepts such as pornography. Thus, our work is an in\-depth exploration of tasks from the literature on Human\-Centric Perception, across the domains of safe images, adult pornography, and CSAI, focusing on targets that enable more objective and explainable pipelines for CSAI classification in the future. We introduce the Body\-Keypoint\-Part Dataset \(BKPD\), gathering images of people from varying age groups and sexual explicitness to approximate the domain of CSAI, along with manually curated hierarchically structured labels for skeletal keypoints and bounding boxes for person and body parts, including head, chest, hip, and hands. We propose two methods, namely BKP\-Association and YOLO\-BKP, for simultaneous pose estimation and detection, with targets associated per individual for a comprehensive decomposed representation of each person. Our methods are benchmarked on COCO\-Keypoints and COCO\-HumanParts, as well as our human\-centric dataset, achieving competitive results with models that jointly perform all tasks. Cross\-domain ablation studies on BKPD and a case study on RCPD highlight the challenges posed by sexually explicit domains. Our study addresses previously unexplored targets in the CSAI domain, paving the way for novel research opportunities.

中文摘要：处理儿童性虐待图像（CSAI）报告的执法机构和非政府组织被大量数据淹没，需要自动化工具的帮助。然而，在儿童图像中定义性虐待本质上具有挑战性，包括露骨的性活动和通过个人姿势或服装传达的性暗示。CSAI分类方法通常依赖于黑盒方法，针对色情等广泛而抽象的概念。因此，我们的工作是对以人为中心的感知文献中的任务进行深入探索，涵盖安全图像、成人色情和CSAI领域，重点关注未来能够为CSAI分类提供更客观和可解释的管道的目标。我们引入了身体关键点部分数据集（BKPD），收集来自不同年龄组和性明确性的人的图像，以近似CSAI的领域，以及手动策划的骨骼关键点分层结构标签和人和身体部分的边界框，包括头部、胸部、臀部和手。我们提出了两种方法，即BKP关联和YOLO-BKP，用于同时进行姿态估计和检测，每个人都有相关的目标，以实现每个人的全面分解表示。我们的方法以COCO Keypoints和COCO HumanParts以及我们以人为中心的数据集为基准，通过共同执行所有任务的模型取得了有竞争力的结果。关于BKPD的跨域消融研究和关于RCPD的案例研究突出了性显性领域带来的挑战。我们的研究解决了CSAI领域以前未探索的目标，为新的研究机会铺平了道路。


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2603.27290v1)

---


## Autonomous overtaking trajectory optimization using reinforcement learning and opponent pose estimation / 基于强化学习和对手姿态估计的自主超车轨迹优化

发布日期：2026-03-28

作者：Matej Rene Cihlar

摘要：Vehicle overtaking is one of the most complex driving maneuvers for autonomous vehicles. To achieve optimal autonomous overtaking, driving systems rely on multiple sensors that enable safe trajectory optimization and overtaking efficiency. This paper presents a reinforcement learning mechanism for multi\-agent autonomous racing environments, enabling overtaking trajectory optimization, based on LiDAR and depth image data. The developed reinforcement learning agent uses pre\-generated raceline data and sensor inputs to compute the steering angle and linear velocity for optimal overtaking. The system uses LiDAR with a 2D detection algorithm and a depth camera with YOLO\-based object detection to identify the vehicle to be overtaken and its pose. The LiDAR and the depth camera detection data are fused using a UKF for improved opponent pose estimation and trajectory optimization for overtaking in racing scenarios. The results show that the proposed algorithm successfully performs overtaking maneuvers in both simulation and real\-world experiments, with pose estimation RMSE of \(0.0816, 0.0531\) m in \(x, y\).

中文摘要：超车是自动驾驶汽车最复杂的驾驶操作之一。为了实现最佳的自动超车，驾驶系统依赖于多个传感器，这些传感器能够实现安全的轨迹优化和超车效率。本文提出了一种基于激光雷达和深度图像数据的多智能体自主赛车环境的强化学习机制，实现了超车轨迹优化。开发的强化学习代理使用预先生成的赛道数据和传感器输入来计算最佳超车的转向角和线速度。该系统使用具有2D检测算法的激光雷达和具有基于YOLO的物体检测的深度相机来识别要超车的车辆及其姿态。使用UKF将LiDAR和深度相机检测数据融合在一起，以改进对手姿态估计和赛道超车轨迹优化。结果表明，所提出的算法在仿真和真实世界的实验中都成功地执行了超车机动，在（x，y）中的姿态估计RMSE为（0.0816，0.0531）m。


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2603.27207v1)

---


## YOLO Object Detectors for Robotics \-\- a Comparative Study / YOLO机器人目标探测器的比较研究

发布日期：2026-03-27

作者：Patryk Niżeniec

摘要：YOLO object detectors recently became a key component of vision systems in many domains. The family of available YOLO models consists of multiple versions, each in various variants. The research reported in this paper aims to validate the applicability of members of this family to detect objects located within the robot workspace. In our experiments, we used our custom dataset and the COCO2017 dataset. To test the robustness of investigated detectors, the images of these datasets were subject to distortions. The results of our experiments, including variations of training/testing configurations and models, may support the choice of the appropriate YOLO version for robotic vision tasks.

中文摘要：YOLO物体探测器最近成为许多领域视觉系统的关键组成部分。可用的YOLO型号系列由多个版本组成，每个版本都有各种变体。本文报告的研究旨在验证该家族成员检测机器人工作空间内物体的适用性。在我们的实验中，我们使用了自定义数据集和COCO2017数据集。为了测试所研究探测器的鲁棒性，这些数据集的图像会发生失真。我们的实验结果，包括训练/测试配置和模型的变化，可能支持为机器人视觉任务选择合适的YOLO版本。


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2603.27029v1)

---


## Implementation of a Near\-Realtime Recording and Reporting System of Solar Radio Bursts / 太阳射电爆发近实时记录与报告系统的实现

发布日期：2026-03-26

作者：Peijin Zhang

摘要：Strong solar activity is often accompanied by a variety of radio bursts. These bursts are valuable diagnostics of coronal and heliospheric processes and also have potential applications in space weather monitoring and forecasting. However, space weather applications require low\-latency, high\-sensitivity radio burst recording and reporting capabilities, which have remained limited. In this work, we present the development of a near\-realtime radio burst recording and reporting system using the Owens Valley Radio Observatory Long Wavelength Array. The system directly clips data from a realtime buffer and streams them as a live radio dynamic spectrogram. These spectrograms are then processed by a deep\-learning\-based burst identification module for type III radio bursts. The identifier is based on a YOLO \(You Only Look Once\) architecture and is trained on synthetic type III radio bursts generated using a physics\-based model to achieve accurate and robust detection. This system enables continuous realtime radio spectrum streaming and automatic reporting of type III radio bursts within approximately 10 seconds of their occurrence.

中文摘要：强烈的太阳活动通常伴随着各种无线电爆发。这些爆发是对日冕和日球层过程的有价值的诊断，在空间天气监测和预报方面也有潜在的应用。然而，空间气象应用需要低延迟、高灵敏度的无线电突发记录和报告能力，而这些能力仍然有限。在这项工作中，我们介绍了使用欧文斯谷射电天文台长波阵列开发的近实时无线电突发记录和报告系统。该系统直接从实时缓冲区中剪辑数据，并将其作为实时无线电动态频谱图进行流式传输。然后，这些频谱图由基于深度学习的III型无线电脉冲串识别模块进行处理。该标识符基于YOLO（You Only Look Once）架构，并在使用基于物理的模型生成的合成III型无线电脉冲串上进行训练，以实现准确和稳健的检测。该系统能够实现连续的实时无线电频谱流，并在III型无线电突发发生后约10秒内自动报告。


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2603.25446v1)

---

