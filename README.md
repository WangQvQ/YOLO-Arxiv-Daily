# 每日从arXiv中获取最新YOLO相关论文


## Towards Self\-Improvement of Diffusion Models via Group Preference Optimization / 

发布日期：2025-05-16

作者：Renjie Chen

摘要：Aligning text\-to\-image \(T2I\) diffusion models with Direct Preference Optimization \(DPO\) has shown notable improvements in generation quality. However, applying DPO to T2I faces two challenges: the sensitivity of DPO to preference pairs and the labor\-intensive process of collecting and annotating high\-quality data. In this work, we demonstrate that preference pairs with marginal differences can degrade DPO performance. Since DPO relies exclusively on relative ranking while disregarding the absolute difference of pairs, it may misclassify losing samples as wins, or vice versa. We empirically show that extending the DPO from pairwise to groupwise and incorporating reward standardization for reweighting leads to performance gains without explicit data selection. Furthermore, we propose Group Preference Optimization \(GPO\), an effective self\-improvement method that enhances performance by leveraging the model's own capabilities without requiring external data. Extensive experiments demonstrate that GPO is effective across various diffusion models and tasks. Specifically, combining with widely used computer vision models, such as YOLO and OCR, the GPO improves the accurate counting and text rendering capabilities of the Stable Diffusion 3.5 Medium by 20 percentage points. Notably, as a plug\-and\-play method, no extra overhead is introduced during inference.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2505.11070v1)

---


## A High\-Performance Thermal Infrared Object Detection Framework with Centralized Regulation / 

发布日期：2025-05-16

作者：Jinke Li

摘要：Thermal Infrared \(TIR\) technology involves the use of sensors to detect and measure infrared radiation emitted by objects, and it is widely utilized across a broad spectrum of applications. The advancements in object detection methods utilizing TIR images have sparked significant research interest. However, most traditional methods lack the capability to effectively extract and fuse local\-global information, which is crucial for TIR\-domain feature attention. In this study, we present a novel and efficient thermal infrared object detection framework, known as CRT\-YOLO, that is based on centralized feature regulation, enabling the establishment of global\-range interaction on TIR information. Our proposed model integrates efficient multi\-scale attention \(EMA\) modules, which adeptly capture long\-range dependencies while incurring minimal computational overhead. Additionally, it leverages the Centralized Feature Pyramid \(CFP\) network, which offers global regulation of TIR features. Extensive experiments conducted on two benchmark datasets demonstrate that our CRT\-YOLO model significantly outperforms conventional methods for TIR image object detection. Furthermore, the ablation study provides compelling evidence of the effectiveness of our proposed modules, reinforcing the potential impact of our approach on advancing the field of thermal infrared object detection.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2505.10825v1)

---


## Geofenced Unmanned Aerial Robotic Defender for Deer Detection and Deterrence \(GUARD\) / 

发布日期：2025-05-16

作者：Ebasa Temesgen

摘要：Wildlife\-induced crop damage, particularly from deer, threatens agricultural productivity. Traditional deterrence methods often fall short in scalability, responsiveness, and adaptability to diverse farmland environments. This paper presents an integrated unmanned aerial vehicle \(UAV\) system designed for autonomous wildlife deterrence, developed as part of the Farm Robotics Challenge. Our system combines a YOLO\-based real\-time computer vision module for deer detection, an energy\-efficient coverage path planning algorithm for efficient field monitoring, and an autonomous charging station for continuous operation of the UAV. In collaboration with a local Minnesota farmer, the system is tailored to address practical constraints such as terrain, infrastructure limitations, and animal behavior. The solution is evaluated through a combination of simulation and field testing, demonstrating robust detection accuracy, efficient coverage, and extended operational time. The results highlight the feasibility and effectiveness of drone\-based wildlife deterrence in precision agriculture, offering a scalable framework for future deployment and extension.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2505.10770v1)

---


## Revisiting Adversarial Perception Attacks and Defense Methods on Autonomous Driving Systems / 

发布日期：2025-05-14

作者：Cheng Chen

摘要：Autonomous driving systems \(ADS\) increasingly rely on deep learning\-based perception models, which remain vulnerable to adversarial attacks. In this paper, we revisit adversarial attacks and defense methods, focusing on road sign recognition and lead object detection and prediction \(e.g., relative distance\). Using a Level\-2 production ADS, OpenPilot by Comma.ai, and the widely adopted YOLO model, we systematically examine the impact of adversarial perturbations and assess defense techniques, including adversarial training, image processing, contrastive learning, and diffusion models. Our experiments highlight both the strengths and limitations of these methods in mitigating complex attacks. Through targeted evaluations of model robustness, we aim to provide deeper insights into the vulnerabilities of ADS perception systems and contribute guidance for developing more resilient defense strategies.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2505.11532v1)

---


## Thermal Detection of People with Mobility Restrictions for Barrier Reduction at Traffic Lights Controlled Intersections / 

发布日期：2025-05-13

作者：Xiao Ni

摘要：Rapid advances in deep learning for computer vision have driven the adoption of RGB camera\-based adaptive traffic light systems to improve traffic safety and pedestrian comfort. However, these systems often overlook the needs of people with mobility restrictions. Moreover, the use of RGB cameras presents significant challenges, including limited detection performance under adverse weather or low\-visibility conditions, as well as heightened privacy concerns. To address these issues, we propose a fully automated, thermal detector\-based traffic light system that dynamically adjusts signal durations for individuals with walking impairments or mobility burden and triggers the auditory signal for visually impaired individuals, thereby advancing towards barrier\-free intersection for all users. To this end, we build the thermal dataset for people with mobility restrictions \(TD4PWMR\), designed to capture diverse pedestrian scenarios, particularly focusing on individuals with mobility aids or mobility burden under varying environmental conditions, such as different lighting, weather, and crowded urban settings. While thermal imaging offers advantages in terms of privacy and robustness to adverse conditions, it also introduces inherent hurdles for object detection due to its lack of color and fine texture details and generally lower resolution of thermal images. To overcome these limitations, we develop YOLO\-Thermal, a novel variant of the YOLO architecture that integrates advanced feature extraction and attention mechanisms for enhanced detection accuracy and robustness in thermal imaging. Experiments demonstrate that the proposed thermal detector outperforms existing detectors, while the proposed traffic light system effectively enhances barrier\-free intersection. The source codes and dataset are available at https://github.com/leon2014dresden/YOLO\-THERMAL.

中文摘要：


代码链接：https://github.com/leon2014dresden/YOLO-THERMAL.

论文链接：[阅读更多](http://arxiv.org/abs/2505.08568v2)

---

