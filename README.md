# 每日从arXiv中获取最新YOLO相关论文


## Implementation of a Near\-Realtime Recording and Reporting System of Solar Radio Bursts / 

发布日期：2026-03-26

作者：Peijin Zhang

摘要：Strong solar activity is often accompanied by a variety of radio bursts. These bursts are valuable diagnostics of coronal and heliospheric processes and also have potential applications in space weather monitoring and forecasting. However, space weather applications require low\-latency, high\-sensitivity radio burst recording and reporting capabilities, which have remained limited. In this work, we present the development of a near\-realtime radio burst recording and reporting system using the Owens Valley Radio Observatory Long Wavelength Array. The system directly clips data from a realtime buffer and streams them as a live radio dynamic spectrogram. These spectrograms are then processed by a deep\-learning\-based burst identification module for type III radio bursts. The identifier is based on a YOLO \(You Only Look Once\) architecture and is trained on synthetic type III radio bursts generated using a physics\-based model to achieve accurate and robust detection. This system enables continuous realtime radio spectrum streaming and automatic reporting of type III radio bursts within approximately 10 seconds of their occurrence.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2603.25446v1)

---


## SDD\-YOLO: A Small\-Target Detection Framework for Ground\-to\-Air Anti\-UAV Surveillance with Edge\-Efficient Deployment / 

发布日期：2026-03-26

作者：Pengyu Chen

摘要：Detecting small unmanned aerial vehicles \(UAVs\) from a ground\-to\-air \(G2A\) perspective presents significant challenges, including extremely low pixel occupancy, cluttered aerial backgrounds, and strict real\-time constraints. Existing YOLO\-based detectors are primarily optimized for general object detection and often lack adequate feature resolution for sub\-pixel targets, while introducing complexities during deployment. In this paper, we propose SDD\-YOLO, a small\-target detection framework tailored for G2A anti\-UAV surveillance. To capture fine\-grained spatial details critical for micro\-targets, SDD\-YOLO introduces a P2 high\-resolution detection head operating at 4 times downsampling. Furthermore, we integrate the recent architectural advancements from YOLO26, including a DFL\-free, NMS\-free architecture for streamlined inference, and the MuSGD hybrid training strategy with ProgLoss and STAL, which substantially mitigates gradient oscillation on sparse small\-target signals. To support our evaluation, we construct DroneSOD\-30K, a large\-scale G2A dataset comprising approximately 30,000 annotated images covering diverse meteorological conditions. Experiments demonstrate that SDD\-YOLO\-n achieves a mAP@0.5 of 86.0% on DroneSOD\-30K, surpassing the YOLOv5n baseline by 7.8 percentage points. Extensive inference analysis shows our model attains 226 FPS on an NVIDIA RTX 5090 and 35 FPS on an Intel Xeon CPU, demonstrating exceptional efficiency for future edge deployment.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2603.25218v1)

---


## Concept\-based explanations of Segmentation and Detection models in Natural Disaster Management / 

发布日期：2026-03-24

作者：Samar Heydari

摘要：Deep learning models for flood and wildfire segmentation and object detection enable precise, real\-time disaster localization when deployed on embedded drone platforms. However, in natural disaster management, the lack of transparency in their decision\-making process hinders human trust required for emergency response. To address this, we present an explainability framework for understanding flood segmentation and car detection predictions on the widely used PIDNet and YOLO architectures. More specifically, we introduce a novel redistribution strategy that extends Layer\-wise Relevance Propagation \(LRP\) explanations for sigmoid\-gated element\-wise fusion layers. This extension allows LRP relevances to flow through the fusion modules of PIDNet, covering the entire computation graph back to the input image. Furthermore, we apply Prototypical Concept\-based Explanations \(PCX\) to provide both local and global explanations at the concept level, revealing which learned features drive the segmentation and detection of specific disaster semantic classes. Experiments on a publicly available flood dataset show that our framework provides reliable and interpretable explanations while maintaining near real\-time inference capabilities, rendering it suitable for deployment on resource\-constrained platforms, such as Unmanned Aerial Vehicles \(UAVs\).

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2603.23020v1)

---


## MS\-CustomNet: Controllable Multi\-Subject Customization with Hierarchical Relational Semantics / 

发布日期：2026-03-22

作者：Pengxiang Cai

摘要：Diffusion\-based text\-to\-image generation has advanced significantly, yet customizing scenes with multiple distinct subjects while maintaining fine\-grained control over their interactions remains challenging. Existing methods often struggle to provide explicit user\-defined control over the compositional structure and precise spatial relationships between subjects. To address this, we introduce MS\-CustomNet, a novel framework for multi\-subject customization. MS\-CustomNet allows zero\-shot integration of multiple user\-provided objects and, crucially, empowers users to explicitly define these hierarchical arrangements and spatial placements within the generated image. Our approach ensures individual subject identity preservation while learning and enacting these user\-specified inter\-subject compositions. We also present the MSI dataset, derived from COCO, to facilitate training on such complex multi\-subject compositions. MS\-CustomNet offers enhanced, fine\-grained control over multi\-subject image generation. Our method achieves a DINO\-I score of 0.61 for identity preservation and a YOLO\-L score of 0.94 for positional control in multi\-subject customization tasks, demonstrating its superior capability in generating high\-fidelity images with precise, user\-directed multi\-subject compositions and spatial control.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2603.21136v1)

---


## Anatomical Prior\-Driven Framework for Autonomous Robotic Cardiac Ultrasound Standard View Acquisition / 

发布日期：2026-03-22

作者：Zhiyan Cao

摘要：Cardiac ultrasound diagnosis is critical for cardiovascular disease assessment, but acquiring standard views remains highly operator\-dependent. Existing medical segmentation models often yield anatomically inconsistent results in images with poor textural differentiation between distinct feature classes, while autonomous probe adjustment methods either rely on simplistic heuristic rules or black\-box learning. To address these issues, our study proposed an anatomical prior \(AP\)\-driven framework integrating cardiac structure segmentation and autonomous probe adjustment for standard view acquisition. A YOLO\-based multi\-class segmentation model augmented by a spatial\-relation graph \(SRG\) module is designed to embed AP into the feature pyramid. Quantifiable anatomical features of standard views are extracted. Their priors are fitted to Gaussian distributions to construct probabilistic APs. The probe adjustment process of robotic ultrasound scanning is formalized as a reinforcement learning \(RL\) problem, with the RL state built from real\-time anatomical features and the reward reflecting the AP matching. Experiments validate the efficacy of the framework. The SRG\-YOLOv11s improves mAP50 by 11.3% and mIoU by 6.8% on the Special Case dataset, while the RL agent achieves a 92.5% success rate in simulation and 86.7% in phantom experiments.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2603.21134v1)

---

