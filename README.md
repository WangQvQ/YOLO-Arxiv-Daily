# 每日从arXiv中获取最新YOLO相关论文


## Automated Defect Detection for Mass\-Produced Electronic Components Based on YOLO Object Detection Models / 

发布日期：2025-10-02

作者：Wei\-Lung Mao

摘要：Since the defect detection of conventional industry components is time\-consuming and labor\-intensive, it leads to a significant burden on quality inspection personnel and makes it difficult to manage product quality. In this paper, we propose an automated defect detection system for the dual in\-line package \(DIP\) that is widely used in industry, using digital camera optics and a deep learning \(DL\)\-based model. The two most common defect categories of DIP are examined: \(1\) surface defects, and \(2\) pin\-leg defects. However, the lack of defective component images leads to a challenge for detection tasks. To solve this problem, the ConSinGAN is used to generate a suitable\-sized dataset for training and testing. Four varieties of the YOLO model are investigated \(v3, v4, v7, and v9\), both in isolation and with the ConSinGAN augmentation. The proposed YOLOv7 with ConSinGAN is superior to the other YOLO versions in accuracy of 95.50%, detection time of 285 ms, and is far superior to threshold\-based approaches. In addition, the supervisory control and data acquisition \(SCADA\) system is developed, and the associated sensor architecture is described. The proposed automated defect detection can be easily established with numerous types of defects or insufficient defect data.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2510.01914v1)

---


## AdvEvo\-MARL: Shaping Internalized Safety through Adversarial Co\-Evolution in Multi\-Agent Reinforcement Learning / 

发布日期：2025-10-02

作者：Zhenyu Pan

摘要：LLM\-based multi\-agent systems excel at planning, tool use, and role coordination, but their openness and interaction complexity also expose them to jailbreak, prompt\-injection, and adversarial collaboration. Existing defenses fall into two lines: \(i\) self\-verification that asks each agent to pre\-filter unsafe instructions before execution, and \(ii\) external guard modules that police behaviors. The former often underperforms because a standalone agent lacks sufficient capacity to detect cross\-agent unsafe chains and delegation\-induced risks; the latter increases system overhead and creates a single\-point\-of\-failure\-once compromised, system\-wide safety collapses, and adding more guards worsens cost and complexity. To solve these challenges, we propose AdvEvo\-MARL, a co\-evolutionary multi\-agent reinforcement learning framework that internalizes safety into task agents. Rather than relying on external guards, AdvEvo\-MARL jointly optimizes attackers \(which synthesize evolving jailbreak prompts\) and defenders \(task agents trained to both accomplish their duties and resist attacks\) in adversarial learning environments. To stabilize learning and foster cooperation, we introduce a public baseline for advantage estimation: agents within the same functional group share a group\-level mean\-return baseline, enabling lower\-variance updates and stronger intra\-group coordination. Across representative attack scenarios, AdvEvo\-MARL consistently keeps attack\-success rate \(ASR\) below 20%, whereas baselines reach up to 38.33%, while preserving\-and sometimes improving\-task accuracy \(up to \+3.67% on reasoning tasks\). These results show that safety and utility can be jointly improved without relying on extra guard agents or added system overhead.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2510.01586v1)

---


## Forestpest\-YOLO: A High\-Performance Detection Framework for Small Forestry Pests / 

发布日期：2025-10-01

作者：Aoduo Li

摘要：Detecting agricultural pests in complex forestry environments using remote sensing imagery is fundamental for ecological preservation, yet it is severely hampered by practical challenges. Targets are often minuscule, heavily occluded, and visually similar to the cluttered background, causing conventional object detection models to falter due to the loss of fine\-grained features and an inability to handle extreme data imbalance. To overcome these obstacles, this paper introduces Forestpest\-YOLO, a detection framework meticulously optimized for the nuances of forestry remote sensing. Building upon the YOLOv8 architecture, our framework introduces a synergistic trio of innovations. We first integrate a lossless downsampling module, SPD\-Conv, to ensure that critical high\-resolution details of small targets are preserved throughout the network. This is complemented by a novel cross\-stage feature fusion block, CSPOK, which dynamically enhances multi\-scale feature representation while suppressing background noise. Finally, we employ VarifocalLoss to refine the training objective, compelling the model to focus on high\-quality and hard\-to\-classify samples. Extensive experiments on our challenging, self\-constructed ForestPest dataset demonstrate that Forestpest\-YOLO achieves state\-of\-the\-art performance, showing marked improvements in detecting small, occluded pests and significantly outperforming established baseline models.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2510.00547v1)

---


## VLOD\-TTA: Test\-Time Adaptation of Vision\-Language Object Detectors / 

发布日期：2025-10-01

作者：Atif Belal

摘要：Vision\-language object detectors \(VLODs\) such as YOLO\-World and Grounding DINO achieve impressive zero\-shot recognition by aligning region proposals with text representations. However, their performance often degrades under domain shift. We introduce VLOD\-TTA, a test\-time adaptation \(TTA\) framework for VLODs that leverages dense proposal overlap and image\-conditioned prompt scores. First, an IoU\-weighted entropy objective is proposed that concentrates adaptation on spatially coherent proposal clusters and reduces confirmation bias from isolated boxes. Second, image\-conditioned prompt selection is introduced, which ranks prompts by image\-level compatibility and fuses the most informative prompts with the detector logits. Our benchmarking across diverse distribution shifts \-\- including stylized domains, driving scenes, low\-light conditions, and common corruptions \-\- shows the effectiveness of our method on two state\-of\-the\-art VLODs, YOLO\-World and Grounding DINO, with consistent improvements over the zero\-shot and TTA baselines. Code : https://github.com/imatif17/VLOD\-TTA

中文摘要：


代码链接：https://github.com/imatif17/VLOD-TTA

论文链接：[阅读更多](http://arxiv.org/abs/2510.00458v1)

---


## Rings of Light, Speed of AI: YOLO for Cherenkov Reconstruction / 

发布日期：2025-09-30

作者：Martino Borsato

摘要：Cherenkov rings play a crucial role in identifying charged particles in high\-energy physics \(HEP\) experiments. Most Cherenkov ring pattern reconstruction algorithms currently used in HEP experiments rely on a likelihood fit to the photo\-detector response, which often consumes a significant portion of the computing budget for event reconstruction. We present a novel approach to Cherenkov ring reconstruction using YOLO, a computer vision algorithm capable of real\-time object identification with a single pass through a neural network. We obtain a reconstruction efficiency above 95% and a pion misidentification rate below 5% across a wide momentum range for all particle species.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2509.26273v1)

---

