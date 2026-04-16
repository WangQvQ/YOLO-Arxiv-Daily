# 每日从arXiv中获取最新YOLO相关论文


## A Multi\-Stage Optimization Pipeline for Bethesda Cell Detection in Pap Smear Cytology / 

发布日期：2026-04-15

作者：Martin Amster

摘要：Computer vision techniques have advanced significantly in recent years, finding diverse and impactful applications within the medical field. In this paper, we introduce a new framework for the detection of Bethesda cells in Pap smear images, developed for Track B of the Riva Cytology Challenge held in association with the International Symposium on Biomedical Imaging \(ISBI\). This work focuses on enhancing computer vision models for cell detection, with performance evaluated using the mAP50\-95 metric. We propose a solution based on an ensemble of YOLO and U\-Net architectures, followed by a refinement stage utilizing overlap removal techniques and a binary classifier. Our framework achieved second place with a mAP50\-95 score of 0.5909 in the competition. The implementation and source code are available at the following repository: github.com/martinamster/riva\-trackb

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2604.13939v1)

---


## Don't Let AI Agents YOLO Your Files: Shifting Information and Control to Filesystems for Agent Safety and Autonomy / 

发布日期：2026-04-15

作者：Shawn

摘要：AI coding agents operate directly on users' filesystems, where they regularly corrupt data, delete files, and leak secrets. Current approaches force a tradeoff between safety and autonomy: unrestricted access risks harm, while frequent permission prompts burden users and block agents. To understand this problem, we conduct the first systematic study of agent filesystem misuse, analyzing 290 public reports across 13 frameworks. Our analysis reveals that today's agents have limited information about their filesystem effects and insufficient control over them. We therefore argue for shifting this information and control to the filesystem itself.   Based on this principle, we design YoloFS, an agent\-native filesystem with three techniques. Staging isolates all mutations before commit, giving users corrective control. Snapshots extend this control to agents, letting them detect and correct their own mistakes. Progressive permission provides users with preventive control by gating access with minimal interaction. To evaluate YoloFS, we introduce a new methodology that captures user\-agent\-filesystem interactions. On 11 tasks with hidden side effects, YoloFS enables agent self\-correction in 8 while keeping all effects staged and reviewable. On 112 routine tasks, YoloFS requires fewer user interactions while matching the baseline success rate.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2604.13536v1)

---


## Multi\-Agent Object Detection Framework Based on Raspberry Pi YOLO Detector and Slack\-Ollama Natural Language Interface / 

发布日期：2026-04-14

作者：Vladimir Kalušev

摘要：The paper presents design and prototype implementation of an edge based object detection system within the new paradigm of AI agents orchestration. It goes beyond traditional design approaches by leveraging on LLM based natural language interface for system control and communication and practically demonstrates integration of all system components into a single resource constrained hardware platform. The method is based on the proposed multi\-agent object detection framework which tightly integrates different AI agents within the same task of providing object detection and tracking capabilities. The proposed design principles highlight the fast prototyping approach that is characteristic for transformational potential of generative AI systems, which are applied during both development and implementation stages. Instead of specialized communication and control interface, the system is made by using Slack channel chatbot agent and accompanying Ollama LLM reporting agent, which are both run locally on the same Raspberry Pi platform, alongside the dedicated YOLO based computer vision agent performing real time object detection and tracking. Agent orchestration is implemented through a specially designed event based message exchange subsystem, which represents an alternative to completely autonomous agent orchestration and control characteristic for contemporary LLM based frameworks like the recently proposed OpenClaw. Conducted experimental investigation provides valuable insights into limitations of the low cost testbed platforms in the design of completely centralized multi\-agent AI systems. The paper also discusses comparative differences between presented approach and the solution that would require additional cloud based external resources.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2604.13345v1)

---


## DroneScan\-YOLO: Redundancy\-Aware Lightweight Detection for Tiny Objects in UAV Imagery / 

发布日期：2026-04-14

作者：Yann V. Bellec

摘要：Aerial object detection in UAV imagery presents unique challenges due to the high prevalence of tiny objects, adverse environmental conditions, and strict computational constraints. Standard YOLO\-based detectors fail to address these jointly: their minimum detection stride of 8 pixels renders sub\-32px objects nearly undetectable, their CIoU loss produces zero gradients for non\-overlapping tiny boxes, and their architectures contain significant filter redundancy. We propose DroneScan\-YOLO, a holistic system contribution that addresses these limitations through four coordinated design choices: \(1\) increased input resolution of 1280x1280 to maximize spatial detail for tiny objects, \(2\) RPA\-Block, a dynamic filter pruning mechanism based on lazy cosine\-similarity updates with a 10\-epoch warm\-up period, \(3\) MSFD, a lightweight P2 detection branch at stride 4 adding only 114,592 parameters \(\+1.1%\), and \(4\) SAL\-NWD, a hybrid loss combining Normalized Wasserstein Distance with size\-adaptive CIoU weighting, integrated into YOLOv8's TaskAligned assignment pipeline. Evaluated on VisDrone2019\-DET, DroneScan\-YOLO achieves 55.3% mAP@50 and 35.6% mAP@50\-95, outperforming the YOLOv8s baseline by \+16.6 and \+12.3 points respectively, improving recall from 0.374 to 0.518, and maintaining 96.7 FPS inference speed with only \+4.1% parameters. Gains are most pronounced on tiny object classes: bicycle AP@50 improves from 0.114 to 0.328 \(\+187%\), and awning\-tricycle from 0.156 to 0.237 \(\+52%\).

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2604.13278v1)

---


## Monte Carlo Stochastic Depth for Uncertainty Estimation in Deep Learning / 

发布日期：2026-04-14

作者：Adam T. Müller

摘要：The deployment of deep neural networks in safety\-critical systems necessitates reliable and efficient uncertainty quantification \(UQ\). A practical and widespread strategy for UQ is repurposing stochastic regularizers as scalable approximate Bayesian inference methods, such as Monte Carlo Dropout \(MCD\) and MC\-DropBlock \(MCDB\). However, this paradigm remains under\-explored for Stochastic Depth \(SD\), a regularizer integral to the residual\-based backbones of most modern architectures. While prior work demonstrated its empirical promise for segmentation, a formal theoretical connection to Bayesian variational inference and a benchmark on complex, multi\-task problems like object detection are missing. In this paper, we first provide theoretical insights connecting Monte Carlo Stochastic Depth \(MCSD\) to principled approximate variational inference. We then present the first comprehensive empirical benchmark of MCSD against MCD and MCDB on state\-of\-the\-art detectors \(YOLO, RT\-DETR\) using the COCO and COCO\-O datasets. Our results position MCSD as a robust and computationally efficient method that achieves highly competitive predictive accuracy \(mAP\), notably yielding slight improvements in calibration \(ECE\) and uncertainty ranking \(AUARC\) compared to MCD. We thus establish MCSD as a theoretically\-grounded and empirically\-validated tool for efficient Bayesian approximation in modern deep learning.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2604.12719v1)

---

