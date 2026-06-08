<div align="center">

# YOLO ArXiv Daily

[![Daily Papers](https://img.shields.io/badge/📅-每日更新-blue)]()
[![arXiv](https://img.shields.io/badge/arXiv-最新论文-red)](https://arxiv.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)

*自动追踪 YOLO 相关最新论文，提供中英文双语摘要*

</div>

---

## 📑 论文列表

> ### 1. HYolo: An Intelligent IoT\-Based Object Detection System Using Hypergraph Learning
> **🔹 中文标题：** HYolo：一种基于超图学习的智能物联网目标检测系统
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-06-03 |
> | 👤 作者 | Isha Abid |
>
> **📄 英文摘要：**
> This paper presents HYolo, an intelligent IoT\-based object detection framework that integrates hypergraph learning into the YOLO architecture. Traditional YOLO\-based object detection models primarily capture pairwise feature interactions and may fail to model complex high\-order relationships among objects and contextual features. To address this limitation, HYolo incorporates hypergraph learning to capture richer contextual dependencies and improve object representation. Experimental evaluation on the COCO dataset demonstrates significant performance improvements over baseline YOLO models. The proposed approach achieves approximately 12% improvement in mAP@50 while enhancing overall detection accuracy and robustness. By modeling high\-order feature relationships, HYolo provides improved contextual understanding and more reliable object detection performance in IoT\-based environments. The results indicate that integrating hypergraph learning into object detection pipelines offers a promising direction for intelligent and context\-aware IoT vision systems.
>
> **📝 中文摘要：**
> 本文提出HYolo框架，这是一种基于物联网的智能目标检测系统，通过将超图学习融入YOLO架构实现性能提升。传统基于YOLO的检测模型主要捕获成对特征交互，难以对目标与上下文特征间复杂的高阶关系进行建模。为突破此局限，HYolo引入超图学习以捕获更丰富的上下文依赖关系，从而优化目标表征能力。在COCO数据集上的实验评估表明，该模型较基线YOLO模型实现显著性能提升：在保持整体检测精度与鲁棒性增强的同时，mAP@50指标提升约12%。通过建模高阶特征关系，HYolo在物联网环境中实现了更优的上下文理解能力与更可靠的目标检测性能。研究结果证实，将超图学习融入目标检测流程，可为智能物联网视觉系统的上下文感知能力发展提供可行技术路径。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.04345v1)

---

> ### 2. Ultralytics YOLO26: Unified Real\-Time End\-to\-End Vision Models
> **🔹 中文标题：** Ultralytics YOLO26：统一的实时端到端视觉模型
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-06-02 |
> | 👤 作者 | Glenn Jocher |
>
> **📄 英文摘要：**
> Real\-time vision demands models that are accurate, efficient, and simple to deploy across diverse hardware. The YOLO family has become widely deployed for this reason, yet most YOLO detectors still rely on non\-maximum suppression at inference, carry heavy detection heads due to Distribution Focal Loss, require long training schedules, and can leave the smallest objects without positive label assignments. We present Ultralytics YOLO26, a unified real\-time vision model family that addresses these limitations through coordinated architecture and training advances. YOLO26 uses a dual\-head design for native NMS\-free end\-to\-end inference and removes DFL entirely, yielding a lighter head with unconstrained regression range. Its training pipeline combines MuSGD, a hybrid Muon\-SGD optimizer adapted from large language model training; Progressive Loss, which shifts supervision toward the inference\-time head; and STAL, a label assignment strategy that guarantees positive coverage for small objects. Beyond detection, YOLO26 introduces task\-specific head and loss designs for instance segmentation, pose estimation, and oriented detection, producing consistent gains across tasks and scales. The family spans five scales \(n/s/m/l/x\) and supports detection, instance segmentation, pose estimation, classification, and oriented detection in a single pipeline, with an open\-vocabulary extension, YOLOE\-26, for text\-, visual\-, and prompt\-free inference. Across all scales, YOLO26 achieves 40.9\-57.5 mAP on COCO at 1.7\-11.8 ms T4 TensorRT latency, advancing the accuracy\-latency Pareto front over prior real\-time detectors, while YOLOE\-26x reaches 40.6 AP on LVIS minival under text prompting. Code and models are available at https://github.com/ultralytics/ultralytics.
>
> **📝 中文摘要：**
> 实时视觉任务需要模型具备高精度、高效率以及跨硬件平台的易部署性。YOLO系列模型因满足这些需求而被广泛采用，但多数YOLO检测器在推理阶段仍依赖非极大值抑制，因采用分布式焦点损失而导致检测头计算量大，需要较长训练周期，且可能遗漏最小目标的正标签分配。我们提出Ultralytics YOLO26——一个通过架构与训练协同创新来突破现有局限的统一实时视觉模型家族。YOLO26采用双检测头设计实现原生无NMS端到端推理，彻底移除分布式焦点损失，构建出无回归范围约束的轻量化检测头。其训练流程结合了融合大语言模型训练经验的混合优化器MuSGD、将监督重心转向推理时检测头的渐进式损失，以及确保小目标正标签覆盖的STAL分配策略。除检测任务外，YOLO26针对实例分割、姿态估计和旋转目标检测设计了专用检测头与损失函数，在不同任务与尺度上实现一致性性能提升。该模型家族涵盖n/s/m/l/x五种规模，支持检测、实例分割、姿态估计、分类及旋转目标检测的单流式处理，并通过开放词汇扩展模块YOLOE-26实现文本、视觉及无提示推理。所有规模模型在COCO数据集上以1.7-11.8毫秒T4 TensorRT延迟实现了40.9-57.5 mAP，较现有实时检测器在精度-延迟帕累托前沿取得显著突破，其中YOLOE-26x在文本提示下于LVIS minival数据集达到40.6 AP。代码与模型详见 https://github.com/ultralytics/ultralytics。
>
> **💻 代码链接：** https://github.com/ultralytics/ultralytics.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.03748v1)

---

> ### 3. Detecting Pen\-In\-Air States from Video: A Proof\-of\-Concept Toward Complementary Handwriting Analysis
> **🔹 中文标题：** 视频中空中笔态的检测：一种辅助手写分析的概念验证
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-06-01 |
> | 👤 作者 | Lauren Sismeiro |
>
> **📄 英文摘要：**
> Dynamic aspects of handwriting are critical for assessing developmental disorders such as dysgraphia and are typically captured using digitizing tablets. However, tablet\-based sensing restricts analysis of Pen\-Up behavior to a short proximity range above the writing surface, potentially missing high\-lift in\-air movements. As a proof of concept, we investigate whether top\-view video can provide a complementary source of information for inferring pen\-contact states without relying on tablet proximity sensing. We propose an interpretable hybrid pipeline combining pen\-tip tracking using a YOLO\-based detector with kinematic feature extraction and machine learning classification. A pilot dataset of diverse handwriting videos was manually annotated at the frame level and evaluation used a Leave\-One\-Video\-Out \(LOVO\) protocol. The method achieved reliable event\-level detection of Pen\-Up segments, with an F\_2 score up to 0.805, consistent with the emphasis on recall in a screening\-oriented setting. These results support the feasibility of video\-based Pen\-Up detection as a low\-cost and non\-intrusive complement to digitizing tablets, and provide a foundation for future large\-scale studies.
>
> **📝 中文摘要：**
> 书写动态特征对于评估书写障碍等发育障碍至关重要，通常通过数位板进行捕捉。然而基于数位板的传感技术将抬笔行为的分析限制在书写表面上方的近距离范围，可能遗漏大幅度提笔的空中动作。作为概念验证，我们探究俯视视角视频能否在不依赖数位板邻近传感的前提下，为推断笔触接触状态提供补充信息。本文提出一种可解释的混合处理框架，结合YOLO检测器的笔尖追踪技术、运动特征提取与机器学习分类方法。通过对多样书写视频数据集进行逐帧人工标注，并采用留一视频交叉验证协议进行评估，该方法实现了可靠的抬笔片段检测，F₂值达0.805，符合筛查场景对召回率的侧重。研究结果证实基于视频的抬笔检测技术作为数位板低成本非接触式补充方案的可行性，为后续大规模研究奠定了基础。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.02342v1)

---

> ### 4. Collaborative Space Object Detection with Multi\-Satellite Viewpoints in LEO Constellations
> **🔹 中文标题：** 近地轨道星座中多卫星视角协同空间目标检测
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-06-01 |
> | 👤 作者 | Xingyu Qu |
>
> **📄 英文摘要：**
> With the growing number of satellites in low Earth orbit \(LEO\) constellations, the near\-Earth space environment has become increasingly congested, making space object detection \(SOD\) a pressing challenge for space safety and sustainability. To mitigate collision risks and ensure the continuity of space operations, SOD systems must deliver fast and accurate detection under stringent onboard constraints. In this paper, we investigate the potential of multi\-viewpoint observation fusion within a deep learning \(DL\) framework to enhance SOD performance. We design a practical multi\-view pipeline and several input representations for feeding multi\-view data into YOLO\-based detectors. Our experiments show that using multi\-view inputs is feasible in most cases and typically produces better results for mAP50 and mAP50\-95. For example, in model YOLOv9\-m, single\-view compared to a three\-view fused RGB setting, mAP50 increases from 0.638 to 0.732, while mAP50\-95 improves from 0.227 to 0.276. Compared with the single\-view setting, the best three\-view grayscale configuration improves mAP50 by 36.3% and mAP50\-95 by 46.5%. These findings establish multi\-view fusion as a viable and effective strategy for SOD, with broad implications for space situational awareness in LEO constellation deployments.
>
> **📝 中文摘要：**
> 随着低地球轨道卫星星座数量的不断增加，近地空间环境日益拥挤，使得空间物体检测成为保障空间安全与可持续发展亟待解决的挑战。为降低碰撞风险并确保空间任务的连续性，空间物体检测系统必须在星载严苛条件下实现快速精准识别。本研究深入探讨了在深度学习框架内融合多视角观测以提升检测性能的潜力，设计了一套实用的多视角处理流程，并开发了多种数据输入方案将多视角信息融入基于YOLO的检测模型。实验结果表明：多视角输入在多数场景下具备可行性，通常能在mAP50和mAP50-95指标上取得更优表现。以YOLOv9-m模型为例，相比单视角设置，三视角RGB融合使mAP50从0.638提升至0.732，mAP50-95从0.227提升至0.276；与单视角相比，最优三视角灰度配置使mAP50提升36.3%，mAP50-95提升46.5%。研究证实多视角融合是切实有效的空间物体检测策略，对低轨卫星星座部署中的空间态势感知具有重要应用价值。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.01895v1)

---

> ### 5. Hierarchically Decoupled Mixture\-of\-Experts for Robust Traffic Sign Recognition in Complex Driving Scenarios
> **🔹 中文标题：** 分层解耦混合专家模型在复杂驾驶场景下的鲁棒交通标志识别
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-06-01 |
> | 👤 作者 | Mingxiao Wang |
>
> **📄 英文摘要：**
> Traffic sign detection is a fundamental component of environmental perception in autonomous driving and intelligent transportation systems. However, most existing detectors rely on static inference with globally shared parameters, limiting their ability to adapt to diverse and unstructured traffic scenarios. As a result, a single static model often struggles to simultaneously handle both clear near\-range samples and challenging conditions such as distant small targets or adverse weather environments. To address this limitation, we propose CBDES MoE TSR, a hierarchically decoupled heterogeneous mixture\-of\-experts\(MoE\) framework for traffic sign recognition. The proposed framework departs from the conventional globally shared parameter paradigm by introducing a heterogeneous You Only Look Once \(YOLO\) expert pool together with a lightweight gating network, enabling an image\-level dynamic routing mechanism. Based on the semantic characteristics of the input image, the gating module selectively activates the most suitable expert model from the expert pool, enabling a shift from fixed parameter fitting to on\-demand dynamic representation. This design enhances feature extraction capability for specific scenarios while maintaining controlled inference overhead. Experimental results demonstrate that the proposed method achieves a remarkable balance between detection accuracy and efficiency on the composite traffic sign dataset. Specifically, our method attains an mAP50\-95 of 76.8%, yielding a 2.3% improvement over the baseline method \(74.5%\) while simultaneously reducing computational overhead by approximately 39.4%. These findings robustly validate the effectiveness of the proposed approach.
>
> **📝 中文摘要：**
> 交通标志检测是自动驾驶与智能交通系统中环境感知的基础组成部分。然而，现有检测器大多依赖全局共享参数的静态推理模式，难以适应多样化且非结构化的交通场景。单一静态模型常难以同时处理清晰的近距离样本与远距离小目标、恶劣天气等挑战性环境。为此，我们提出CBDES MoE TSR——一种分层解耦的异构混合专家框架用于交通标志识别。该框架突破传统全局共享参数范式，通过引入异构You Only Look Once专家池与轻量级门控网络，实现图像级动态路由机制。门控模块根据输入图像的语义特征，从专家池中选择性激活最匹配的专家模型，将固定参数拟合转变为按需动态表征。该设计在保持可控推理开销的同时，增强了特定场景的特征提取能力。实验结果表明，所提方法在复合交通标志数据集上实现了检测精度与效率的显著平衡。具体而言，本方法在mAP50-95指标上达到76.8%，较基线方法（74.5%）提升2.3%，同时降低约39.4%的计算开销。实验数据有力验证了该方法的有效性。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.01822v2)

---

<div align="center">

*由 [YOLO-Arxiv-Daily](https://github.com/WangQvQ/YOLO-Arxiv-Daily) 自动生成*

</div>