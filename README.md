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
> **🔹 中文标题：** HYolo：一种基于智能物联网的超图学习目标检测系统
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
> 本文提出HYolo——一种基于物联网的智能目标检测框架，该框架将超图学习整合到YOLO架构中。传统基于YOLO的目标检测模型主要捕捉成对特征交互，难以建模目标与上下文特征间的复杂高阶关系。为突破这一局限，HYolo通过引入超图学习来捕获更丰富的上下文依赖关系，从而增强目标表征能力。在COCO数据集上的实验评估表明，相比基线YOLO模型，该方法在mAP@50指标上实现了约12%的显著提升，同时增强了整体检测精度与鲁棒性。通过建模高阶特征关系，HYolo在基于物联网的环境中提供了更优的上下文理解能力与更可靠的目标检测性能。研究结果证实，将超图学习整合至目标检测流程中，为构建智能且具有上下文感知能力的物联网视觉系统提供了极具前景的发展方向。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.04345v1)

---

> ### 2. Ultralytics YOLO26: Unified Real\-Time End\-to\-End Vision Models
> **🔹 中文标题：** Ultralytics YOLO26：统一实时端到端视觉模型
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
> 实时视觉需要模型具备高准确性、高效率以及跨硬件的易部署性。YOLO系列因此得到广泛应用，但现有的YOLO检测器在推理时大多仍依赖非极大值抑制，因分布焦点损失而具有沉重的检测头，需要较长的训练周期，且可能遗漏最小目标的正标签分配。

本文提出Ultralytics YOLO26——一个通过架构与训练协同优化来突破上述局限的统一实时视觉模型家族。YOLO26采用双检测头设计实现原生无NMS端到端推理，并完全移除分布焦点损失，从而构建出回归范围无约束的轻量化检测头。其训练流程整合三项创新：改编自大语言模型训练的混合优化器MuSGD（Muon-SGD）、将监督重心向推理时检测头倾斜的渐进式损失，以及确保小目标正样本覆盖的标签分配策略STAL。

除检测任务外，YOLO26还引入针对实例分割、姿态估计和旋转检测的任务专用头与损失函数设计，在不同任务和尺度上均实现稳定性能提升。该模型家族涵盖五种规模（n/s/m/l/x），支持检测、实例分割、姿态估计、分类和旋转检测的统一流水线，并通过开放词表扩展YOLOE-26实现文本无关、视觉无关和无提示推理。在COCO数据集上，所有规模模型以1.7-11.8毫秒（T4 TensorRT延迟）实现40.9-57.5 mAP，在精度-延迟帕累托前沿超越现有实时检测器；而YOLOE-26x在LVIS minival数据集上达到40.6 AP（文本提示条件下）。代码与模型已发布于https://github.com/ultralytics/ultralytics。
>
> **💻 代码链接：** https://github.com/ultralytics/ultralytics.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.03748v1)

---

> ### 3. Detecting Pen\-In\-Air States from Video: A Proof\-of\-Concept Toward Complementary Handwriting Analysis
> **🔹 中文标题：** 通过视频检测悬空笔状态：一种面向互补性手写分析的概念验证
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
> 书写动态特征对于评估书写障碍等发育性疾病至关重要，通常通过数字化书写板进行采集。然而基于书写板的感知技术将抬笔行为分析限制在书写表面上方的短距离范围内，可能遗漏高幅度的空中运笔动作。作为概念验证，本研究探究顶视视频能否在不依赖书写板邻近感知的情况下，为推断笔触状态提供补充信息源。我们提出一种可解释的混合处理流程，结合基于YOLO的笔尖追踪检测器与运动特征提取及机器学习分类器。通过人工标注的多视角书写视频数据集进行帧级别标注，采用留一视频交叉验证法进行评估。该方法实现了可靠的抬笔片段事件级检测，F2分数最高达0.805，这与筛查场景中对召回率的重视相一致。研究结果表明，基于视频的抬笔检测技术可作为数字化书写板低成本、非侵入性的补充手段，并为未来大规模研究奠定了基础。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.02342v1)

---

> ### 4. Collaborative Space Object Detection with Multi\-Satellite Viewpoints in LEO Constellations
> **🔹 中文标题：** 低地球轨道星座中基于多卫星视角的协同空间目标检测
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
> 随着低地球轨道（LEO）卫星星座数量的持续增长，近地空间环境日益拥挤，使得空间目标检测（SOD）成为航天安全与可持续发展面临的紧迫挑战。为降低碰撞风险并保障空间任务的延续性，空间目标检测系统必须在严格的星载设备约束下实现快速精准的探测。本文研究了在深度学习（DL）框架内融合多视角观测数据以提升空间目标检测性能的可行性。我们设计了一套实用的多视角处理流程，并提出多种输入表示方法，用于将多视角数据馈送至基于YOLO的检测器。实验结果表明，在多数场景下采用多视角输入具有可行性，且能显著提升mAP50与mAP50-95指标：以YOLOv9-m模型为例，与单视角输入相比，三视角融合RGB配置使mAP50从0.638提升至0.732，mAP50-95从0.227提升至0.276；最优三视角灰度配置相比单视角设置，分别实现mAP50与mAP50-95指标36.3%和46.5%的性能提升。这些研究结果证实多视角融合是实现空间目标检测的有效策略，对低轨卫星星座的空间态势感知具有重要应用价值。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.01895v1)

---

> ### 5. Hierarchically Decoupled Mixture\-of\-Experts for Robust Traffic Sign Recognition in Complex Driving Scenarios
> **🔹 中文标题：** 分层解耦混合专家模型在复杂驾驶场景中的鲁棒交通标志识别
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
> 交通标志检测是自动驾驶与智能交通系统中环境感知的基础组成部分。然而，现有大多数检测器依赖全局共享参数的静态推理，限制了其适应多样化、非结构化交通场景的能力。单一静态模型往往难以同时处理清晰的近距样本，以及远距离小目标或恶劣天气等复杂环境下的挑战性场景。为解决这一局限，本文提出CBDES MoE TSR——一种面向交通标志识别的层级解耦异构混合专家模型框架。该框架突破传统全局共享参数范式，引入异构YOLO专家池与轻量化门控网络，构建图像级动态路由机制。根据输入图像的语义特征，门控模块可从专家池中选择性激活最适合的专家模型，从而实现从固定参数拟合到按需动态表征的转变。此设计在保持可控推理开销的同时，增强了特定场景的特征提取能力。实验结果表明，所提方法在复合交通标志数据集上实现了检测精度与效率的显著平衡。具体而言，该方法在mAP50-95指标上达到76.8%，较基线方法（74.5%）提升2.3个百分点，同时计算开销降低约39.4%。这些发现充分验证了所提方法的有效性。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.01822v1)

---

<div align="center">

*由 [YOLO-Arxiv-Daily](https://github.com/WangQvQ/YOLO-Arxiv-Daily) 自动生成*

</div>