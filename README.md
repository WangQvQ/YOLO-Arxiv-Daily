<div align="center">

# YOLO ArXiv Daily

[![Daily Papers](https://img.shields.io/badge/📅-每日更新-blue)]()
[![arXiv](https://img.shields.io/badge/arXiv-最新论文-red)](https://arxiv.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)

*自动追踪 YOLO 相关最新论文，提供中英文双语摘要*

</div>

---

## 📑 论文列表

> ### 1. Attention\-Guided Autoencoder Fusion for Insulator Defect Detection Using UAV Transmission\-Line Imaging
> **🔹 中文标题：** 基于无人机输电线路成像的注意力引导自编码器融合绝缘子缺陷检测
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-06-03 |
> | 👤 作者 | Malak Allam |
>
> **📄 英文摘要：**
> Automated defect detection in high\-voltage transmission\-line insulators remains challenging due to severe class imbalance, large scale variation, and the small spatial extent of defect instances in Unmanned Aerial Vehicle \(UAV\) imagery. To address these challenges, this paper proposes AE\-YOLO, an Attention\-Guided AutoEncoder\-Enhanced YOLO framework for robust insulator defect detection. The architecture integrates lightweight bottleneck autoencoders within a Feature Pyramid Network\-Path Aggregation Network \(FPN\-PAN\) neck. This preserves anomaly\-sensitive information during multi\-scale feature fusion. Convolutional Block Attention Modules \(CBAM\) are used throughout the backbone, enhancing feature discrimination and suppressing background interference. The framework also introduces a variance\-maximizing autoencoder regularization strategy, which encourages diverse, defect\-discriminative latent representations. The network trains using a unified objective that combines focal loss, Complete IoU \(CIoU\) loss, and autoencoder regularization to address foreground\-background imbalance and improve localization accuracy. During inference, Weighted Boxes Fusion \(WBF\) combines predictions from YOLOv8, YOLOv10, and YOLO11. An autoencoder\-guided confidence boosting mechanism improves sensitivity to rare defect categories. Experiments on the Insulator\-Defect Detection dataset show that AE\-YOLO with an EfficientNetV2 backbone achieves 95.10 percent mAP at 0.5, 96.40 percent precision, and 93.80 percent recall. This performance surpasses the strongest YOLO\-family baseline by 5.0 points in mAP at 0.5 and 6.7 points in recall. These results confirm the effectiveness and adaptability of the framework. The model is a practical and scalable solution for UAV\-based transmission\-line inspection and defect monitoring.
>
> **📝 中文摘要：**
> 针对无人机影像中高压输电线路绝缘子缺陷检测存在的类别不平衡严重、尺度差异大及缺陷实例空间范围小等挑战，本文提出一种注意力引导的自编码器增强型YOLO框架。该框架在特征金字塔网络-路径聚合网络颈部架构中集成轻量级瓶颈自编码器，在多尺度特征融合过程中保留异常敏感信息。骨干网络全程采用卷积块注意力模块，增强特征判别能力并抑制背景干扰。同时引入方差最大化自编码器正则化策略，鼓励模型学习多样化且具有缺陷判别性的潜在表征。网络采用融合焦点损失、完整交并比损失与自编码器正则化的统一目标函数进行训练，以解决前景背景不平衡问题并提升定位精度。推理阶段采用加权框融合技术聚合YOLOv8、YOLOv10及YOLO11的预测结果，通过自编码器引导的置信度增强机制提升稀有缺陷类别的检测敏感性。在绝缘子缺陷检测数据集上的实验表明，采用EfficientNetV2骨干网络的AE-YOLO模型在0.5阈值下平均精度均值达95.10%，精确率为96.40%，召回率为93.80%。该性能较最优YOLO系列基线模型分别提升5.0个百分点（mAP@0.5）和6.7个百分点（召回率），验证了框架的有效性与适应性。该模型为基于无人机的输电线路巡检与缺陷监测提供了实用且可扩展的解决方案。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.06536v1)

---

> ### 2. HYolo: An Intelligent IoT\-Based Object Detection System Using Hypergraph Learning
> **🔹 中文标题：** HYolo：基于超图学习的物联网智能目标检测系统
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
> 本文提出HYolo——一种基于物联网的智能目标检测框架，该框架将超图学习集成到YOLO架构中。传统基于YOLO的目标检测模型主要捕捉成对特征交互，可能无法有效建模目标与上下文特征间的复杂高阶关系。为解决这一局限，HYolo引入超图学习机制以捕获更丰富的上下文依赖关系，从而增强目标表征能力。在COCO数据集上的实验评估表明，该模型相比基线YOLO模型实现了显著的性能提升：mAP@50指标提升约12%，同时整体检测精度与鲁棒性均得到增强。通过建模高阶特征关系，HYolo在基于物联网的环境中提供了更优的上下文理解能力和更可靠的目标检测性能。研究结果表明，将超图学习集成到目标检测流程中，为智能感知物联网视觉系统的发展提供了极具前景的研究方向。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.04345v1)

---

> ### 3. Ultralytics YOLO26: Unified Real\-Time End\-to\-End Vision Models
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
> 实时视觉要求模型具备高精度、高效率和跨硬件易部署的特性。YOLO系列模型因满足这些需求而得到广泛应用，但现有YOLO检测器在推理阶段仍普遍依赖非极大值抑制算法、因分布式焦点损失而需采用厚重的检测头、需要冗长的训练周期，且可能遗漏最小目标的正标签分配。我们提出Ultralytics YOLO26——一个通过协同架构与训练创新解决上述局限的统一实时视觉模型家族。YOLO26采用双头设计实现原生无NMS的端到端推理，并彻底移除分布式焦点损失，从而构建出更轻量化且具有无约束回归范围的检测头。其训练流程融合了三大创新：适应大语言模型训练的混合Muon-SGD优化器（MuSGD）、将监督重心转向推理阶段检测头的渐进式损失（Progressive Loss），以及确保小目标正样本覆盖的STAL标签分配策略。在检测任务之外，YOLO26还为实例分割、姿态估计和旋转目标检测设计了专用检测头与损失函数，在各任务与模型尺度上均实现一致性性能提升。该模型家族提供五种规模（n/s/m/l/x），在单一流程中支持检测、实例分割、姿态估计、分类和旋转目标检测任务，并配备支持文本/视觉/无提示推理的开放词汇扩展模块YOLOE-26。全系列模型在COCO数据集上达到40.9-57.5 mAP的精度（T4 TensorRT推理延迟仅1.7-11.8毫秒），在精度-延迟帕累托前沿显著超越现有实时检测器，其中YOLOE-26x在LVIS minival数据集上通过文本提示达到40.6 AP。代码与模型已开源：https://github.com/ultralytics/ultralytics。
>
> **💻 代码链接：** https://github.com/ultralytics/ultralytics.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.03748v1)

---

> ### 4. Detecting Pen\-In\-Air States from Video: A Proof\-of\-Concept Toward Complementary Handwriting Analysis
> **🔹 中文标题：** 基于视频的悬空笔状态检测：实现互补性笔迹分析的概念验证
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
> 书写动态特征对评估书写障碍等发育性疾病至关重要，通常通过数字化书写板进行捕捉。然而，基于书写板的传感技术将提笔行为的分析局限于书写表面上方的短距离范围，可能遗漏高位抬起的空中运笔轨迹。为验证概念可行性，本研究探究顶视角视频能否在不依赖书写板邻近传感的前提下，为推断笔尖接触状态提供补充信息。我们提出一种可解释的混合处理流程，结合基于YOLO检测器的笔尖追踪、运动特征提取与机器学习分类。通过人工标注的多样化手写视频初步数据集实现逐帧标注，并采用留一视频法进行评估。该方法在提笔片段检测中达到了可靠水平，F_2分数最高达0.805，符合筛查场景对召回率的侧重。结果表明，基于视频的提笔检测技术作为低成本、非侵入式的书写板补充方案具有可行性，为未来大规模研究奠定了基础。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.02342v1)

---

> ### 5. Collaborative Space Object Detection with Multi\-Satellite Viewpoints in LEO Constellations
> **🔹 中文标题：** 基于多卫星视角的低轨星座协作式空间目标检测
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
> 随着低轨卫星星座中卫星数量持续增加，近地空间环境日益拥挤，使得空间目标检测成为保障太空安全与可持续发展的紧迫挑战。为降低碰撞风险并确保空间作业连续性，空间目标检测系统必须在严格的星载设备限制下实现快速精准的探测。本文研究在深度学习框架内融合多视角观测以提升空间目标检测性能的潜力。我们设计了一套实用的多视角处理流程，并开发了多种输入表示方法，将多视角数据有效输入基于YOLO的检测网络。实验表明：多视角输入在大多数情况下具有可行性，通常能在mAP50和mAP50-95指标上取得更优表现。以YOLOv9-m模型为例，相比单视角输入，三视角融合RGB设置使mAP50从0.638提升至0.732，mAP50-95从0.227提升至0.276。与单视角配置相比，最优三视角灰度配置使mAP50提升36.3%，mAP50-95提升46.5%。这些发现证实了多视角融合作为空间目标检测策略的可行性与有效性，对低轨卫星星座的空间态势感知具有重要启示意义。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.01895v1)

---

<div align="center">

*由 [YOLO-Arxiv-Daily](https://github.com/WangQvQ/YOLO-Arxiv-Daily) 自动生成*

</div>