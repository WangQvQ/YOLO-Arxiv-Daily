# 每日从arXiv中获取最新YOLO相关论文


## BLO\-Inst: Bi\-Level Optimization Based Alignment of YOLO and SAM for Robust Instance Segmentation / BLO-Inst：基于双层优化的YOLO和SAM对齐，用于鲁棒实例分割

发布日期：2026-01-29

作者：Li Zhang

摘要：The Segment Anything Model has revolutionized image segmentation with its zero\-shot capabilities, yet its reliance on manual prompts hinders fully automated deployment. While integrating object detectors as prompt generators offers a pathway to automation, existing pipelines suffer from two fundamental limitations: objective mismatch, where detectors optimized for geometric localization do not correspond to the optimal prompting context required by SAM, and alignment overfitting in standard joint training, where the detector simply memorizes specific prompt adjustments for training samples rather than learning a generalizable policy. To bridge this gap, we introduce BLO\-Inst, a unified framework that aligns detection and segmentation objectives by bi\-level optimization. We formulate the alignment as a nested optimization problem over disjoint data splits. In the lower level, the SAM is fine\-tuned to maximize segmentation fidelity given the current detection proposals on a subset \($D\_1$\). In the upper level, the detector is updated to generate bounding boxes that explicitly minimize the validation loss of the fine\-tuned SAM on a separate subset \($D\_2$\). This effectively transforms the detector into a segmentation\-aware prompt generator, optimizing the bounding boxes not just for localization accuracy, but for downstream mask quality. Extensive experiments demonstrate that BLO\-Inst achieves superior performance, outperforming standard baselines on tasks in general and biomedical domains.

中文摘要：Segment Anything Model凭借其零样本功能彻底改变了图像分割，但其对手动提示的依赖阻碍了完全自动化的部署。虽然将对象检测器集成为提示生成器提供了一条实现自动化的途径，但现有的管道存在两个基本局限性：目标不匹配，即针对几何定位进行优化的检测器与SAM所需的最佳提示上下文不符，以及标准联合训练中的对齐过拟合，即检测器只是记忆训练样本的特定提示调整，而不是学习可推广的策略。为了弥合这一差距，我们引入了BLO-Inst，这是一个通过双层优化将检测和分割目标对齐的统一框架。我们将对齐问题表述为不相交数据分割上的嵌套优化问题。在较低级别，SAM经过微调，以在给定子集（$D_1$）上的当前检测建议的情况下最大限度地提高分割保真度。在上层，更新检测器以生成边界框，明确地最小化单独子集（$D_2$）上微调SAM的验证损失。这有效地将检测器转换为分段感知提示生成器，优化边界框不仅是为了定位精度，也是为了下游掩模质量。大量实验表明，BLO-Inst在一般和生物医学领域的任务中表现优异，优于标准基线。


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2601.22061v1)

---


## An AI Framework for Microanastomosis Motion Assessment / 微吻合运动评估的人工智能框架

发布日期：2026-01-28

作者：Yan Meng

摘要：Proficiency in microanastomosis is a fundamental competency across multiple microsurgical disciplines. These procedures demand exceptional precision and refined technical skills, making effective, standardized assessment methods essential. Traditionally, the evaluation of microsurgical techniques has relied heavily on the subjective judgment of expert raters. They are inherently constrained by limitations such as inter\-rater variability, lack of standardized evaluation criteria, susceptibility to cognitive bias, and the time\-intensive nature of manual review. These shortcomings underscore the urgent need for an objective, reliable, and automated system capable of assessing microsurgical performance with consistency and scalability. To bridge this gap, we propose a novel AI framework for the automated assessment of microanastomosis instrument handling skills. The system integrates four core components: \(1\) an instrument detection module based on the You Only Look Once \(YOLO\) architecture; \(2\) an instrument tracking module developed from Deep Simple Online and Realtime Tracking \(DeepSORT\); \(3\) an instrument tip localization module employing shape descriptors; and \(4\) a supervised classification module trained on expert\-labeled data to evaluate instrument handling proficiency. Experimental results demonstrate the effectiveness of the framework, achieving an instrument detection precision of 97%, with a mean Average Precision \(mAP\) of 96%, measured by Intersection over Union \(IoU\) thresholds ranging from 50% to 95% \(mAP50\-95\).

中文摘要：精通显微吻合是多个显微外科学科的基本能力。这些程序要求极高的精度和精湛的技术技能，因此有效、标准化的评估方法至关重要。传统上，显微外科技术的评估在很大程度上依赖于专家评分员的主观判断。它们固有地受到限制，如评分者之间的差异性、缺乏标准化的评估标准、易受认知偏见的影响以及人工审查的时间密集性。这些缺点强调了迫切需要一个客观、可靠和自动化的系统，能够以一致性和可扩展性评估显微外科手术的性能。为了弥合这一差距，我们提出了一种新的人工智能框架，用于自动评估微吻合手术器械的操作技能。该系统集成了四个核心组件：（1）基于You Only Look Once（YOLO）架构的仪器检测模块；（2）由Deep Simple Online and Real-time tracking（DeepSORT）开发的仪器跟踪模块；（3）采用形状描述符的器械尖端定位模块；以及（4）在专家标记数据上训练的监督分类模块，用于评估仪器处理能力。实验结果证明了该框架的有效性，实现了97%的仪器检测精度，平均精度（mAP）为96%，通过50%至95%的联合交叉（IoU）阈值（mAP50-95）进行测量。


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2601.21120v1)

---


## VisGuardian: A Lightweight Group\-based Privacy Control Technique For Front Camera Data From AR Glasses in Home Environments / VisGuardian：一种基于组的轻量级隐私控制技术，用于家庭环境中AR眼镜的前置摄像头数据

发布日期：2026-01-27

作者：Shuning Zhang

摘要：Always\-on sensing of AI applications on AR glasses makes traditional permission techniques ill\-suited for context\-dependent visual data, especially within home environments. The home presents a highly challenging privacy context due to the high density of sensitive objects, and the frequent presence of non\-consenting family members, and the intimate nature of daily routines, making it a critical focus area for scalable privacy control mechanisms. Existing fine\-grained controls, while offering nuanced choices, are inefficient for managing multiple private objects. We propose VisGuardian, a fine\-grained content\-based visual permission technique for AR glasses. VisGuardian features a group\-based control mechanism that enables users to efficiently manage permissions for multiple private objects. VisGuardian detects objects using YOLO and adopts a pre\-classified schema to group them. By selecting a single object, users can efficiently obscure groups of related objects based on criteria including privacy sensitivity, object category, or spatial proximity. A technical evaluation shows VisGuardian achieves mAP50 of 0.6704 with only 14.0 ms latency and a 1.7% increase in battery consumption per hour. Furthermore, a user study \(N=24\) comparing VisGuardian to slider\-based and object\-based baselines found it to be significantly faster for setting permissions and was preferred by users for its efficiency, effectiveness, and ease of use.

中文摘要：AR眼镜上的AI应用程序的持续感知使得传统的许可技术不适合依赖于上下文的视觉数据，尤其是在家庭环境中。由于敏感对象的高密度、非自愿家庭成员的频繁出现以及日常生活的亲密性，家庭呈现出极具挑战性的隐私环境，使其成为可扩展隐私控制机制的关键焦点领域。现有的细粒度控件虽然提供了细微的选择，但对于管理多个私有对象来说效率低下。我们提出了VisGuardian，这是一种用于AR眼镜的细粒度基于内容的视觉许可技术。VisGuardian具有基于组的控制机制，使用户能够有效地管理多个私有对象的权限。VisGuardian使用YOLO检测对象，并采用预分类模式对其进行分组。通过选择单个对象，用户可以根据隐私敏感性、对象类别或空间接近度等标准有效地模糊相关对象组。技术评估显示，VisGuardian的mAP50为0.6704，延迟仅为14.0ms，每小时电池消耗量增加1.7%。此外，一项用户研究（N=24）将VisGuardian与基于滑块和基于对象的基线进行了比较，发现它在设置权限方面明显更快，并且因其效率、有效性和易用性而受到用户的青睐。


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2601.19502v1)

---


## YOLO\-DS: Fine\-Grained Feature Decoupling via Dual\-Statistic Synergy Operator for Object Detection / YOLO-DS：基于双统计协同算子的细粒度特征解耦用于目标检测

发布日期：2026-01-26

作者：Lin Huang

摘要：One\-stage object detection, particularly the YOLO series, strikes a favorable balance between accuracy and efficiency. However, existing YOLO detectors lack explicit modeling of heterogeneous object responses within shared feature channels, which limits further performance gains. To address this, we propose YOLO\-DS, a framework built around a novel Dual\-Statistic Synergy Operator \(DSO\). The DSO decouples object features by jointly modeling the channel\-wise mean and the peak\-to\-mean difference. Building upon the DSO, we design two lightweight gating modules: the Dual\-Statistic Synergy Gating \(DSG\) module for adaptive channel\-wise feature selection, and the Multi\-Path Segmented Gating \(MSG\) module for depth\-wise feature weighting. On the MS\-COCO benchmark, YOLO\-DS consistently outperforms YOLOv8 across five model scales \(N, S, M, L, X\), achieving AP gains of 1.1% to 1.7% with only a minimal increase in inference latency. Extensive visualization, ablation, and comparative studies validate the effectiveness of our approach, demonstrating its superior capability in discriminating heterogeneous objects with high efficiency.

中文摘要：单级目标检测，特别是YOLO系列，在精度和效率之间取得了良好的平衡。然而，现有的YOLO检测器缺乏对共享特征通道内异构对象响应的显式建模，这限制了进一步的性能提升。为了解决这个问题，我们提出了YOLO-DS，这是一个围绕新型双统计协同算子（DSO）构建的框架。DSO通过联合建模通道平均值和峰均差来解耦对象特征。基于DSO，我们设计了两个轻量级门控模块：用于自适应通道特征选择的双统计协同门控（DSG）模块和用于深度特征加权的多路径分段门控（MSG）模块。在MS-COCO基准测试中，YOLO-DS在五个模型尺度（N、S、M、L、X）上始终优于YOLOv8，实现了1.1%至1.7%的AP增益，而推理延迟仅略有增加。广泛的可视化、消融和比较研究验证了我们的方法的有效性，证明了它在高效区分异质物体方面的优越能力。


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2601.18172v1)

---


## The Latency Wall: Benchmarking Off\-the\-Shelf Emotion Recognition for Real\-Time Virtual Avatars / 延迟墙：实时虚拟化身现成情感识别的基准测试

发布日期：2026-01-22

作者：Yarin Benyamin

摘要：In the realm of Virtual Reality \(VR\) and Human\-Computer Interaction \(HCI\), real\-time emotion recognition shows promise for supporting individuals with Autism Spectrum Disorder \(ASD\) in improving social skills. This task requires a strict latency\-accuracy trade\-off, with motion\-to\-photon \(MTP\) latency kept below 140 ms to maintain contingency. However, most off\-the\-shelf Deep Learning models prioritize accuracy over the strict timing constraints of commodity hardware. As a first step toward accessible VR therapy, we benchmark State\-of\-the\-Art \(SOTA\) models for Zero\-Shot Facial Expression Recognition \(FER\) on virtual characters using the UIBVFED dataset. We evaluate Medium and Nano variants of YOLO \(v8, v11, and v12\) for face detection, alongside general\-purpose Vision Transformers including CLIP, SigLIP, and ViT\-FER.Our results on CPU\-only inference demonstrate that while face detection on stylized avatars is robust \(100% accuracy\), a "Latency Wall" exists in the classification stage. The YOLOv11n architecture offers the optimal balance for detection \(~54 ms\). However, general\-purpose Transformers like CLIP and SigLIP fail to achieve viable accuracy \(<23%\) or speed \(>150 ms\) for real\-time loops. This study highlights the necessity for lightweight, domain\-specific architectures to enable accessible, real\-time AI in therapeutic settings.

中文摘要：在虚拟现实（VR）和人机交互（HCI）领域，实时情绪识别显示出支持自闭症谱系障碍（ASD）患者提高社交技能的前景。这项任务需要严格的延迟精度权衡，将运动到光子（MTP）延迟保持在140毫秒以下，以保持偶然性。然而，大多数现成的深度学习模型将准确性置于商品硬件的严格时间限制之上。作为实现可访问虚拟现实治疗的第一步，我们使用UIBVFED数据集对虚拟角色上的零样本面部表情识别（FER）的启动状态（SOTA）模型进行了基准测试。我们评估了YOLO的Medium和Nano变体（v8、v11和v12）用于面部检测，以及包括CLIP、SigLIP和ViT FER在内的通用视觉转换器。我们在仅CPU推理上的结果表明，虽然风格化化身的面部检测是稳健的（100%准确率），但在分类阶段存在“延迟墙”。YOLOv11n架构为检测提供了最佳平衡（~54ms）。然而，像CLIP和SigLIP这样的通用变压器无法实现实时环路的可行精度（<23%）或速度（>150ms）。这项研究强调了轻量级、特定领域的架构的必要性，以在治疗环境中实现可访问的实时人工智能。


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2601.15914v1)

---

