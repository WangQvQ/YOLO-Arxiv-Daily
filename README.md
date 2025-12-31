# 每日从arXiv中获取最新YOLO相关论文


## Detection Fire in Camera RGB\-NIR / RGB-NIR摄像机火灾探测

发布日期：2025-12-29

作者：Nguyen Truong Khai

摘要：Improving the accuracy of fire detection using infrared night vision cameras remains a challenging task. Previous studies have reported strong performance with popular detection models. For example, YOLOv7 achieved an mAP50\-95 of 0.51 using an input image size of 640 x 1280, RT\-DETR reached an mAP50\-95 of 0.65 with an image size of 640 x 640, and YOLOv9 obtained an mAP50\-95 of 0.598 at the same resolution. Despite these results, limitations in dataset construction continue to cause issues, particularly the frequent misclassification of bright artificial lights as fire.   This report presents three main contributions: an additional NIR dataset, a two\-stage detection model, and Patched\-YOLO. First, to address data scarcity, we explore and apply various data augmentation strategies for both the NIR dataset and the classification dataset. Second, to improve night\-time fire detection accuracy while reducing false positives caused by artificial lights, we propose a two\-stage pipeline combining YOLOv11 and EfficientNetV2\-B0. The proposed approach achieves higher detection accuracy compared to previous methods, particularly for night\-time fire detection. Third, to improve fire detection in RGB images, especially for small and distant objects, we introduce Patched\-YOLO, which enhances the model's detection capability through patch\-based processing. Further details of these contributions are discussed in the following sections.

中文摘要：使用红外夜视摄像机提高火灾探测的准确性仍然是一项具有挑战性的任务。之前的研究报告了流行检测模型的强大性能。例如，YOLOv7使用640 x 1280的输入图像大小获得了0.51的mAP50-95，RT-DETR使用640 x 640的图像大小达到了0.65的mAP50-195，YOLOv9在相同分辨率下获得了0.598的mAP50-295。尽管有这些结果，数据集构建的局限性仍然会导致问题，特别是经常将明亮的人造光误分类为火灾。本报告介绍了三个主要贡献：一个额外的近红外数据集、一个两阶段检测模型和补丁YOLO。首先，为了解决数据稀缺问题，我们探索并应用了各种数据增强策略，用于近红外数据集和分类数据集。其次，为了提高夜间火灾探测的准确性，同时减少人造光引起的误报，我们提出了一种结合YOLOv11和EfficientNetV2-B0的两级管道。与之前的方法相比，所提出的方法实现了更高的检测精度，特别是对于夜间火灾检测。第三，为了改进RGB图像中的火灾检测，特别是对于小而遥远的物体，我们引入了补丁YOLO，通过基于补丁的处理增强了模型的检测能力。这些贡献的更多细节将在以下章节中讨论。


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2512.23594v1)

---


## YOLO\-Master: MOE\-Accelerated with Specialized Transformers for Enhanced Real\-time Detection / YOLO Master：通过专用变压器加速MOE，增强实时检测能力

发布日期：2025-12-29

作者：Xu Lin

摘要：Existing Real\-Time Object Detection \(RTOD\) methods commonly adopt YOLO\-like architectures for their favorable trade\-off between accuracy and speed. However, these models rely on static dense computation that applies uniform processing to all inputs, misallocating representational capacity and computational resources such as over\-allocating on trivial scenes while under\-serving complex ones. This mismatch results in both computational redundancy and suboptimal detection performance. To overcome this limitation, we propose YOLO\-Master, a novel YOLO\-like framework that introduces instance\-conditional adaptive computation for RTOD. This is achieved through a Efficient Sparse Mixture\-of\-Experts \(ES\-MoE\) block that dynamically allocates computational resources to each input according to its scene complexity. At its core, a lightweight dynamic routing network guides expert specialization during training through a diversity enhancing objective, encouraging complementary expertise among experts. Additionally, the routing network adaptively learns to activate only the most relevant experts, thereby improving detection performance while minimizing computational overhead during inference. Comprehensive experiments on five large\-scale benchmarks demonstrate the superiority of YOLO\-Master. On MS COCO, our model achieves 42.4% AP with 1.62ms latency, outperforming YOLOv13\-N by \+0.8% mAP and 17.8% faster inference. Notably, the gains are most pronounced on challenging dense scenes, while the model preserves efficiency on typical inputs and maintains real\-time inference speed. Code will be available.

中文摘要：现有的实时目标检测（RTOD）方法通常采用类似YOLO的架构，因为它们在精度和速度之间进行了良好的权衡。然而，这些模型依赖于静态密集计算，该计算对所有输入应用统一处理，错配表示能力和计算资源，例如在琐碎场景上过度分配，而在复杂场景上分配不足。这种不匹配导致计算冗余和次优检测性能。为了克服这一局限性，我们提出了YOLO Master，这是一种新的类似YOLO的框架，为RTOD引入了实例条件自适应计算。这是通过高效稀疏专家混合（ES-MoE）块实现的，该块根据场景复杂度动态地为每个输入分配计算资源。其核心是，一个轻量级的动态路由网络通过一个增强多样性的目标在培训期间指导专家专业化，鼓励专家之间的互补专业知识。此外，路由网络自适应地学习仅激活最相关的专家，从而提高检测性能，同时最大限度地减少推理过程中的计算开销。在五个大型基准上的综合实验证明了YOLO Master的优越性。在MS COCO上，我们的模型实现了42.4%的AP和1.62ms的延迟，比YOLOv13-N高出+0.8%的mAP和17.8%的推理速度。值得注意的是，在具有挑战性的密集场景中，收益最为明显，而该模型在典型输入上保持了效率，并保持了实时推理速度。代码将可用。


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2512.23273v1)

---


## YOLO\-IOD: Towards Real Time Incremental Object Detection / YOLO-IOD：迈向实时增量目标检测

发布日期：2025-12-28

作者：Shizhou Zhang

摘要：Current methods for incremental object detection \(IOD\) primarily rely on Faster R\-CNN or DETR series detectors; however, these approaches do not accommodate the real\-time YOLO detection frameworks. In this paper, we first identify three primary types of knowledge conflicts that contribute to catastrophic forgetting in YOLO\-based incremental detectors: foreground\-background confusion, parameter interference, and misaligned knowledge distillation. Subsequently, we introduce YOLO\-IOD, a real\-time Incremental Object Detection \(IOD\) framework that is constructed upon the pretrained YOLO\-World model, facilitating incremental learning via a stage\-wise parameter\-efficient fine\-tuning process. Specifically, YOLO\-IOD encompasses three principal components: 1\) Conflict\-Aware Pseudo\-Label Refinement \(CPR\), which mitigates the foreground\-background confusion by leveraging the confidence levels of pseudo labels and identifying potential objects relevant to future tasks. 2\) Importancebased Kernel Selection \(IKS\), which identifies and updates the pivotal convolution kernels pertinent to the current task during the current learning stage. 3\) Cross\-Stage Asymmetric Knowledge Distillation \(CAKD\), which addresses the misaligned knowledge distillation conflict by transmitting the features of the student target detector through the detection heads of both the previous and current teacher detectors, thereby facilitating asymmetric distillation between existing and newly introduced categories. We further introduce LoCo COCO, a more realistic benchmark that eliminates data leakage across stages. Experiments on both conventional and LoCo COCO benchmarks show that YOLO\-IOD achieves superior performance with minimal forgetting.

中文摘要：目前用于增量目标检测（IOD）的方法主要依赖于Faster R-CNN或DETR系列检测器；然而，这些方法不适应实时YOLO检测框架。在本文中，我们首先确定了基于YOLO的增量检测器中导致灾难性遗忘的三种主要知识冲突类型：前景背景混淆、参数干扰和未对齐的知识蒸馏。随后，我们介绍了YOLO-IOD，这是一个基于预训练的YOLO World模型构建的实时增量对象检测（IOD）框架，通过分阶段的参数高效微调过程促进增量学习。具体来说，YOLO-IOD包括三个主要组成部分：1）冲突感知伪标签细化（CPR），通过利用伪标签的置信度和识别与未来任务相关的潜在对象来减轻前景背景混淆。2）基于重要性的核选择（IKS），在当前学习阶段识别和更新与当前任务相关的关键卷积核。3）跨阶段非对称知识蒸馏（CAKD），通过将学生目标检测器的特征通过先前和当前教师检测器的检测头传输，从而促进现有和新引入类别之间的非对称蒸馏，解决了未对齐的知识蒸馏冲突。我们进一步介绍了LoCo COCO，这是一个更现实的基准，可以消除跨阶段的数据泄漏。在传统和LoCo COCO基准上的实验表明，YOLO-IOD在最小的遗忘下实现了卓越的性能。


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2512.22973v1)

---


## CLIP\-Joint\-Detect: End\-to\-End Joint Training of Object Detectors with Contrastive Vision\-Language Supervision / CLIP联合检测：具有对比视觉语言监督的目标检测器端到端联合训练

发布日期：2025-12-28

作者：Behnam Raoufi

摘要：Conventional object detectors rely on cross\-entropy classification, which can be vulnerable to class imbalance and label noise. We propose CLIP\-Joint\-Detect, a simple and detector\-agnostic framework that integrates CLIP\-style contrastive vision\-language supervision through end\-to\-end joint training. A lightweight parallel head projects region or grid features into the CLIP embedding space and aligns them with learnable class\-specific text embeddings via InfoNCE contrastive loss and an auxiliary cross\-entropy term, while all standard detection losses are optimized simultaneously. The approach applies seamlessly to both two\-stage and one\-stage architectures. We validate it on Pascal VOC 2007\+2012 using Faster R\-CNN and on the large\-scale MS COCO 2017 benchmark using modern YOLO detectors \(YOLOv11\), achieving consistent and substantial improvements while preserving real\-time inference speed. Extensive experiments and ablations demonstrate that joint optimization with learnable text embeddings markedly enhances closed\-set detection performance across diverse architectures and datasets.

中文摘要：传统的目标检测器依赖于交叉熵分类，这很容易受到类别不平衡和标签噪声的影响。我们提出了CLIP联合检测，这是一个简单且与检测器无关的框架，通过端到端的联合训练集成了CLIP风格的对比视觉语言监督。轻量级并行头将区域或网格特征投影到CLIP嵌入空间中，并通过InfoNCE对比损失和辅助交叉熵项将其与可学习的类特定文本嵌入对齐，同时同时优化所有标准检测损失。该方法无缝应用于两级和一级架构。我们使用Faster R-CNN在Pascal VOC 2007+2012上验证了它，并使用现代YOLO检测器（YOLOv11）在大规模MS COCO 2017基准上验证了其有效性，在保持实时推理速度的同时实现了一致和实质性的改进。大量的实验和消融表明，使用可学习文本嵌入的联合优化显著提高了不同架构和数据集的闭集检测性能。


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2512.22969v1)

---


## Comparative Analysis of Deep Learning Models for Perception in Autonomous Vehicles / 

发布日期：2025-12-25

作者：Jalal Khan

摘要：Recently, a plethora of machine learning \(ML\) and deep learning \(DL\) algorithms have been proposed to achieve the efficiency, safety, and reliability of autonomous vehicles \(AVs\). The AVs use a perception system to detect, localize, and identify other vehicles, pedestrians, and road signs to perform safe navigation and decision\-making. In this paper, we compare the performance of DL models, including YOLO\-NAS and YOLOv8, for a detection\-based perception task. We capture a custom dataset and experiment with both DL models using our custom dataset. Our analysis reveals that the YOLOv8s model saves 75% of training time compared to the YOLO\-NAS model. In addition, the YOLOv8s model \(83%\) outperforms the YOLO\-NAS model \(81%\) when the target is to achieve the highest object detection accuracy. These comparative analyses of these new emerging DL models will allow the relevant research community to understand the models' performance under real\-world use case scenarios.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2512.21673v1)

---

