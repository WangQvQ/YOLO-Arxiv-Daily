# 每日从arXiv中获取最新YOLO相关论文


## maxVSTAR: Maximally Adaptive Vision\-Guided CSI Sensing with Closed\-Loop Edge Model Adaptation for Robust Human Activity Recognition / maxVSTAR：具有闭环边缘模型自适应的最大自适应视觉引导CSI传感，用于鲁棒的人类活动识别

发布日期：2025-10-30

作者：Kexing Liu

摘要：WiFi Channel State Information \(CSI\)\-based human activity recognition \(HAR\) provides a privacy\-preserving, device\-free sensing solution for smart environments. However, its deployment on edge devices is severely constrained by domain shift, where recognition performance deteriorates under varying environmental and hardware conditions. This study presents maxVSTAR \(maximally adaptive Vision\-guided Sensing Technology for Activity Recognition\), a closed\-loop, vision\-guided model adaptation framework that autonomously mitigates domain shift for edge\-deployed CSI sensing systems. The proposed system integrates a cross\-modal teacher\-student architecture, where a high\-accuracy YOLO\-based vision model serves as a dynamic supervisory signal, delivering real\-time activity labels for the CSI data stream. These labels enable autonomous, online fine\-tuning of a lightweight CSI\-based HAR model, termed Sensing Technology for Activity Recognition \(STAR\), directly at the edge. This closed\-loop retraining mechanism allows STAR to continuously adapt to environmental changes without manual intervention. Extensive experiments demonstrate the effectiveness of maxVSTAR. When deployed on uncalibrated hardware, the baseline STAR model's recognition accuracy declined from 93.52% to 49.14%. Following a single vision\-guided adaptation cycle, maxVSTAR restored the accuracy to 81.51%. These results confirm the system's capacity for dynamic, self\-supervised model adaptation in privacy\-conscious IoT environments, establishing a scalable and practical paradigm for long\-term autonomous HAR using CSI sensing at the network edge.

中文摘要：基于WiFi信道状态信息（CSI）的人类活动识别（HAR）为智能环境提供了一种隐私保护、无需设备的传感解决方案。然而，它在边缘设备上的部署受到域转移的严重限制，在不同的环境和硬件条件下，识别性能会恶化。本研究提出了maxVSTAR（用于活动识别的最大自适应视觉引导传感技术），这是一种闭环视觉引导模型自适应框架，可自主缓解边缘部署的CSI传感系统的域偏移。所提出的系统集成了跨模式师生架构，其中基于YOLO的高精度视觉模型作为动态监控信号，为CSI数据流提供实时活动标签。这些标签能够直接在边缘对基于CSI的轻量级HAR模型（称为活动识别传感技术（STAR））进行自主的在线微调。这种闭环再培训机制使STAR能够在没有人工干预的情况下不断适应环境变化。大量实验证明了maxVSTAR的有效性。当部署在未校准的硬件上时，基线STAR模型的识别准确率从93.52%下降到49.14%。经过一个视觉引导的适应周期后，maxVSTAR将准确率恢复到81.51%。这些结果证实了该系统在隐私意识的物联网环境中动态、自我监督的模型适应能力，为在网络边缘使用CSI传感的长期自主HAR建立了一个可扩展和实用的范式。


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2510.26146v1)

---


## DINO\-YOLO: Self\-Supervised Pre\-training for Data\-Efficient Object Detection in Civil Engineering Applications / DINO-YOLO：土木工程应用中数据高效目标检测的自我监督预训练

发布日期：2025-10-29

作者：Malaisree P

摘要：Object detection in civil engineering applications is constrained by limited annotated data in specialized domains. We introduce DINO\-YOLO, a hybrid architecture combining YOLOv12 with DINOv3 self\-supervised vision transformers for data\-efficient detection. DINOv3 features are strategically integrated at two locations: input preprocessing \(P0\) and mid\-backbone enhancement \(P3\). Experimental validation demonstrates substantial improvements: Tunnel Segment Crack detection \(648 images\) achieves 12.4% improvement, Construction PPE \(1K images\) gains 13.7%, and KITTI \(7K images\) shows 88.6% improvement, while maintaining real\-time inference \(30\-47 FPS\). Systematic ablation across five YOLO scales and nine DINOv3 variants reveals that Medium\-scale architectures achieve optimal performance with DualP0P3 integration \(55.77% mAP@0.5\), while Small\-scale requires Triple Integration \(53.63%\). The 2\-4x inference overhead \(21\-33ms versus 8\-16ms baseline\) remains acceptable for field deployment on NVIDIA RTX 5090. DINO\-YOLO establishes state\-of\-the\-art performance for civil engineering datasets \(<10K images\) while preserving computational efficiency, providing practical solutions for construction safety monitoring and infrastructure inspection in data\-constrained environments.

中文摘要：土木工程应用中的目标检测受到专业领域中有限注释数据的限制。我们介绍了DINO-YOLO，这是一种混合架构，将YOLOv12与DINOv3自监督视觉变换器相结合，用于数据高效检测。DINOv3功能在两个位置进行了战略性集成：输入预处理（P0）和中间骨干增强（P3）。实验验证证明了显著的改进：隧道段裂缝检测（648张图像）实现了12.4%的改进，施工PPE（1K图像）提高了13.7%，KITTI（7K图像）显示了88.6%的改进，同时保持了实时推理（30-47FPS）。跨五个YOLO量表和九个DINOv3变体的系统消融表明，中等规模架构通过DualP0P3集成实现了最佳性能（55.77%mAP@0.5)，而小规模需要三重整合（53.63%）。2-4x的推理开销（21-33ms对8-16ms基线）对于NVIDIA RTX 5090上的现场部署来说仍然是可以接受的。DINO-YOLO为土木工程数据集（<10K图像）建立了最先进的性能，同时保持了计算效率，为数据受限环境中的建筑安全监测和基础设施检查提供了实用的解决方案。


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2510.25140v1)

---


## Delving into Cascaded Instability: A Lipschitz Continuity View on Image Restoration and Object Detection Synergy / 深入研究级联不稳定性：图像恢复和目标检测协同的Lipschitz连续性观点

发布日期：2025-10-28

作者：Qing Zhao

摘要：To improve detection robustness in adverse conditions \(e.g., haze and low light\), image restoration is commonly applied as a pre\-processing step to enhance image quality for the detector. However, the functional mismatch between restoration and detection networks can introduce instability and hinder effective integration \-\- an issue that remains underexplored. We revisit this limitation through the lens of Lipschitz continuity, analyzing the functional differences between restoration and detection networks in both the input space and the parameter space. Our analysis shows that restoration networks perform smooth, continuous transformations, while object detectors operate with discontinuous decision boundaries, making them highly sensitive to minor perturbations. This mismatch introduces instability in traditional cascade frameworks, where even imperceptible noise from restoration is amplified during detection, disrupting gradient flow and hindering optimization. To address this, we propose Lipschitz\-regularized object detection \(LROD\), a simple yet effective framework that integrates image restoration directly into the detector's feature learning, harmonizing the Lipschitz continuity of both tasks during training. We implement this framework as Lipschitz\-regularized YOLO \(LR\-YOLO\), extending seamlessly to existing YOLO detectors. Extensive experiments on haze and low\-light benchmarks demonstrate that LR\-YOLO consistently improves detection stability, optimization smoothness, and overall accuracy.

中文摘要：为了提高在不利条件下（如雾霾和低光照）的检测鲁棒性，图像恢复通常被用作预处理步骤，以提高检测器的图像质量。然而，恢复和检测网络之间的功能不匹配可能会引入不稳定性并阻碍有效的集成，这是一个尚未得到充分探索的问题。我们通过Lipschitz连续性的视角重新审视了这一局限性，分析了恢复和检测网络在输入空间和参数空间中的功能差异。我们的分析表明，恢复网络执行平滑、连续的变换，而对象检测器在不连续的决策边界下运行，使其对微小扰动高度敏感。这种不匹配在传统级联框架中引入了不稳定性，在检测过程中，即使是来自恢复的不可察觉的噪声也会被放大，扰乱梯度流并阻碍优化。为了解决这个问题，我们提出了Lipschitz正则化对象检测（LROD），这是一个简单而有效的框架，将图像恢复直接集成到检测器的特征学习中，在训练过程中协调两个任务的Lipschitz连续性。我们将此框架实现为Lipschitz正则化YOLO（LR-YOLO），无缝扩展到现有的YOLO检测器。对雾度和低光基准的广泛实验表明，LR-YOLO始终如一地提高了检测稳定性、优化平滑度和整体准确性。


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2510.24232v1)

---


## CSST Slitless Spectra: Target Detection and Classification with YOLO / CSST无缝光谱：使用YOLO进行目标检测和分类

发布日期：2025-10-28

作者：Yingying Zhou

摘要：Addressing the spatial uncertainty and spectral blending challenges in CSST slitless spectroscopy, we present a deep learning\-driven, end\-to\-end framework based on the You Only Look Once \(YOLO\) models. This approach directly detects, classifies, and analyzes spectral traces from raw 2D images, bypassing traditional, error\-accumulating pipelines. YOLOv5 effectively detects both compact zero\-order and extended first\-order traces even in highly crowded fields. Building on this, YOLO11 integrates source classification \(star/galaxy\) and discrete astrophysical parameter estimation \(e.g., redshift bins\), showcasing complete spectral trace analysis without other manual preprocessing. Our framework processes large images rapidly, learning spectral\-spatial features holistically to minimize errors. We achieve high trace detection precision \(YOLOv5\) and demonstrate successful quasar identification and binned redshift estimation \(YOLO11\). This study establishes machine learning as a paradigm shift in slitless spectroscopy, unifying detection, classification, and preliminary parameter estimation in a scalable system. Future research will concentrate on direct, continuous prediction of astrophysical parameters from raw spectral traces.

中文摘要：针对CSST无狭缝光谱学中的空间不确定性和光谱混合挑战，我们提出了一种基于You Only Look Once（YOLO）模型的深度学习驱动的端到端框架。这种方法直接检测、分类和分析原始2D图像中的光谱痕迹，绕过传统的误差累积管道。YOLOv5即使在高度拥挤的场中也能有效地检测紧凑的零阶和扩展的一阶迹线。在此基础上，YOLO11集成了源分类（恒星/星系）和离散天体物理参数估计（例如红移箱），展示了完整的光谱轨迹分析，而无需其他手动预处理。我们的框架可以快速处理大型图像，全面学习光谱空间特征以最大限度地减少误差。我们实现了高跟踪检测精度（YOLOv5），并成功地进行了类星体识别和分箱红移估计（YOLO11）。本研究将机器学习确立为无狭缝光谱学的范式转变，在可扩展的系统中统一检测、分类和初步参数估计。未来的研究将集中于从原始光谱轨迹中直接、连续地预测天体物理参数。


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2510.24087v1)

---


## hYOLO Model: Enhancing Object Classification with Hierarchical Context in YOLOv8 / hYOLO模型：在YOLOv8中利用层次上下文增强对象分类

发布日期：2025-10-27

作者：Veska Tsenkova

摘要：Current convolution neural network \(CNN\) classification methods are predominantly focused on flat classification which aims solely to identify a specified object within an image. However, real\-world objects often possess a natural hierarchical organization that can significantly help classification tasks. Capturing the presence of relations between objects enables better contextual understanding as well as control over the severity of mistakes. Considering these aspects, this paper proposes an end\-to\-end hierarchical model for image detection and classification built upon the YOLO model family. A novel hierarchical architecture, a modified loss function, and a performance metric tailored to the hierarchical nature of the model are introduced. The proposed model is trained and evaluated on two different hierarchical categorizations of the same dataset: a systematic categorization that disregards visual similarities between objects and a categorization accounting for common visual characteristics across classes. The results illustrate how the suggested methodology addresses the inherent hierarchical structure present in real\-world objects, which conventional flat classification algorithms often overlook.

中文摘要：当前的卷积神经网络（CNN）分类方法主要侧重于平面分类，其目的仅在于识别图像中的指定对象。然而，现实世界的对象通常具有一个自然的层次结构，可以显著帮助分类任务。捕捉对象之间关系的存在可以更好地理解上下文，并控制错误的严重程度。考虑到这些方面，本文提出了一种基于YOLO模型族的端到端分层图像检测和分类模型。引入了一种新的分层架构、一种改进的损失函数和一种针对模型分层性质量身定制的性能指标。所提出的模型在同一数据集的两个不同层次分类上进行训练和评估：一个是忽略对象之间视觉相似性的系统分类，另一个是考虑类间共同视觉特征的分类。结果说明了所提出的方法如何解决现实世界对象中存在的固有层次结构，而传统的平面分类算法往往忽略了这一点。


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2510.23278v1)

---

