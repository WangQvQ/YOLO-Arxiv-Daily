# 每日从arXiv中获取最新YOLO相关论文


## Robust and Annotation\-Free Wound Segmentation on Noisy Real\-World Pressure Ulcer Images: Towards Automated DESIGN\-R\textsuperscript\{\textregistered\} Assessment / 噪声真实世界压疮图像的鲁棒无注释伤口分割：迈向自动化的DESIGN-R\text上标{\textregistered}评估

发布日期：2025-05-29

作者：Yun\-Cheng Tsai

摘要：Purpose: Accurate wound segmentation is essential for automated DESIGN\-R scoring. However, existing models such as FUSegNet, which are trained primarily on foot ulcer datasets, often fail to generalize to wounds on other body sites.   Methods: We propose an annotation\-efficient pipeline that combines a lightweight YOLOv11n\-based detector with the pre\-trained FUSegNet segmentation model. Instead of relying on pixel\-level annotations or retraining for new anatomical regions, our method achieves robust performance using only 500 manually labeled bounding boxes. This zero fine\-tuning approach effectively bridges the domain gap and enables direct deployment across diverse wound types. This is an advance not previously demonstrated in the wound segmentation literature.   Results: Evaluated on three real\-world test sets spanning foot, sacral, and trochanter wounds, our YOLO plus FUSegNet pipeline improved mean IoU by 23 percentage points over vanilla FUSegNet and increased end\-to\-end DESIGN\-R size estimation accuracy from 71 percent to 94 percent \(see Table 3 for details\).   Conclusion: Our pipeline generalizes effectively across body sites without task\-specific fine\-tuning, demonstrating that minimal supervision, with 500 annotated ROIs, is sufficient for scalable, annotation\-light wound segmentation. This capability paves the way for real\-world DESIGN\-R automation, reducing reliance on pixel\-wise labeling, streamlining documentation workflows, and supporting objective and consistent wound scoring in clinical practice. We will publicly release the trained detector weights and configuration to promote reproducibility and facilitate downstream deployment.

中文摘要：目的：准确的伤口分割对于自动化DESIGN-R评分至关重要。然而，主要在足部溃疡数据集上训练的现有模型，如FUSegNet，往往无法推广到其他身体部位的伤口。方法：我们提出了一种注释高效的管道，该管道将基于YOLOv11n的轻量级检测器与预训练的FUSegNet分割模型相结合。我们的方法不依赖于像素级注释或重新训练新的解剖区域，只使用500个手动标记的边界框即可实现稳健的性能。这种零微调方法有效地弥合了领域差距，并实现了跨不同伤口类型的直接部署。这是伤口分割文献中以前没有证明的进步。结果：在跨越足部、骶骨和转子伤口的三个真实世界测试集上进行评估，我们的YOLO加FUSegNet管道将平均IoU比普通FUSegNet提高了23个百分点，并将端到端DESIGN-R尺寸估计精度从71%提高到94%（详见表3）。结论：我们的管道在没有特定任务微调的情况下有效地跨身体部位进行了推广，证明了500个注释ROI的最小监督足以进行可扩展的注释轻伤口分割。该功能为现实世界的DESIGN-R自动化铺平了道路，减少了对像素标签的依赖，简化了文档工作流程，并支持临床实践中客观一致的伤口评分。我们将公开发布经过培训的探测器重量和配置，以提高可重复性并促进下游部署。


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2505.23392v1)

---


## YOLO\-SPCI: Enhancing Remote Sensing Object Detection via Selective\-Perspective\-Class Integration / YOLO-SPIC：通过选择性透视类集成增强遥感目标检测

发布日期：2025-05-27

作者：Xinyuan Wang

摘要：Object detection in remote sensing imagery remains a challenging task due to extreme scale variation, dense object distributions, and cluttered backgrounds. While recent detectors such as YOLOv8 have shown promising results, their backbone architectures lack explicit mechanisms to guide multi\-scale feature refinement, limiting performance on high\-resolution aerial data. In this work, we propose YOLO\-SPCI, an attention\-enhanced detection framework that introduces a lightweight Selective\-Perspective\-Class Integration \(SPCI\) module to improve feature representation. The SPCI module integrates three components: a Selective Stream Gate \(SSG\) for adaptive regulation of global feature flow, a Perspective Fusion Module \(PFM\) for context\-aware multi\-scale integration, and a Class Discrimination Module \(CDM\) to enhance inter\-class separability. We embed two SPCI blocks into the P3 and P5 stages of the YOLOv8 backbone, enabling effective refinement while preserving compatibility with the original neck and head. Experiments on the NWPU VHR\-10 dataset demonstrate that YOLO\-SPCI achieves superior performance compared to state\-of\-the\-art detectors.

中文摘要：由于极端的尺度变化、密集的物体分布和杂乱的背景，遥感图像中的物体检测仍然是一项具有挑战性的任务。虽然YOLOv8等最近的探测器显示出了有希望的结果，但它们的骨干架构缺乏指导多尺度特征细化的明确机制，限制了高分辨率航空数据的性能。在这项工作中，我们提出了YOLO-SPIC，这是一种注意力增强的检测框架，它引入了一个轻量级的选择性透视类集成（SPCI）模块来改进特征表示。SPCI模块集成了三个组件：用于自适应调节全局特征流的选择性流门（SSG）、用于上下文感知多尺度集成的透视融合模块（PFM）和用于增强类间可分离性的类判别模块（CDM）。我们在YOLOv8主干的P3和P5级中嵌入了两个SPCI块，在保持与原始颈部和头部兼容性的同时实现了有效的改进。在NWPU VHR-10数据集上的实验表明，与最先进的探测器相比，YOLO-SPIC具有更优的性能。


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2505.21370v1)

---


## YOLO\-FireAD: Efficient Fire Detection via Attention\-Guided Inverted Residual Learning and Dual\-Pooling Feature Preservation / YOLO FireAD：通过注意力引导的逆残差学习和双池特征保存实现高效火灾检测

发布日期：2025-05-27

作者：Weichao Pan

摘要：Fire detection in dynamic environments faces continuous challenges, including the interference of illumination changes, many false detections or missed detections, and it is difficult to achieve both efficiency and accuracy. To address the problem of feature extraction limitation and information loss in the existing YOLO\-based models, this study propose You Only Look Once for Fire Detection with Attention\-guided Inverted Residual and Dual\-pooling Downscale Fusion \(YOLO\-FireAD\) with two core innovations: \(1\) Attention\-guided Inverted Residual Block \(AIR\) integrates hybrid channel\-spatial attention with inverted residuals to adaptively enhance fire features and suppress environmental noise; \(2\) Dual Pool Downscale Fusion Block \(DPDF\) preserves multi\-scale fire patterns through learnable fusion of max\-average pooling outputs, mitigating small\-fire detection failures. Extensive evaluation on two public datasets shows the efficient performance of our model. Our proposed model keeps the sum amount of parameters \(1.45M, 51.8% lower than YOLOv8n\) \(4.6G, 43.2% lower than YOLOv8n\), and mAP75 is higher than the mainstream real\-time object detection models YOLOv8n, YOL\-Ov9t, YOLOv10n, YOLO11n, YOLOv12n and other YOLOv8 variants 1.3\-5.5%.

中文摘要：动态环境中的火灾探测面临着持续的挑战，包括光照变化的干扰、许多误报或漏报，很难同时实现效率和准确性。为了解决现有基于YOLO的模型中特征提取限制和信息丢失的问题，本研究提出了“你只看一次”的火灾探测与注意力引导的逆残差和双池降尺度融合（YOLO FireAD），其核心创新有两个：（1）注意力引导的倒残差块（AIR）将混合通道空间注意力与逆残差相结合，以自适应地增强火灾特征并抑制环境噪声；（2）双池降尺度融合块（DPDF）通过最大平均池输出的可学习融合来保留多尺度火灾模式，减轻了小型火灾探测故障。对两个公共数据集的广泛评估表明了我们模型的有效性能。我们提出的模型保持了参数的总和（1.45M，比YOLOv8n低51.8%）（4.6G，比YOLON8n低43.2%），mAP75比主流实时目标检测模型YOLOv8n、YOL-Ov9t、YOLOv10n、YOLO11n、YOLOv12n和其他YOLOv8变体高1.3-5.5%。


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2505.20884v1)

---


## Knowledge Distillation Approach for SOS Fusion Staging: Towards Fully Automated Skeletal Maturity Assessment / SOS融合分期的知识蒸馏方法：迈向全自动骨骼成熟度评估

发布日期：2025-05-27

作者：Omid Halimi Milani

摘要：We introduce a novel deep learning framework for the automated staging of spheno\-occipital synchondrosis \(SOS\) fusion, a critical diagnostic marker in both orthodontics and forensic anthropology. Our approach leverages a dual\-model architecture wherein a teacher model, trained on manually cropped images, transfers its precise spatial understanding to a student model that operates on full, uncropped images. This knowledge distillation is facilitated by a newly formulated loss function that aligns spatial logits as well as incorporates gradient\-based attention spatial mapping, ensuring that the student model internalizes the anatomically relevant features without relying on external cropping or YOLO\-based segmentation. By leveraging expert\-curated data and feedback at each step, our framework attains robust diagnostic accuracy, culminating in a clinically viable end\-to\-end pipeline. This streamlined approach obviates the need for additional pre\-processing tools and accelerates deployment, thereby enhancing both the efficiency and consistency of skeletal maturation assessment in diverse clinical settings.

中文摘要：我们介绍了一种新的深度学习框架，用于蝶枕软骨融合（SOS）的自动分期，SOS是正畸学和法医人类学中的关键诊断标志。我们的方法利用了双模型架构，其中教师模型在手动裁剪的图像上训练，将其精确的空间理解转移到对完整、未裁剪图像进行操作的学生模型上。这种知识提炼是由一个新制定的损失函数促进的，该函数对齐了空间逻辑，并结合了基于梯度的注意力空间映射，确保学生模型内化了解剖学上相关的特征，而不依赖于外部裁剪或基于YOLO的分割。通过在每个步骤中利用专家策划的数据和反馈，我们的框架实现了强大的诊断准确性，最终形成了一个临床可行的端到端管道。这种简化的方法消除了对额外预处理工具的需求，并加速了部署，从而提高了不同临床环境中骨骼成熟评估的效率和一致性。


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2505.21561v1)

---


## Distill CLIP \(DCLIP\): Enhancing Image\-Text Retrieval via Cross\-Modal Transformer Distillation / Distill CLIP（DCLIP）：通过跨模态变换提取增强图像文本检索

发布日期：2025-05-25

作者：Daniel Csizmadia

摘要：We present Distill CLIP \(DCLIP\), a fine\-tuned variant of the CLIP model that enhances multimodal image\-text retrieval while preserving the original model's strong zero\-shot classification capabilities. CLIP models are typically constrained by fixed image resolutions and limited context, which can hinder their effectiveness in retrieval tasks that require fine\-grained cross\-modal understanding. DCLIP addresses these challenges through a meta teacher\-student distillation framework, where a cross\-modal transformer teacher is fine\-tuned to produce enriched embeddings via bidirectional cross\-attention between YOLO\-extracted image regions and corresponding textual spans. These semantically and spatially aligned global representations guide the training of a lightweight student model using a hybrid loss that combines contrastive learning and cosine similarity objectives. Despite being trained on only ~67,500 samples curated from MSCOCO, Flickr30k, and Conceptual Captions\-just a fraction of CLIP's original dataset\-DCLIP significantly improves image\-text retrieval metrics \(Recall@K, MAP\), while retaining approximately 94% of CLIP's zero\-shot classification performance. These results demonstrate that DCLIP effectively mitigates the trade\-off between task specialization and generalization, offering a resource\-efficient, domain\-adaptive, and detail\-sensitive solution for advanced vision\-language tasks. Code available at https://anonymous.4open.science/r/DCLIP\-B772/README.md.

中文摘要：


代码链接：https://anonymous.4open.science/r/DCLIP-B772/README.md.

论文链接：[阅读更多](http://arxiv.org/abs/2505.21549v2)

---

