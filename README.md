<div align="center">

# YOLO ArXiv Daily

[![Daily Papers](https://img.shields.io/badge/📅-每日更新-blue)]()
[![arXiv](https://img.shields.io/badge/arXiv-最新论文-red)](https://arxiv.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)

*自动追踪 YOLO 相关最新论文，提供中英文双语摘要*

</div>

---

## 📑 论文列表

> ### 1. TimeLens: On\-Device Artifact Recognition with Retrieval\-Augmented Question Answering for the Grand Egyptian Museum
> **🔹 中文标题：** TimeLens：应用于大埃及博物馆的设备端文物识别与检索增强问答技术
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-06-11 |
> | 👤 作者 | Rawan Hesham |
>
> **📄 英文摘要：**
> TimeLens is an AI\-powered bilingual mobile guide for the Grand Egyptian Museum \(GEM\). Pointing a phone at an exhibit, a visitor sees the artifact recognized in real time and can ask follow\-up questions answered in English or Arabic. The work addresses three problems specific to in\-gallery deployment: fine\-grained visual similarity among 51 catalogued artifacts \(many near\-identical Ramesside statues\), the gap between curated training data and handheld camera conditions, and the risk of an AI guide stating unsupported historical facts. Two engineering contributions are reported. First, an on\-device artifact detector was developed through a data\-quality\-driven iteration study \-\- from foundation\-model auto\-annotation \(YOLO\-World\), through spatial label\-cleaning rules, to a fully hand\-annotated dataset \-\- isolating label quality as the decisive factor: the final YOLOv8n model resolves every previously failing class while remaining a 5.97 MB TensorFlow Lite asset that runs in real time on a mid\-range phone \(mAP@0.5 = 0.995, mAP@0.5:0.95 = 0.924\). Second, a bilingual Retrieval\-Augmented Generation \(RAG\) guide, grounded in a 108\-record ChromaDB knowledge base, was benchmarked across seven candidate language models, with Gemma 4 E2B \(Q4 K M\) selected; ten targeted optimizations reduce end\-to\-end latency from over 30 s to approximately 10 s. Both subsystems are integrated in a production Flutter application with bilingual interface, museum location gating, and text\-to\-speech support.
>
> **📝 中文摘要：**
> TimeLens是为大埃及博物馆开发的人工智能双语移动导览系统。游客只需用手机对准展品，即可实时识别文物并提出后续问题，系统将提供英语或阿拉伯语的解答。该项目针对展厅部署的特殊性解决了三大问题：51件馆藏文物间的细粒度视觉相似性（包括许多高度相似的拉美西斯时期雕像）、策划训练数据与手持相机实景拍摄的差异，以及AI导览可能陈述未经证实历史事实的风险。

本研究报告了两项工程贡献：第一，通过数据质量驱动的迭代研究开发了文物识别检测器——从基础模型自动标注（YOLO-World）到空间标签清洗规则，最终形成完全手工标注的数据集——证明标签质量是决定性因素：最终的YOLOv8n模型成功识别了所有此前识别失败的类别，同时保持5.97MB的TensorFlow Lite轻量化体积，可在中端手机实时运行（mAP@0.5=0.995，mAP@0.5:0.95=0.924）。第二，构建了基于108条记录的ChromaDB知识库的双语检索增强生成导览系统，经七种候选语言模型基准测试后选用Gemma 4 E2B（Q4 K M）；通过十项针对性优化将端到端延迟从30余秒降至约10秒。

这两个子系统已整合至正式发布的Flutter应用中，具备双语界面、博物馆位置权限控制和文本转语音功能。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.13267v1)

---

> ### 2. YOLO\-AMC: An Improved YOLO Architecture with Attention Mechanisms for Building Crack Detection
> **🔹 中文标题：** YOLO-AMC：一种基于注意力机制的改进型YOLO建筑裂缝检测模型
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-06-11 |
> | 👤 作者 | Ching\-Yu Tsai |
>
> **📄 英文摘要：**
> Crack detection plays an important role in infrastructure inspection and Structural Health Monitoring \(SHM\). However, cracks typically appear as thin, low\-contrast structures and are easily affected by background noise, posing challenges for existing object detection models. This study proposes an improved YOLO\-based architecture with integrated attention mechanisms, termed YOLO\-AMC \(YOLO with Attention Mechanisms for Crack Detection\), to enhance automated crack detection performance. Based on YOLOv11, the original C2PSA module is removed, and multiple attention mechanisms, including Global Attention Mechanism \(GAM\), Residual Convolutional Block Attention Module \(Res\-CBAM\), and Shuffle Attention \(SA\), are introduced into the multi\-scale feature fusion layers of the Neck to strengthen cross\-scale feature integration. Experimental results demonstrate that YOLO\-AMC consistently outperforms baseline models YOLOv11n and YOLOv8n across multiple evaluation metrics. Among the evaluated attention modules, GAM achieves the best detection performance, obtaining mAP@0.5 = 0.9917 and mAP@0.5:0.95 = 0.9506 on the test dataset, which are higher than those of YOLOv11 \(0.9833 / 0.9112\) and YOLOv8 \(0.9707 / 0.8921\). Furthermore, while maintaining a computational complexity of 7.6 GFLOPs, the proposed model achieves 110.95 FPS on an NVIDIA RTX 4090 platform and approximately 5 FPS on a Raspberry Pi 5 edge device, demonstrating a favorable trade\-off between accuracy and deployment efficiency. The implementation code for this study is available on GitHub at https://github.com/CY\-Tsai24/YOLO\-AMC.
>
> **📝 中文摘要：**
> 裂缝检测在基础设施检测和结构健康监测中扮演着重要角色。然而，裂缝通常表现为细长、低对比度的结构，且易受背景噪声干扰，这对现有目标检测模型构成了挑战。本研究提出一种改进的YOLO架构，通过集成注意力机制构建裂缝检测专用模型——YOLO-AMC（集成注意力机制的YOLO裂缝检测模型），以提升自动化裂缝检测性能。该模型基于YOLOv11架构，移除了原有的C2PSA模块，并在Neck网络的多尺度特征融合层中引入全局注意力机制、残差卷积注意力模块及混洗注意力等多重注意力机制，从而增强跨尺度特征整合能力。实验结果表明，YOLO-AMC在多项评估指标上持续优于基准模型YOLOv11n和YOLOv8n。其中采用全局注意力机制的模型表现最优，在测试集上获得mAP@0.5 = 0.9917和mAP@0.5:0.95 = 0.9506，显著高于YOLOv11（0.9833 / 0.9112）和YOLOv8（0.9707 / 0.8921）的性能。在保持7.6 GFLOPs计算复杂度的情况下，该模型在NVIDIA RTX 4090平台上达到110.95 FPS，在Raspberry Pi 5边缘设备上实现约5 FPS的推理速度，展现出精度与部署效率的良好平衡。本研究代码已发布于GitHub平台：https://github.com/CY-Tsai24/YOLO-AMC。
>
> **💻 代码链接：** https://github.com/CY-Tsai24/YOLO-AMC.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.12958v1)

---

> ### 3. Performance Analysis of YOLOv11 and YOLOv8 for Mixed Traffic Object Detection under Adverse Weather Conditions in Developing Countries
> **🔹 中文标题：** YOLOv11与YOLOv8在发展中国家恶劣天气条件下混合交通目标检测的性能分析
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-06-10 |
> | 👤 作者 | Quoc Thuan Nguyen |
>
> **📄 英文摘要：**
> In modern vehicular systems, robust performance under harsh conditions has become a critical problem of autonomous driving. Our study delivers a comprehensive evaluation of the newest iteration of the YOLO series, which is YOLOv11 Nano architecture benchmarked against the widely adopted YOLOv8 Nano as a baseline on a custom fused dataset that combines the Indian Driving Dataset \(IDD\) \[1\] and Berkeley Deep Drive Dataset \(BDD100K\) \[2\]. We have analyzed the trade\-offs among detection accuracy, inference speed, and computational efficiency in high\-entropy scenarios involving dense mixed traffic, rain, and low\-light conditions. Specifically, YOLOv11n achieves a mean Average Precision \(mAP@50\) of 46.6%, with a notable 3.2% improvement in Precision over the baseline, effectively reducing false positives in cluttered scenes. Furthermore, the proposed model exhibits enhanced energy efficiency, requiring 22% fewer FLOPs \(6.3G vs. 8.1G\) while maintaining real\-time inference speed of 70.9 FPS on a Tesla T4 GPU, offering an optimal trade\-off for safety\-critical edge deployment.
>
> **📝 中文摘要：**
> 在现代车辆系统中，恶劣条件下的鲁棒性能已成为自动驾驶的关键挑战。本研究通过印度驾驶数据集(IDD)[1]与伯克利深度驾驶数据集(BDD100K)[2]构建的融合定制数据集，对YOLO系列最新迭代版本YOLOv11 Nano架构进行综合评估，并以广泛应用的YOLOv8 Nano作为基准模型。我们深入分析了在密集混合交通、降雨及弱光等高复杂场景中，检测精度、推理速度与计算效率之间的平衡关系。实验表明，YOLOv11n在50%交并比阈值下的平均精度(mAP@50)达到46.6%，相比基准模型精度提升3.2%，有效降低了复杂场景中的误检率。更值得关注的是，该模型在保持70.9 FPS实时推理速度（基于Tesla T4 GPU）的同时，计算量减少22%（6.3G FLOPs对比8.1G FLOPs），展现出卓越的能效比，为安全关键型边缘部署提供了优化解决方案。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.12066v1)

---

> ### 4. Patient\-Level Diagnosis of Acute Myeloid Leukemia via Deep Learning Analysis of Bone Marrow Smear
> **🔹 中文标题：** 基于骨髓涂片深度学习分析的急性髓系白血病患者水平诊断
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-06-09 |
> | 👤 作者 | Yuqi Ma |
>
> **📄 英文摘要：**
> Bone marrow smear review remains important for acute myeloid leukemia \(AML\) assessment, but manual single\-cell interpretation is labor\-intensive and patient\-level diagnosis requires aggregation of many cellular observations. We present a cell\-to\-patient deep learning pipeline for AML\-assisted diagnosis from bone marrow smear images. The study included 258 patients from six anonymized centers, including a main cohort of 169 patients from Centers 1\-3 and an external validation cohort of 89 patients from Centers 4\-6. A 16\-category cell annotation vocabulary was used to describe the global cellular composition, including granulocytic, monocytic, erythroid, lymphoid, eosinophilic, and other cells. Rather than identifying strict AML blasts or leukemic blasts, the model targets an expert\-defined composite category termed Composite Blast\-like Cells \(CBLC\), comprising N, N1, M, M1, R, R1, J, and J1 according to the project\-wide morphological standard. A fixed YOLO\-based segmentation module detected cells, predicted contours were matched to expert polygon annotations by contour IoU, and standardized single\-cell crops were generated. An EfficientNet\-B0 classifier was trained through a two\-stage GT\-to\-YOLO and YOLO\-to\-YOLO strategy with class\-imbalance correction, center\-border regularization, and morphology\-assisted supervision. Cell\-level predictions were aggregated into patient\-level CBLC ratios for AML\-oriented diagnostic support. The pipeline achieved stable internal validation and maintained external generalization, with ensemble weighted F1\-scores of 0.9076, 0.8696, and 0.9124 on Centers 4, 5, and 6, respectively.
>
> **📝 中文摘要：**
> 骨髓涂片检查对急性髓系白血病（AML）评估仍至关重要，但人工单细胞判读耗时费力，且患者级诊断需汇总大量细胞观察数据。本研究提出一种从骨髓涂片图像到AML辅助诊断的细胞-患者深度学习流程。研究纳入来自六家匿名中心的258例患者，其中1-3中心的169例患者构成主队列，4-6中心的89例患者构成外部验证队列。采用16类细胞注释词库描述全局细胞构成，涵盖粒细胞、单核细胞、红系细胞、淋巴细胞、嗜酸性粒细胞等细胞类型。模型未聚焦于严格意义上的AML原始细胞或白血病原始细胞，而是针对专家界定的复合类别——复合原始样细胞（CBLC），该类别根据项目统一形态学标准包含N、N1、M、M1、R、R1、J及J1等细胞类型。采用固定YOLO分割模块检测细胞，通过轮廓交并比匹配预测轮廓与专家多边形标注，并生成标准化单细胞图像。通过两阶段"GT→YOLO→YOLO"策略训练EfficientNet-B0分类器，结合类别不平衡校正、中心-边界正则化和形态学辅助监督。将细胞级预测汇总为患者级CBLC比例，为AML导向诊断提供支持。该流程在内部验证中表现稳定，且保持良好的外部泛化能力，在4、5、6中心的集成加权F1值分别为0.9076、0.8696和0.9124。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.10735v1)

---

> ### 5. Time\-frequency localization of bird calls in dense soundscapes
> **🔹 中文标题：** 密集声景中鸟鸣声的时频定位
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-06-09 |
> | 👤 作者 | Simen Hexeberg |
>
> **📄 英文摘要：**
> Passive acoustic monitoring enables large\-scale observation of wildlife, but most bioacoustic classifiers only predict species presence in a time window without localizing vocalizations precisely in time or frequency, limiting downstream analyses. We formulate bird vocalization detection as an object detection task on spectrograms and train YOLO11 models to localize bird calls in dense tropical soundscapes from Singapore. We additionally introduce an open\-source browser\-based annotation tool and propose Intersection over Minimum \(IoMin\), an evaluation metric that better handles ambiguous acoustic boundaries than standard IoU and is better suited to the problem at hand. The best YOLO model nearly doubles baseline performance on in\-distribution soundscapes from Singapore \(81.8% vs. 42.1% IoMin@50 F1\-score\) while still outperforming the baseline on unseen out\-of\-distribution recordings from Hawaii \(58.6% vs. 48.6%\). These results suggest that object detection frameworks are a promising approach to time\-frequency localization of animal vocalizations in complex soundscapes.
>
> **📝 中文摘要：**
> 被动声学监测能够实现大规模野生动物观测，但大多数生物声学分类器仅能在时间窗口内预测物种存在，无法在时间或频率上精确标定发声位置，这限制了后续分析。本研究将鸟鸣检测重构为频谱图上的目标检测任务，利用YOLO11模型在新加坡密集热带声景中定位鸟鸣。我们开发了开源的浏览器标注工具，并提出最小交并比评估指标——该指标相比传统交并比能更好处理模糊声学边界，更适用于此类问题。最佳YOLO模型在新加坡本土声景中性能较基线近倍增（81.8% vs. 42.1% IoMin@50 F1分数），同时在未接触的夏威夷异域录音中仍保持优势（58.6% vs. 48.6%）。实验表明，目标检测框架为复杂声景中动物发声的时频定位提供了创新解决方案。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.10407v1)

---

<div align="center">

*由 [YOLO-Arxiv-Daily](https://github.com/WangQvQ/YOLO-Arxiv-Daily) 自动生成*

</div>