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
> **🔹 中文标题：** TimeLens：面向大埃及博物馆的设备端制品识别与检索增强型问答系统
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
> TimeLens 是一款面向大英博物馆（GEM）的AI双语移动导览应用。参观者将手机对准展品时，可实时识别文物并能以英语或阿拉伯语获取后续问题的解答。该研究针对展厅部署的三个特定问题：51件登记文物间的细粒度视觉相似性（包括许多几乎相同的拉美西斯雕像）、策划训练数据集与实际拍摄条件间的差异、以及AI导览生成未经证实历史事实的风险。文章报告了两项工程贡献：首先，通过数据质量驱动的迭代研究开发了端侧文物检测器——从基础模型自动标注（YOLO-World）到空间标签清洗规则，最终形成全手工标注数据集——确定标签质量为决定性因素：最终的YOLOv8n模型在保持5.97MB TensorFlow Lite资产体积、可在中端手机实时运行的前提下（mAP@0.5=0.995, mAP@0.5:0.95=0.924），成功解决了此前所有失效类别。其次，构建了基于108条ChromaDB知识库的双语检索增强生成导览系统，通过七种候选语言模型的基准测试选定Gemma 4 E2B（Q4 K M），并通过十项定向优化将端到端延迟从30秒以上缩短至约10秒。两个子系统已集成至具备双语界面、博物馆定位门控和文本转语音功能的生产级Flutter应用中。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.13267v1)

---

> ### 2. YOLO\-AMC: An Improved YOLO Architecture with Attention Mechanisms for Building Crack Detection
> **🔹 中文标题：** YOLO-AMC：一种引入注意力机制的改进型YOLO建筑裂缝检测架构
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
> 裂缝检测在基础设施巡检与结构健康监测中扮演着重要角色。然而，裂缝通常呈现为细长、低对比度的结构，且易受背景噪声干扰，这对现有目标检测模型提出了挑战。本研究提出一种基于YOLO改进的集成注意力机制架构，命名为YOLO-AMC（融合注意力机制的裂缝检测YOLO模型），以提升自动化裂缝检测性能。该模型以YOLOv11为基础架构，移除原始C2PSA模块，并在Neck网络的多尺度特征融合层中引入全局注意力机制、残差卷积注意力模块和随机注意力机制等多种注意力组件，强化跨尺度特征整合能力。实验结果表明，YOLO-AMC在多维度评估指标上均优于基线模型YOLOv11n与YOLOv8n。在对比的注意力模块中，全局注意力机制取得最优检测性能，在测试集上实现mAP@0.5=0.9917与mAP@0.5:0.95=0.9506，显著优于YOLOv11（0.9833/0.9112）与YOLOv8（0.9707/0.8921）。在保持7.6 GFLOPs计算复杂度的同时，所提模型在NVIDIA RTX 4090平台上达到110.95 FPS推理速度，在树莓派5边缘设备上亦可实现约5 FPS，体现了精度与部署效率的良好平衡。本研究实现代码已开源于GitHub：https://github.com/CY-Tsai24/YOLO-AMC。
>
> **💻 代码链接：** https://github.com/CY-Tsai24/YOLO-AMC.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.12958v1)

---

> ### 3. Performance Analysis of YOLOv11 and YOLOv8 for Mixed Traffic Object Detection under Adverse Weather Conditions in Developing Countries
> **🔹 中文标题：** 关于恶劣天气条件下发展中国家混合交通目标检测的YOLOv11与YOLOv8性能分析
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
> 在现代车辆系统中，恶劣条件下的鲁棒性能已成为自动驾驶领域的关键挑战。本研究对YOLO系列最新迭代版本YOLOv11 Nano架构进行了全面评估，以广泛采用的YOLOv8 Nano为基准，在融合印度驾驶数据集（IDD）[1]与伯克利深度驾驶数据集（BDD100K）[2]的定制混合数据集上进行对比分析。我们深入研究了模型在复杂混合交通、雨雾及低照度等高熵场景中检测精度、推理速度与计算效率间的权衡关系。实验表明，YOLOv11 Nano在平均精度（mAP@50）达到46.6%的同时，精确率较基准模型提升3.2%，有效降低了杂乱场景中的误检率。此外，该模型展现出卓越的能效优势，仅需6.3G FLOPs（较基准8.1G减少22%），即可在Tesla T4 GPU上实现70.9 FPS的实时推理速度，为安全关键型边缘部署提供了最优的性能平衡方案。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.12066v1)

---

> ### 4. Patient\-Level Diagnosis of Acute Myeloid Leukemia via Deep Learning Analysis of Bone Marrow Smear
> **🔹 中文标题：** 基于骨髓涂片深度学习分析的急性髓系白血病患者级诊断
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
> 骨髓涂片审查对于急性髓系白血病的评估至关重要，但人工单细胞判读费时费力，且患者级诊断需要综合大量细胞观察结果。本文提出一种基于骨髓涂片图像的细胞-患者深度学习流程，用于AML辅助诊断。研究纳入来自六个匿名中心的258例患者，包括来自1-3号中心的主要队列（169例）和来自4-6号中心的外部验证队列（89例）。采用16类细胞标注词典描述整体细胞构成，涵盖粒系、单核系、红系、淋巴系、嗜酸系及其他细胞类别。模型并非针对传统AML原始细胞或白血病原始细胞，而是聚焦于专家定义的复合类别——复合型原始样细胞，依据项目统一形态学标准包含N、N1、M、M1、R、R1、J、J1等类型。通过固定式YOLO分割模块检测细胞，利用轮廓IoU匹配预测轮廓与专家多边形标注，并生成标准化单细胞图像。采用双阶段GT-to-YOLO与YOLO-to-YOLO策略训练EfficientNet-B0分类器，结合类别不平衡校正、中心-边界正则化和形态学辅助监督。细胞级预测结果经聚合转化为患者级CBLC比例，为AML导向诊断提供支持。该流程在内部验证中表现稳定，并在外部队列中保持泛化能力，在4、5、6号中心的集成加权F1值分别为0.9076、0.8696和0.9124。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.10735v1)

---

> ### 5. Time\-frequency localization of bird calls in dense soundscapes
> **🔹 中文标题：** 密集声景中鸟类叫声的时频定位
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
> 被动声学监测能够实现野生动物的大规模观测，但大多数生物声学分类器仅能预测物种在特定时间段内的存在性，无法精确定位发声事件在时域或频域中的位置，从而限制了后续分析。我们将鸟鸣检测构建为频谱图上的目标检测任务，训练YOLO11模型在新加坡密集的热带声景中定位鸟鸣。此外，我们开发了开源浏览器端标注工具，并提出最小交并比（IoMin）评估指标——相较于标准交并比（IoU），该指标能更好地处理模糊的声学边界，更契合实际问题需求。最优YOLO模型在新加坡本地声景中的基线性能近乎倍增（81.8% vs. 42.1% IoMin@50 F1值），同时在夏威夷未参与训练的异地录音中仍优于基线（58.6% vs. 48.6%）。结果表明，目标检测框架为复杂声景中动物发声的时频定位提供了极具潜力的技术路径。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.10407v1)

---

<div align="center">

*由 [YOLO-Arxiv-Daily](https://github.com/WangQvQ/YOLO-Arxiv-Daily) 自动生成*

</div>