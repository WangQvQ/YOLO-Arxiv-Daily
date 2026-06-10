<div align="center">

# YOLO ArXiv Daily

[![Daily Papers](https://img.shields.io/badge/📅-每日更新-blue)]()
[![arXiv](https://img.shields.io/badge/arXiv-最新论文-red)](https://arxiv.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)

*自动追踪 YOLO 相关最新论文，提供中英文双语摘要*

</div>

---

## 📑 论文列表

> ### 1. Feasibility to detect rapid change and disappearance of seagrass: Lessons from nearly 80 years of vegetation change in the Ako, Seto Inland Sea, Japan
> **🔹 中文标题：** 探测海草快速变化与消亡的可行性：来自日本濑户内海Ako地区近80年植被变迁的启示
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-06-06 |
> | 👤 作者 | Takehisa Yamakita |
>
> **📄 英文摘要：**
> This study analyses the Ako tidal flat in the Seto Inland Sea, Japan, where nearly all Zostera marina disappeared within a single year in 2025. Using aerial photographs from the 1940s onward, high\-resolution satellite imagery, GRUS images \(2.5\-5 m\), and monthly Sentinel\-2 composites \(10 m\), we reconstructed approximately 80 years of seagrass distribution. YOLO\-based segmentation using deep learning achieved high accuracy \(overall accuracy >= 0.9\) across these datasets; although species could not be discriminated, the models captured the major temporal dynamics in vegetation area. The long\-term mean seagrass area was 6.8 ha, but values fluctuated widely, from 3.5 ha in 1974 to 41.3 ha in 1989 except 0.2 ha in 2025. Sentinel\-2 composites from 2019 to 2026 revealed clear seasonality, with vegetation increasing in early summer and declining from autumn. In 2025, however, the area decreased sharply after summer and remained anomalously low throughout the winter of 2025\-2026. Our results, indicating that the 2025 event was not a normal fluctuation but a rapid ecosystem shift involving the loss of the dominant canopy\-forming species, most plausibly driven by regionally elevated summer water temperatures. The findings also have implications for seagrass Essential Ocean Variables \(EOVs\) and the State of Nature \(SoN\) metrics used in TNFD\-aligned nature\-related disclosures. Unlike forests, seagrass meadows require finer temporal resolution because both pronounced seasonality and abrupt collapse strongly influence area\-based indicators. Therefore, in addition to previously noted issues such as species\-level classification accuracy, we recommend that \(1\) baselines be defined over the longest available record and justified ecologically, \(2\) seasonal standardization be applied before inter\-annual comparisons, and \(3\) years with extreme area anomalies be flagged rather than used as reference points.
>
> **📝 中文摘要：**
> 本研究分析了日本濑户内海英虞湾潮间带的大叶藻消失现象——该区域近95%的大叶藻在2025年内完全消失。通过融合20世纪40年代以来的航拍照片、高分辨率卫星影像、GRUS影像（分辨率2.5-5米）及月度哨兵2号合成影像（分辨率10米），我们重建了约80年的海草分布演变。基于深度学习的YOLO目标检测算法在所有数据集中均实现高精度识别（总体精度≥0.9），虽无法区分海草种类，但能有效捕捉植被面积的时空动态。研究发现该区域海草长期平均面积为6.8公顷，但年际波动剧烈：从1974年的3.5公顷到1989年的41.3公顷，直至2025年骤降至0.2公顷。哨兵2号数据显示2019-2026年间存在显著季节性规律——初夏植被扩张，秋季开始衰退。然而2025年夏季后面积锐减，且在2025-2026年冬季持续维持异常低值。结果表明2025年的突变并非正常生态波动，而是以冠层优势种消失为特征的快速生态系统转型，最可能驱动因素是区域性夏季水温异常升高。该发现对与自然相关财务披露（TNFD）框架下的海草海洋关键变量（EOVs）和自然状况（SoN）指标具有重要启示：与森林生态系统不同，海草草甸需更高时间分辨率监测，因为显著的季节性特征和突发性衰退均会强烈影响基于面积的评估指标。因此，除物种分类精度等已知问题外，我们建议：(1)基于可获取的最长历史记录定义生态基准线并提供生态学依据；(2)年度比较前需进行季节性标准化处理；(3)标记极端异常年份而非将其用作参考基准。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.07949v1)

---

> ### 2. Attention\-Guided Autoencoder Fusion for Insulator Defect Detection Using UAV Transmission\-Line Imaging
> **🔹 中文标题：** 基于无人机输电线路成像的注意力引导自编码器融合绝缘子缺陷检测方法
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
> 高压输电线路绝缘子的自动缺陷检测仍具挑战性，原因在于无人机图像中存在严重的类别不平衡、巨大的尺度变化以及缺陷实例空间占比小等问题。为应对这些挑战，本文提出AE-YOLO——一种注意力引导的自编码器增强型YOLO框架，用于实现鲁棒的绝缘子缺陷检测。该架构在特征金字塔网络-路径聚合网络（FPN-PAN）颈部结构中整合了轻量级瓶颈自编码器，在多尺度特征融合过程中保留了异常敏感信息。主干网络全程采用卷积块注意力模块（CBAM），增强了特征判别能力并抑制背景干扰。框架还引入了最大化方差自编码器正则化策略，以促进多样化且具有缺陷判别性的潜在表征。网络采用统一目标函数进行训练，结合焦点损失、完整交并比损失与自编码器正则化，以解决前景-背景不平衡问题并提升定位精度。推理阶段采用加权框融合方法，整合来自YOLOv8、YOLOv10和YOLO11的预测结果。自编码器引导的置信度增强机制提升了模型对稀有缺陷类别的敏感性。绝缘子缺陷检测数据集上的实验表明，采用EfficientNetV2主干网络的AE-YOLO在0.5交并比阈值下达到95.10%的mAP值、96.40%的精确率和93.80%的召回率。该性能在0.5交并比阈值下mAP超越最强YOLO系列基线5.0个百分点，召回率提升6.7个百分点。这些结果证实了该框架的有效性与适应性，为无人机输电线路巡检与缺陷监测提供了实用且可扩展的解决方案。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.06536v1)

---

> ### 3. PereStruct: Multimodal Semantic Assembly for Robust Historical Document Parsing
> **🔹 中文标题：** PereStruct：面向鲁棒历史文档解析的多模态语义组装方法
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-06-03 |
> | 👤 作者 | Maksim Shandybo |
>
> **📄 英文摘要：**
> Parsing historical documents with complex, non\-standard layouts remains a fundamental bottleneck in large\-scale archival digitization. Unlike modern typography, historical newspapers exhibit severe physical degradation and highly irregular page structures that confound even state\-of\-the\-art vision\-language models, presenting severe out\-of\-distribution challenges. We address this gap with an automated pipeline specifically designed for parsing historical newspapers, documents characterized by particularly intricate multi\-column layouts. Our approach combines a fine\-tuned YOLO architecture for layout analysis and block detection, trained on 1,426 fully human\-annotated scanned pages, with a novel semantic assembly module that reconstructs articles by jointly modeling lexical\-semantic similarity via TF\-IDF, visual embeddings from our fine\-tuned YOLO, and geometric layout constraints. This multi\-modal integration yields state\-of\-the\-art performance, achieving an F1 score of 0.904 on block\-to\-article mapping. Notably, end\-to\-end evaluation against vision\-language models \(Qwen3.6\-35B\-A3B and Qwen3.6\-Plus\) demonstrates that PereStruct achieves substantially higher fidelity \(BLEU approximately 0.96 vs 0.34\), validating that modular architectures excel where generic VLMs fail on complex historical layouts. To support reproducibility and advance research in this domain, we release both the training corpus of 599 annotated pages and a curated PereStruct benchmark of 93 pages with expert\-verified ground\-truth block\-to\-article mappings. This framework establishes a robust foundation for high\-fidelity digitization and semantic reconstruction of complex archival materials.
>
> **📝 中文摘要：**
> 解析具有复杂非标准版式的历史文献仍是大规模档案数字化的根本瓶颈。与现代排版不同，历史报纸存在严重的物理损毁和极不规则的页面结构，这些难题即使最先进的视觉-语言模型也难以应对，呈现出严峻的分布外挑战。为此，我们开发了专为解析历史报纸设计的自动化流程，针对具有复杂多栏版式的文献进行处理。该方法将基于1426页全人工标注扫描件训练的微调YOLO架构（用于版面分析与版块检测），与创新的语义组装模块相结合，该模块通过联合建模TF-IDF词频语义相似度、微调YOLO的视觉嵌入信息以及几何版面约束来重构文章。这种多模态融合实现了最先进性能，在版块到文章的映射任务中F1分数达0.904。值得注意的是，通过与视觉-语言模型（Qwen3.6-35B-A3B和Qwen3.6-Plus）的端到端评估对比表明，PereStruct系统具有显著更高的保真度（BLEU值约0.96 vs 0.34），验证了模块化架构在处理复杂历史版式时优于通用视觉-语言模型的卓越性能。为支持该领域的可复现研究，我们公开了包含599页标注数据的训练语料，以及经专家验证的版块到文章映射基准测试集（93页精选页面）。本框架为复杂档案材料的高保真数字化与语义重构奠定了坚实基础。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.07661v1)

---

> ### 4. Real\-Time Industrial Defect Detection on Edge Hardware Using Fine\-Tuned YOLOv8: A Systematic Benchmark on the NEU Surface Defect Database and MVTec AD with Automotive & Battery Manufacturing Extensions
> **🔹 中文标题：** 基于微调YOLOv8的边缘硬件实时工业缺陷检测：在NEU表面缺陷数据库与MVTec AD上的系统性基准测试及汽车与电池制造扩展
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-06-03 |
> | 👤 作者 | Emmanuel Ezeji Somtochukwu |
>
> **📄 英文摘要：**
> Automated surface defect detection is critical for ensuring rigorous quality control in high\-speed manufacturing environments. While deep learning models offer remarkable accuracy, deploying them on resource\-constrained edge hardware without introducing significant latency remains a persistent challenge. This paper presents Industrial\-YOLO, an edge\-optimized framework built upon a fine\-tuned YOLOv8 architecture specifically engineered for real\-time industrial defect detection. We conduct a systematic benchmark utilizing the NEU surface defect database for steel sheets and the MVTec AD dataset, supplemented with custom automotive manufacturing extensions representing real\-world structural anomalies \(scratches, pits, and inclusions\). To bridge the gap between algorithmic complexity and edge hardware constraints, target\-specific optimizations are introduced via TensorRT and OpenVINO acceleration engines. Experimental results demonstrate that Industrial\-YOLO achieves a high\-velocity inference speed exceeding 120 FPS on the NVIDIA Jetson Orin platform while maintaining an exceptional mean Average Precision \(mAP\) of 98.5%. The proposed framework showcases highly robust, zero\-latency performance when deployed directly onto an active automotive assembly line, offering a scalable blueprint for next\-generation automated optical inspection \(AOI\) systems.
>
> **📝 中文摘要：**
> 表面缺陷自动检测对于确保高速制造环境下的严格质量控制至关重要。尽管深度学习模型具有卓越的准确性，但如何在资源受限的边缘硬件上部署这些模型且不引入显著延迟，仍然是一个持续存在的挑战。本文提出Industrial-YOLO框架——一个基于YOLOv8架构精细调优构建、专为实时工业缺陷检测设计的边缘优化框架。我们利用NEU钢板表面缺陷数据库和MVTec AD数据集进行系统性基准测试，并补充了代表真实世界结构异常（划痕、凹坑和夹杂物）的定制汽车制造扩展数据。为弥合算法复杂性与边缘硬件限制之间的差距，通过TensorRT和OpenVINO加速引擎引入了针对特定目标的优化方案。实验结果表明，Industrial-YOLO在NVIDIA Jetson Orin平台上实现了超过120 FPS的高速推理速度，同时保持了98.5%的卓越平均精度均值。该框架在直接部署到实际汽车装配线时展现出高度稳健的零延迟性能，为新一代自动光学检测系统提供了可扩展的蓝图。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.07659v1)

---

> ### 5. HYolo: An Intelligent IoT\-Based Object Detection System Using Hypergraph Learning
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
> 本文提出HYolo，一种基于物联网的智能目标检测框架，将超图学习整合至YOLO架构中。传统基于YOLO的目标检测模型主要捕捉成对特征交互，难以建模目标与上下文特征间的复杂高阶关系。为解决这一局限，HYolo引入超图学习以捕捉更丰富的上下文依赖关系，从而优化目标表征能力。在COCO数据集上的实验评估表明，该框架相比基线YOLO模型实现了显著性能提升：在mAP@50指标上获得约12%的提升，同时增强了整体检测精度与鲁棒性。通过建模高阶特征关系，HYolo在基于物联网的环境中提供了更优的上下文理解能力与更可靠的目标检测性能。研究结果表明，将超图学习整合至目标检测流程，为智能且具备上下文感知能力的物联网视觉系统开辟了极具潜力的发展方向。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.04345v1)

---

<div align="center">

*由 [YOLO-Arxiv-Daily](https://github.com/WangQvQ/YOLO-Arxiv-Daily) 自动生成*

</div>