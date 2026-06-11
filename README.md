<div align="center">

# YOLO ArXiv Daily

[![Daily Papers](https://img.shields.io/badge/📅-每日更新-blue)]()
[![arXiv](https://img.shields.io/badge/arXiv-最新论文-red)](https://arxiv.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)

*自动追踪 YOLO 相关最新论文，提供中英文双语摘要*

</div>

---

## 📑 论文列表

> ### 1. Patient\-Level Diagnosis of Acute Myeloid Leukemia via Deep Learning Analysis of Bone Marrow Smear
> **🔹 中文标题：** 基于深度学习分析的急性髓系白血病患者级骨髓涂片诊断
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
> 骨髓涂片复查对于急性髓系白血病评估仍至关重要，但人工单细胞判读耗时费力，且患者层面诊断需要整合大量细胞观测数据。我们提出了一种基于骨髓涂片图像的AML辅助诊断的"细胞-患者"深度学习流程。研究纳入来自六个匿名中心的258例患者，其中中心1-3的169例构成主要队列，中心4-6的89例构成外部验证队列。采用16类细胞注释体系描述全局细胞构成，涵盖粒细胞、单核细胞、红细胞系、淋巴细胞、嗜酸性粒细胞等细胞类型。模型并非针对严格的AML原始细胞或白血病原始细胞，而是聚焦于专家定义的复合类别——复合型原始样细胞，依据项目统一形态学标准包含N、N1、M、M1、R、R1、J及J1等类型。研究采用固定YOLO分割模块检测细胞，通过轮廓交并比匹配预测轮廓与专家多边形标注，生成标准化单细胞裁剪图。EfficientNet-B0分类器采用"真值引导至YOLO"和"YOLO自优化"两阶段训练策略，结合类别不平衡校正、中心-边缘正则化和形态学辅助监督。细胞层面预测被整合为患者层面的CBLC比例，为AML诊断提供定向支持。该流程在内部验证中表现稳定，外部泛化能力良好，在中心4、5、6的集成加权F1分数分别为0.9076、0.8696和0.9124。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.10735v1)

---

> ### 2. Time\-frequency localization of bird calls in dense soundscapes
> **🔹 中文标题：** 密集声景中的鸟类叫声时频定位
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
> 被动声学监测可实现大规模野生动物观测，但大多数生物声学分类器仅能预测物种在时间窗口内的存在性，无法精确锁定发声的时间或频率位置，这限制了后续分析的深度。我们将鸟鸣检测构建为声谱图上的目标检测任务，训练YOLO11模型以定位新加坡密集热带声景中的鸟类鸣叫。研究进一步引入开源浏览器标注工具，并提出"最小交并比"评估指标——该指标相较于标准交并比能更有效地处理声学边界模糊问题，更契合本研究场景。最优YOLO模型在新加坡分布内声景中的性能近乎达到基线模型的两倍（81.8% vs. 42.1% 最小交并比@50 F1分数），同时在来自夏威夷的未见过分布外录音中仍优于基线（58.6% vs. 48.6%）。这些结果表明，目标检测框架是解决复杂声景中动物发声时频定位的极具前景的技术路径。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.10407v1)

---

> ### 3. Feasibility to detect rapid change and disappearance of seagrass: Lessons from nearly 80 years of vegetation change in the Ako, Seto Inland Sea, Japan
> **🔹 中文标题：** 探测海草快速变化和消失的可行性：来自日本濑户内海阿子近80年植被变化的启示
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
> 本研究分析了日本濑户内海的Ako潮滩，该区域几乎所有大叶藻在2025年内近乎完全消失。我们利用20世纪40年代以来的航拍照片、高分辨率卫星图像、GRUS影像（分辨率2.5-5米）及月度Sentinel-2合成影像（10米），重构了约80年的海草分布变化。基于深度学习的YOLO分割算法在这些数据集中实现了高精度（总体精度≥0.9）；尽管未能区分物种，但模型准确捕捉了植被面积的主要时间动态特征。长期平均海草面积为6.8公顷，但数值波动显著，从1974年的3.5公顷到1989年的41.3公顷（2025年仅0.2公顷除外）。2019至2026年的Sentinel-2合成影像显示明显的季节性规律：初夏植被面积增长，秋季开始衰退。然而2025年夏季后面积急剧缩减，并在整个2025-2026冬季持续异常低迷。研究结果表明，2025年事件并非正常波动，而是涉及优势冠层物种丧失的快速生态系统转变，最可能由区域性夏季水温升高驱动。该发现对海草核心海洋变量及符合TNFD框架的自然相关披露中采用的自然状态指标具有重要启示。与森林不同，海草草甸需要更精细的时间分辨率，因为显著的季节性和突发性崩溃都会强烈影响面积指标。因此，除已指出的物种级分类精度问题外，我们建议：（1）基于最长可用记录并经生态学论证确定基线；（2）进行年际比较前先进行季节性标准化；（3）标记极端面积异常年份而非将其作为参考基准。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.07949v1)

---

> ### 4. Attention\-Guided Autoencoder Fusion for Insulator Defect Detection Using UAV Transmission\-Line Imaging
> **🔹 中文标题：** 基于无人机输电线路图像的绝缘子缺陷检测：注意力引导自编码器融合方法
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
> 高压输电线路绝缘子的自动化缺陷检测在无人机影像中面临严重类别不平衡、尺度变化大以及缺陷实例空间占比小等挑战。为解决这些问题，本文提出AE-YOLO——一种注意力引导的自编码器增强YOLO框架，用于稳健的绝缘子缺陷检测。该架构在特征金字塔网络-路径聚合网络颈部中集成了轻量级瓶颈自编码器，可在多尺度特征融合过程中保留对异常敏感的信息。骨干网络全面采用卷积块注意力模块，增强特征判别能力并抑制背景干扰。框架还引入方差最大化自编码器正则化策略，促使网络学习多样化且具有缺陷判别性的潜在表征。网络训练采用统一目标函数，结合焦点损失、完整交并比损失和自编码器正则化，以解决前景-背景不平衡问题并提升定位精度。推理阶段通过加权框融合技术整合YOLOv8、YOLOv10及YOLO11的预测结果，并利用自编码器引导的置信度增强机制提升对罕见缺陷类别的敏感性。在绝缘子缺陷检测数据集上的实验表明，采用EfficientNetV2骨干的AE-YOLO在0.5阈值下达到95.10%平均精度、96.40%精确率和93.80%召回率。该性能在平均精度和召回率上分别超越最强YOLO系列基线5.0和6.7个百分点，验证了框架的有效性与适应性，为基于无人机的输电线路巡检与缺陷监测提供了实用可扩展的解决方案。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.06536v1)

---

> ### 5. PereStruct: Multimodal Semantic Assembly for Robust Historical Document Parsing
> **🔹 中文标题：** 佩雷斯特：面向鲁棒历史文档解析的多模态语义组装方法
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
> 解析具有复杂非标准布局的历史文档，仍是大规模档案数字化进程中的根本瓶颈。与现代排版不同，历史报刊存在严重的物理老化和高度不规则的页面结构，即使最先进的视觉语言模型也难以应对，呈现出严峻的分布外挑战。针对这一差距，我们设计了专门解析历史报刊（尤其以复杂多栏版面为特征）的自动化处理流程。该方法将基于1426页全人工标注扫描件训练的微调YOLO架构（用于版面分析与区块检测）与新型语义组装模块相结合，通过联合建模TF-IDF词汇语义相似性、微调YOLO的视觉嵌入以及几何版面约束来重构文章内容。这种多模态集成方案达到了最优性能，在区块-文章映射任务中取得0.904的F1值。值得注意的是，通过与视觉语言模型（Qwen3.6-35B-A3B和Qwen3.6-Plus）的端到端对比评估，PereStruct展现出显著更高的保真度（BLEU值约0.96 vs 0.34），验证了模块化架构在处理复杂历史版面时较通用视觉语言模型的优越性。为支持可重复性研究并推动该领域进展，我们发布了包含599页标注文档的训练语料库，以及由专家核验区块-文章映射的93页PereStruct基准数据集。该框架为高保真数字化处理与复杂档案材料的语义重构奠定了坚实基础。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.07661v1)

---

<div align="center">

*由 [YOLO-Arxiv-Daily](https://github.com/WangQvQ/YOLO-Arxiv-Daily) 自动生成*

</div>