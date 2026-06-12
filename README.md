<div align="center">

# YOLO ArXiv Daily

[![Daily Papers](https://img.shields.io/badge/📅-每日更新-blue)]()
[![arXiv](https://img.shields.io/badge/arXiv-最新论文-red)](https://arxiv.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)

*自动追踪 YOLO 相关最新论文，提供中英文双语摘要*

</div>

---

## 📑 论文列表

> ### 1. Performance Analysis of YOLOv11 and YOLOv8 for Mixed Traffic Object Detection under Adverse Weather Conditions in Developing Countries
> **🔹 中文标题：** 在发展中国家恶劣天气条件下混合交通目标检测的YOLOv11与YOLOv8性能分析
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
> 在现代车辆系统中，严苛条件下的鲁棒性能已成为自动驾驶领域的关键难题。本研究对YOLO系列最新架构YOLOv11 Nano展开全面评估，以当前广泛应用的YOLOv8 Nano作为基准，在融合印度驾驶数据集(IDD) [1]与伯克利深度驾驶数据集(BDD100K) [2]的自定义混合数据集上进行性能对比分析。我们针对包含密集混合交通流、降雨及低光照环境的高熵场景，系统评估了目标检测精度、推理速度与计算效率之间的权衡关系。实验结果表明，YOLOv11n在平均精度均值(mAP@50)达到46.6%的同时，较基准模型提升3.2%的精确率，显著降低了复杂场景中的误检率。此外，该模型在保持70.9 FPS（基于Tesla T4 GPU）实时推理速度的前提下，计算量减少22%（6.3G vs. 8.1G FLOPs），为安全关键型边缘部署提供了效率与性能的优化平衡方案。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.12066v1)

---

> ### 2. Patient\-Level Diagnosis of Acute Myeloid Leukemia via Deep Learning Analysis of Bone Marrow Smear
> **🔹 中文标题：** 基于深度学习分析骨髓涂片的急性髓系白血病患者级诊断
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
> 骨髓涂片复审对急性髓系白血病的评估至关重要，但人工逐细胞解读过程耗时费力，且患者层级的诊断需要整合大量细胞观察数据。本文提出一种基于深度学习的细胞-患者级流水线，用于从骨髓涂片图像辅助诊断急性髓系白血病。研究纳入来自六个匿名中心的258例患者，其中主要队列包含来自中心1-3的169例患者，外部验证队列包含来自中心4-6的89例患者。采用16类细胞注释体系描述全局细胞构成，涵盖粒细胞、单核细胞、红系细胞、淋巴细胞、嗜酸性粒细胞等类型。该模型不局限于识别严格定义的急性髓系白血病原始细胞或白血病原始细胞，而是针对专家定义的复合类别——复合型原始样细胞（包含根据全项目形态学标准划分的N、N1、M、M1、R、R1、J、J1类型）。通过固定YOLO分割模块检测细胞，利用轮廓IoU将预测轮廓匹配专家多边形标注，并生成标准化单细胞图像。采用两阶段训练策略（GT到YOLO、YOLO到YOLO）训练EfficientNet-B0分类器，结合类别不平衡校正、中心边界正则化及形态学辅助监督。将细胞层级预测结果聚合为患者层级的复合型原始样细胞比例，为急性髓系白血病导向的诊断提供支持。该流水线在内部验证中表现稳定，并在外部数据中保持泛化能力，在中心4、5、6的集成加权F1分数分别为0.9076、0.8696和0.9124。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.10735v1)

---

> ### 3. Time\-frequency localization of bird calls in dense soundscapes
> **🔹 中文标题：** 密集声景中鸟鸣的时频定位研究
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
> 被动声学监测技术可实现野生动物的大规模观测，但当前大多数生物声学分类器仅能预测时间窗口内的物种存在，无法精确定位发声的时间或频率位置，这限制了后续分析。本研究将鸟鸣检测转化为声谱图上的目标检测任务，训练YOLO11模型在新加坡密集热带声景中定位鸟鸣。我们开发了开源浏览器标注工具，并提出最小交集比（IoMin）评估指标——相较于标准IoU指标，该指标能更好地处理模糊声学边界，更适用于此类复杂声景分析。最优YOLO模型在新加坡本地声景中的性能较基线提升近一倍（IoMin@50 F1-score达81.8%，对比基线的42.1%），同时在未见过的夏威夷声景数据集上仍优于基线（58.6% vs. 48.6%）。研究结果表明，目标检测框架为复杂声景中动物发声的时间-频率定位提供了新思路。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.10407v1)

---

> ### 4. Feasibility to detect rapid change and disappearance of seagrass: Lessons from nearly 80 years of vegetation change in the Ako, Seto Inland Sea, Japan
> **🔹 中文标题：** 检测海草快速变化与消失的可行性：基于日本濑户内海阿久区域近80年植被变迁的经验启示
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
> 本研究分析了日本濑户内海赤穂潮间带2025年大叶藻在一年内几近消失的现象。通过整合20世纪40年代以来的航空照片、高分辨率卫星影像、GRUS图像（2.5-5米）及月度Sentinel-2合成影像（10米），我们重建了约80年的海草分布演变过程。基于深度学习的YOLO分割模型在各类数据源中均达到高精度（总体精度≥0.9），虽然未能实现物种级区分，但有效捕捉了植被面积的主要时间动态变化。长期平均海草覆盖面积为6.8公顷，但年际波动显著，从1974年的3.5公顷到1989年的41.3公顷，而2025年骤降至仅0.2公顷。2019至2026年的Sentinel-2合成影像揭示出明显的季节性规律：植被面积在初夏增长，秋季开始衰减。然而2025年夏季后面积急剧下降，并在整个2025至2026年冬季持续异常偏低。研究表明2025年事件并非正常波动，而是涉及优势冠层形成种丧失的快速生态系统转变，最可能由区域性夏季水温异常升高驱动。该发现对与自然相关披露框架（TNFD）接轨的海草核心海洋变量（EOVs）及自然状态（SoN）指标具有重要启示。与森林不同，海草草甸因显著的季节性和突发性崩溃强烈影响面积指标，需要更高时间分辨率监测。因此，除既往提出的物种分类精度问题外，我们建议：（1）基于最长可获取记录并符合生态学逻辑定义基准线，（2）在年际比较前进行季节标准化处理，（3）标记极端面积异常年份而非将其作为参考基准。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.07949v1)

---

> ### 5. Attention\-Guided Autoencoder Fusion for Insulator Defect Detection Using UAV Transmission\-Line Imaging
> **🔹 中文标题：** 基于无人机输电线路成像的注意力引导自动编码器融合绝缘子缺陷检测
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
> 由于无人机图像中存在严重的类别不平衡、大尺度变化以及缺陷实例空间范围小等问题，高压输电线路绝缘子的自动缺陷检测仍然面临挑战。为应对这些挑战，本文提出AE-YOLO——一种注意力引导的自编码器增强型YOLO框架，以实现鲁棒的绝缘子缺陷检测。该架构在特征金字塔网络-路径聚合网络颈部集成了轻量级瓶颈自编码器，可在多尺度特征融合过程中保留异常敏感信息。主干网络中广泛使用卷积注意力模块以增强特征判别能力并抑制背景干扰。框架还引入了方差最大化自编码器正则化策略，促使网络学习多样化且具有缺陷判别性的潜在表征。网络采用统一目标函数进行训练，该函数结合了焦点损失、完整交并比损失和自编码器正则化，以解决前景背景不平衡问题并提升定位精度。在推理阶段，采用加权框融合方法整合来自YOLOv8、YOLOv10和YOLO11的预测结果，并通过自编码器引导的置信度增强机制提升对罕见缺陷类别的检测灵敏度。在绝缘子缺陷检测数据集上的实验表明，采用EfficientNetV2主干的AE-YOLO在0.5交并比阈值下达到95.10%的平均精度均值，精确率为96.40%，召回率为93.80%。该性能在0.5交并比阈值下的平均精度均值和召回率分别超越最强的YOLO系列基线模型5.0个百分点和6.7个百分点。实验结果证实了该框架的有效性与适应性，为基于无人机的输电线路巡检与缺陷监测提供了实用且可扩展的解决方案。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.06536v1)

---

<div align="center">

*由 [YOLO-Arxiv-Daily](https://github.com/WangQvQ/YOLO-Arxiv-Daily) 自动生成*

</div>