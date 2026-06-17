<div align="center">

# YOLO ArXiv Daily

[![Daily Papers](https://img.shields.io/badge/📅-每日更新-blue)]()
[![arXiv](https://img.shields.io/badge/arXiv-最新论文-red)](https://arxiv.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)

*自动追踪 YOLO 相关最新论文，提供中英文双语摘要*

</div>

---

## 📑 论文列表

> ### 1. MOSAIC: Mobile Object Segmentation under Adverse Imaging Conditions for Rapid L\-PBF Keyhole Behavior Characterization
> **🔹 中文标题：** MOSAIC：面向恶劣成像条件的移动目标分割技术——用于快速表征激光粉末床熔融匙孔行为
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-06-15 |
> | 👤 作者 | Garrett Mathesen |
>
> **📄 英文摘要：**
> In laser powder bed fusion \(L\-PBF\) processes, the rapid evolution of gas and fluid interactions complicates our ability to properly monitor or control the process, with unstable keyholes leading to porosity and spatter formation. High\-speed operando x\-ray imaging of the keyhole has been used to better understand the impact of these interactions on the monitoring and control of the L\-PBF process. MOSAIC, a Mobile Object Segmentation algorithm for experiments under Adverse Imaging Conditions, is designed to perform rapid analysis of keyhole dynamics during active beamline experimentation without needing time consuming manual labeling or model training. Validation studies performed on 12 unique samples proved the robustness of MOSAIC with an average F1 score of 0.894 and a precision of 0.953 when compared to manually segmented images, performing equally or better than the SAM and YOLO machine learning methods tested. MOSAIC is efficient, processing frames cropped to a moving window approximately 150x250 pixels at 19.9 milliseconds per image on CPU, compared to 54 and 5284 milliseconds per image for inference on CPU for YOLO and SAM models.
>
> **📝 中文摘要：**
> 在激光粉末床熔融（L-PBF）工艺中，气体与流体相互作用的快速演变使得准确监控或控制该工艺变得复杂，不稳定的匙孔易导致气孔形成和飞溅产生。通过高速原位X射线成像技术观察匙孔状态，有助于深入理解这些相互作用对L-PBF工艺监控与控制的影响。MOSAIC（适用于恶劣成像条件的移动对象分割算法）旨在实现无需耗时人工标注或模型训练即可对光束线实验过程中的匙孔动态进行快速分析。基于12个不同样本的验证研究表明，MOSAIC算法具有强健的适应性，与人工分割图像相比，其平均F1分数达0.894，精确度为0.953，表现达到甚至超越SAM与YOLO机器学习方法。该算法处理效率突出：在中央处理器上，处理约150×250像素的移动窗口裁剪帧仅需19.9毫秒/图像，而YOLO与SAM模型在同等条件下的推理时间分别为54毫秒和5284毫秒/图像。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.16186v1)

---

> ### 2. TimeLens: On\-Device Artifact Recognition with Retrieval\-Augmented Question Answering for the Grand Egyptian Museum
> **🔹 中文标题：** 《TimeLens：基于检索增强问答的设备端文物识别系统——应用于大埃及博物馆场景》
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
> TimeLens是为大埃及博物馆开发的AI双语移动导览系统。参观者将手机对准展品时，系统可实时识别文物并支持以英语或阿拉伯语进行追问，由导览系统提供解答。本研究针对展厅部署的三大特殊问题：51件馆藏文物间的细粒度视觉相似性（许多拉美西斯雕像几乎相同）、策展训练数据与手持拍摄条件之间的差异，以及AI导览可能生成无史料支撑的历史陈述的风险。文中报告了两项工程贡献：首先通过数据质量驱动的迭代研究开发了移动端文物检测器——从基础模型自动标注（YOLO-World）到空间标签清洗规则，最终建立全手工标注数据集——验证标签质量为决定性因素：最终的YOLOv8n模型成功识别所有先前识别失败的类别，同时作为5.97MB的TensorFlow Lite资源文件，可在中端手机实时运行（mAP@0.5=0.995，mAP@0.5:0.95=0.924）。其次，基于108条记录的ChromaDB知识库构建双语检索增强生成（RAG）导览系统，经七种候选语言模型基准测试选定Gemma 4E2B（Q4 K M）；通过十项针对性优化将端到端延迟从超过30秒降至约10秒。两个子系统已集成至支持双语界面、博物馆位置识别和文本转语音功能的Flutter生产应用中。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.13267v1)

---

> ### 3. YOLO\-AMC: An Improved YOLO Architecture with Attention Mechanisms for Building Crack Detection
> **🔹 中文标题：** YOLO-AMC：基于注意力机制的改进型YOLO建筑裂缝检测架构
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
> 裂缝检测在基础设施巡检与结构健康监测（SHM）中具有重要作用。然而裂缝通常呈现为细长、低对比度的结构，易受背景噪声干扰，这对现有目标检测模型构成了挑战。本研究提出一种集成注意力机制的改进型YOLO架构，命名为YOLO-AMC（基于注意力机制的裂缝检测YOLO模型），以提升自动化裂缝检测性能。该模型基于YOLOv11架构，移除了原始C2PSA模块，并在颈部网络的多尺度特征融合层中引入全局注意力机制（GAM）、残差卷积块注意力模块（Res-CBAM）和通道重排注意力（SA）等多种注意力机制，以增强跨尺度特征融合能力。实验结果表明，YOLO-AMC在多项评估指标上均优于基线模型YOLOv11n与YOLOv8n。在对比的注意力模块中，GAM取得最佳检测性能，在测试集上实现mAP@0.5=0.9917与mAP@0.5:0.95=0.9506，优于YOLOv11（0.9833/0.9112）和YOLOv8（0.9707/0.8921）的性能。此外，该模型在保持7.6 GFLOPs计算复杂度的同时，在NVIDIA RTX 4090平台上达到110.95 FPS，在树莓派5边缘设备上实现约5 FPS的推理速度，体现了精度与部署效率的良好平衡。研究代码已发布于GitHub仓库：https://github.com/CY-Tsai24/YOLO-AMC。
>
> **💻 代码链接：** https://github.com/CY-Tsai24/YOLO-AMC.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.12958v1)

---

> ### 4. Performance Analysis of YOLOv11 and YOLOv8 for Mixed Traffic Object Detection under Adverse Weather Conditions in Developing Countries
> **🔹 中文标题：** 恶劣天气条件下发展中国家混合交通目标检测中YOLOv11与YOLOv8的性能分析
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
> 在现代车辆系统中，恶劣条件下的鲁棒性能已成为自动驾驶领域的关键课题。本研究对最新版YOLO系列——YOLOv11 Nano架构进行了全面评估，以广泛采用的YOLOv8 Nano作为基线模型，在融合印度驾驶数据集（IDD）[1]与伯克利深度驾驶数据集（BDD100K）[2]构建的定制混合数据集上进行基准测试。我们分析了在密集混合交通、降雨及低照度等高熵场景下，检测精度、推理速度与计算效率之间的权衡关系。具体而言，YOLOv11n在mAP@50指标上达到46.6%，相比基线模型其精确率显著提升3.2%，有效降低了杂乱场景中的误检率。此外，该模型展现出更优的能效比，在维持Tesla T4 GPU上70.9 FPS实时推理速度的同时，浮点运算量减少22%（6.3G vs 8.1G），为安全关键型边缘部署提供了理想权衡方案。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.12066v1)

---

> ### 5. Patient\-Level Diagnosis of Acute Myeloid Leukemia via Deep Learning Analysis of Bone Marrow Smear
> **🔹 中文标题：** 通过深度学习分析骨髓涂片实现急性髓系白血病的患者层面诊断
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
> 骨髓涂片复检在急性髓系白血病评估中仍具关键作用，但人工单细胞解读耗费人力，且患者级诊断需要整合大量细胞观察数据。本研究提出一种基于深度学习的"细胞-患者"级诊断流程，用于辅助分析骨髓涂片图像中的急性髓系白血病特征。

研究纳入来自六个匿名医疗中心的258例患者数据，其中中心1-3构成含169例患者的主队列，中心4-6构成含88例患者的外部验证队列。采用16类细胞标注术语体系描述整体细胞构成，涵盖粒细胞、单核细胞、红系、淋巴细胞、嗜酸性粒细胞等细胞类型。模型并非识别严格的原始细胞或白血病细胞，而是聚焦于专家定义的复合类别——复合原始样细胞，该类别依据跨中心形态学标准包含N、N1、M、M1、R、R1、J及J1八种亚型。

技术流程上，采用固定参数的YOLO分割模块进行细胞检测，通过轮廓交并比匹配专家多边形标注并生成标准化单细胞图像。EfficientNet-B0分类器采用两阶段训练策略，依次通过真实标注→YOLO模型、YOLO模型→YOLO模型的迁移学习，结合类别不平衡校正、中心-边界正则化及形态学辅助监督进行优化。单细胞预测结果通过聚合计算患者级复合原始样细胞比例，为急性髓系白血病诊断提供支持。

该流程在内部验证中表现稳定，并保持良好的外部泛化能力，在中心4、5、6的集成加权F1分数分别达到0.9076、0.8696和0.9124。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.10735v1)

---

<div align="center">

*由 [YOLO-Arxiv-Daily](https://github.com/WangQvQ/YOLO-Arxiv-Daily) 自动生成*

</div>