<div align="center">

# YOLO ArXiv Daily

[![Daily Papers](https://img.shields.io/badge/📅-每日更新-blue)]()
[![arXiv](https://img.shields.io/badge/arXiv-最新论文-red)](https://arxiv.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)

*自动追踪 YOLO 相关最新论文，提供中英文双语摘要*

</div>

---

## 📑 论文列表

> ### 1. Identifying the Unknown: Prompt\-Free Open Vocabulary Anomaly Recognition for Robot\-Object Interaction
> **🔹 中文标题：** 未知识别：面向机器人-物体交互的无需提示开放词汇异常识别
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-06-25 |
> | 👤 作者 | Philipp Allgeuer |
>
> **📄 英文摘要：**
> Robots operating in real\-world environments must in general be able to recognize previously unseen objects. As robotic systems move toward open\-world autonomy, there is a growing, yet largely unmet, need for open vocabulary object detectors that are prompt\-free and efficient enough for continuous deployment. We present AnomNOVIC, a two\-stage known\-workspace framework that combines a masked autoencoder \(MAE\) trained for anomaly detection, with NOVIC, a powerful real\-time prompt\-free open vocabulary image classifier. The MAE produces generic object\-agnostic bounding boxes, allowing NOVIC to classify salient image regions without requiring a predefined candidate class list. We evaluate AnomNOVIC against strong open vocabulary baselines in a tabletop robot\-object environment featuring the NICOL humanoid robot, reaching 47.1% AP / 57.5% AP50 for prompt\-free recognition, and 59.0% AP / 72.5% AP50 if class candidates are provided. Across additional datasets, including an in\-the\-wild test set with 48 unique objects, AnomNOVIC reaches up to 82.6% prompt\-free detection and classification accuracy. These results significantly surpass all tested open vocabulary baselines, including YOLO\-World\-v2, OWLv2, and YOLOE.
>
> **📝 中文摘要：**
> 在真实环境中运行的机器人通常需要能够识别此前未见过的物体。随着机器人系统向开放世界自主性发展，对无需提示、效率足以支持持续部署的开放词汇物体检测器的需求日益增长，但目前尚无完善解决方案。我们提出AnomNOVIC——这是一个两阶段的已知场景框架，结合了用于异常检测的掩码自编码器（MAE）与强大的实时无提示开放词汇图像分类器NOVIC。MAE生成通用物体无关边界框，使NOVIC能够对显著图像区域进行分类，而无需预定义候选类别列表。我们在配备NICOL仿人机器人的桌面机器人物体环境中对AnomNOVIC进行评估，该系统在无需提示的识别任务中达到47.1% AP / 57.5% AP50，在提供类别候选的情况下达到59.0% AP / 72.5% AP50。在包含48种独特物体的野外测试集等多个额外数据集上，AnomNOVIC的无提示检测与分类准确率最高可达82.6%。这些结果显著超越了包括YOLO-World-v2、OWLv2和YOLOE在内的所有测试基准的开放词汇检测器。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.26829v1)

---

> ### 2. Application of Machine Learning for the Identification of 2D Colloidal Assemblies: A Case Study on Particles of Distinct Shapes
> **🔹 中文标题：** 机器学习在二维胶体组装体识别中的应用：基于不同形状颗粒的案例研究
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-06-22 |
> | 👤 作者 | L. T. Khusainova |
>
> **📄 英文摘要：**
> This work addresses the problem of identifying colloidal monolayer assemblies using particles of various shapes \(two\-dimensional coatings\): spheres, ellipsoids, cuboids, and rods. The following classification of assemblies is considered: isolated particles, dimers, chains, clusters, and loops. The YOLO model was chosen as the identification method. Synthetic datasets were prepared for each of the four particle shapes to train the models. The paper discusses the application of models trained on synthetic data to experimental images. An analysis was carried out on the feasibility of using such models for recognizing configurations in real images. While recognition on artificial images is nearly perfect, tests on experimental images showed a significant deviation. The average error across all particle types was 43.1%, but a considerable spread in values is observed: from 20% for spheres to 58.5% for cuboids, indicating the algorithm's selective sensitivity to object geometry. The created datasets and trained models are freely available for use. The corresponding modules have been integrated into the previously developed information system \(https://isanm.space/\). To further improve prediction results, it is necessary to prepare datasets based on experimental images.
>
> **📝 中文摘要：**
> 本研究旨在解决使用不同形状粒子（二维涂层）——球体、椭球体、长方体和棒状体——进行胶体单层组装体识别的问题。我们考虑以下组装体分类：孤立颗粒、二聚体、链状结构、簇状结构和环状结构。选择YOLO模型作为识别方法。为四种粒子形状分别准备了合成数据集以训练模型。本文探讨了将合成数据训练的模型应用于实验图像的效果。我们分析了此类模型用于实际图像构型识别的可行性。虽然在人工图像上的识别近乎完美，但在实验图像上的测试显示出显著偏差。所有粒子类型的平均误差为43.1%，但数值分布范围较广：从球体的20%到长方体的58.5%，表明该算法对物体几何形状具有选择性敏感度。所创建的数据集和训练模型可免费使用。相应模块已集成到先前开发的信息系统中（https://isanm.space/）。为进一步提高预测结果，有必要基于实验图像准备数据集。
>
> **💻 代码链接：** https://isanm.space/).
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.23639v1)

---

> ### 3. Fursee: Hybrid YOLO\-DINOv3 Framework for Fursuit Identity Retrieval and Clustering
> **🔹 中文标题：** Fursee：基于混合YOLO-DINOv3框架的毛绒玩偶身份检索与聚类方法
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-06-22 |
> | 👤 作者 | Jundi Wu |
>
> **📄 英文摘要：**
> Global furry conventions produce massive fursuit photographs, while manual sorting brings heavy labor costs and calls for automatic identity retrieval and clustering solutions. General multimodal models lack dedicated optimization for complex fursuit scenes, and no public benchmark dataset exists for this task. To fill this gap, we build a specialized fursuit image dataset and present a three\-stage hybrid pipeline Fursee for fursuit identity retrieval and clustering. First, YOLO detects and crops high\-resolution fursuit head patches to improve localization of small and overlapping targets. Second, ArcFace optimizes DINOv3 embeddings to enlarge angular separation between different identities on the feature hypersphere. Third, DBSCAN performs unsupervised clustering, with silhouette\-coefficient\-driven search automatically selecting optimal hyperparameters rather than fixed manual radius. Retrieval and clustering experiments verify that our pipeline outperforms mainstream multimodal models including GPT5.5, Claude Opus 4.8 and Qwen3.7\-Plus on all evaluation metrics, achieving competitive performance for fursuit head retrieval and grouping.
>
> **📝 中文摘要：**
> 全球兽装展会产出海量兽装照片，人工筛选带来沉重人力成本，亟需自动化身份检索与聚类解决方案。通用多模态模型缺乏针对复杂兽装场景的专项优化，且该领域尚无公开基准数据集。为填补此空白，本研究构建专属兽装图像数据集，并提出三阶段混合流程Fursee实现兽装身份检索与聚类：首先通过YOLO检测裁剪高分辨率兽装头部区块，提升对小尺寸及重叠目标的定位精度；其次利用ArcFace优化DINOv3特征嵌入，在特征超球面上扩大不同身份间的角度分离度；最终通过DBSCAN执行无监督聚类，基于轮廓系数驱动搜索自动优选超参数（替代固定人工设定半径）。检索与聚类实验表明，本流程在所有评估指标上均优于GPT5.5、Claude Opus 4.8及Qwen3.7-Plus等主流多模态模型，在兽装头部检索与分组任务中展现出竞争力性能。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.22872v1)

---

> ### 4. NegAS: Negative Label Guided Attention and Scoring for Out\-of\-Distribution Object Detection with Vision\-Language Models
> **🔹 中文标题：** NegAS：面向视觉语言模型分布外目标检测的负标签引导注意力与评分机制
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-06-21 |
> | 👤 作者 | Yingjie Zhang |
>
> **📄 英文摘要：**
> Out\-of\-Distribution \(OOD\) detection is essential for ensuring the robustness and reliability of object detection systems deployed in safety\-critical applications. While prior research has mainly focused on uni\-modal detectors or vision\-language model \(VLM\) based classifiers, the potential of VLM\-based object detectors in OOD scenarios remains underexplored. In this work, we take the first step toward building OOD object detection methods upon VLMs. We identify two challenges specific to VLM detectors: \(i\) their text\-guided attention enhances foreground with ID labels but treats background uniformly, leaving potential OOD regions unexploited for separating in\-distribution \(ID\) from OOD instances; and \(ii\) their sigmoid\-based multi\-label outputs are incompatible with softmax\-based OOD scores, calling for scoring functions consistent with VLM probabilistic outputs. Hence, we introduce Negative Label Guided Attention and Scoring \(NegAS\). To address \(i\), we propose a negative label guided attention module \(NegA\), where LLM\-generated, visually\-similar but semantically\-different negative labels are used to guide attention toward potential OOD background regions. To address \(ii\), we introduce a novel sigmoid\-based OOD scoring function \(NegS\) that leverages both ID and negative labels, producing strong responses for ID instances and suppressed responses for OOD ones. Extensive experiments demonstrate that our approach improves OOD detection performance by a large margin while maintaining ID accuracy, e.g., reducing the FPR95 by 11.4% on the COCO dataset and 25.5% on the OpenImages dataset compared to the baseline model. While initially designed for dense VLM detectors like YOLO\-World, we successfully adapt NegAS to Grounding DINO, a query\-based VLM transformer and achieve significant improvements, demonstrating the generalizability of our framework.
>
> **📝 中文摘要：**
> 分布外（OOD）检测对于确保部署在安全关键应用中的目标检测系统的鲁棒性与可靠性至关重要。既有研究主要集中于单模态检测器或基于视觉语言模型的分类器，但基于视觉语言模型的目标检测器在OOD场景中的潜力尚未得到充分探索。本文首次尝试构建基于视觉语言模型的OOD目标检测方法。我们指出VLM检测器面临的两项特有挑战：（i）其文本引导注意力机制虽能通过ID标签增强前景区域，但对背景处理过于均质化，导致可能分离分布内与分布外样本的OOD潜在区域未被有效利用；（ii）其基于Sigmoid的多标签输出与基于Softmax的OOD评分机制存在不兼容性，需要设计与VLM概率输出一致的评分函数。为此，我们提出负标签引导注意力与评分机制。针对挑战（i），我们设计了负标签引导注意力模块，利用大语言模型生成视觉相似但语义相异的负标签，引导模型关注潜在OOD背景区域；针对挑战（ii），我们创新性地提出基于Sigmoid的OOD评分函数，同时利用ID标签与负标签信息，使分布内实例产生强响应而分布外实例响应受到抑制。大量实验表明，本方法在保持分布内检测精度的同时，显著提升了OOD检测性能：相较基线模型，在COCO数据集上降低FPR95指标达11.4%，在OpenImages数据集上达25.5%。虽然最初为YOLO-World等稠密型VLM检测器设计，但我们成功将负标签引导注意力与评分机制移植至基于查询机制的VLMTransformer模型Grounding DINO，取得显著性能提升，证明了本框架的通用性。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.22537v2)

---

> ### 5. A Smart Classroom Behavior Analysis Framework with a New Highly Congested Classroom Dataset
> **🔹 中文标题：** 一种新型高度拥挤教室数据集支持的智能教室行为分析框架
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-06-19 |
> | 👤 作者 | Wei Xu |
>
> **📄 英文摘要：**
> Student behavior detection is important for intelligent classroom analysis but remains challenging in large\-class scenarios due to dense instance co\-occurrence, asymmetric occlusion, depth\-wise scale variation, and fine\-grained semantic degradation in distant targets. Existing classroom behavior datasets and general\-purpose detectors are insufficient to characterize and address these challenges. This paper constructs the Highly Congested Classroom Behavior \(HCCB\) dataset, containing 50,229 student behavior instances across seven categories: reading, writing, heads up, sleeping, looking around, bowing head, and using phone. HCCB provides a challenging benchmark that integrates dense distributions, severe occlusion, scale variation, and fine\-grained behavioral semantics. To address these issues, we propose ODER\-HSFNet, a YOLO\-based detection framework tailored to highly crowded classrooms. At its core, ODER\-HSFNet introduces three task\-specific innovations: the Occlusion\-aware Deformable Edge Rectifier \(ODER\), which strengthens boundary evidence under occlusion; the Hypergraph\-State Spatial Fusion \(HSSF\) module, which integrates local structure enhancement, state\-space contextual modeling, and high\-order relation aggregation; and the Occlusion\-Calibrated Detection Head \(OCDetect\), which suppresses low\-quality Pre\-NMS candidates and reduces false positives from occlusion boundaries and neighboring instances. Experiments on two classroom behavior detection datasets show that ODER\-HSFNet outperforms mainstream YOLO\-series methods, achieving 60.60%/80.12% mAP50:95/mAP50 on HCCB and 57.36%/74.65% on SCB\-D3\-S. Ablation studies further verify the effectiveness of the proposed design for highly crowded classroom behavior detection.
>
> **📝 中文摘要：**
> 学生行为检测对于智能课堂分析至关重要，但在大规模课堂场景中，由于密集实例共存、非对称遮挡、深度方向尺度变化以及远距离目标的细粒度语义退化等问题，该任务仍面临严峻挑战。现有课堂行为数据集与通用检测器难以有效刻画和应对这些挑战。本文构建了高度拥挤课堂行为数据集，包含阅读、书写、抬头、睡觉、环视、低头、使用手机等七类共计50,229个学生行为实例。该数据集整合了密集分布、严重遮挡、尺度变化与细粒度行为语义，为相关研究提供了具有挑战性的基准。

针对上述问题，本文提出专为高度拥挤课堂设计的ODER-HSFNet检测框架。该模型基于YOLO架构，包含三项任务特定创新：遮挡感知可变形边缘修正模块通过强化遮挡边界证据提升特征提取；超图状态空间融合模块整合局部结构增强、状态空间上下文建模与高阶关系聚合；遮挡校准检测头则能抑制低质量非极大值抑制前候选框，减少遮挡边界及相邻实例导致的误检。在两个课堂行为检测数据集上的实验表明，ODER-HSFNet显著优于主流YOLO系列方法，在HCCB数据集上达到60.60%/80.12%的mAP50:95/mAP50指标，在SCB-D3-S数据集上达到57.36%/74.65%。消融研究进一步验证了该设计在高度拥挤课堂行为检测中的有效性。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.21568v1)

---

<div align="center">

*由 [YOLO-Arxiv-Daily](https://github.com/WangQvQ/YOLO-Arxiv-Daily) 自动生成*

</div>