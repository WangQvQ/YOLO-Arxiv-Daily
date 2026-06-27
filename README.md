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
> **🔹 中文标题：** 识别未知：面向机器人-物体交互的无需提示开放词汇异常识别
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
> 运行在真实环境中的机器人通常需要具备识别先前未见过物体的能力。随着机器人系统朝着开放世界自主化方向发展，对无需提示且能效足以支持持续部署的开放词汇物体检测器的需求日益增长，但目前这一需求尚未得到充分满足。本文提出AnomNOVIC框架——一个基于已知工作区的两阶段系统，该系统将专门用于异常检测的掩码自编码器与强大的实时无需提示开放词汇图像分类器NOVIC相结合。掩码自编码器生成与物体无关的通用边界框，使NOVIC无需预定义候选类别列表即可对显著图像区域进行分类。我们在配备NICOL人形机器人的桌面机器人-物体环境中，将AnomNOVIC与强大的开放词汇基线进行评估：无需提示识别达到47.1% AP / 57.5% AP50，提供类别候选时达到59.0% AP / 72.5% AP50。在额外数据集（包含48个独特物体的真实场景测试集）上，AnomNOVIC的无提示检测与分类准确率最高达82.6%。这些结果显著超越了所有测试的开放词汇基线方法，包括YOLO-World-v2、OWLv2和YOLOE。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.26829v1)

---

> ### 2. Application of Machine Learning for the Identification of 2D Colloidal Assemblies: A Case Study on Particles of Distinct Shapes
> **🔹 中文标题：** 机器学习在二维胶体组装体识别中的应用：基于不同形状粒子的案例研究
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
> 本研究针对利用不同形状（二维包覆层）颗粒——球体、椭球体、长方体及棒状体——识别胶体单层组装结构的问题展开探讨。研究涵盖以下组装类型分类：孤立颗粒、二聚体、链状结构、团簇及环状结构。采用YOLO模型作为识别方法，为四种颗粒形状分别构建合成数据集以训练模型。文中讨论了将合成数据训练的模型应用于实验图像的效果，并分析了此类模型识别真实图像构型的可行性。在人工图像上识别准确率接近完美，但实验图像测试显示显著偏差：所有颗粒类型平均误差达43.1%，且数值分布差异明显——球体误差为20%，长方体则高达58.5%，表明算法对几何形状具有选择性敏感特征。所建数据集及训练模型已开放获取，相应模块已集成至前期开发的信息系统（https://isanm.space/）。为提升预测效果，未来需基于实验图像构建专用数据集。
>
> **💻 代码链接：** https://isanm.space/).
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.23639v1)

---

> ### 3. Fursee: Hybrid YOLO\-DINOv3 Framework for Fursuit Identity Retrieval and Clustering
> **🔹 中文标题：** Fursee: 混合YOLO-DINOv3框架用于毛绒装束身份检索与聚类
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
> 全球兽装展会产出海量兽装照片，而人工整理带来高昂人力成本，亟需开发自动身份检索与聚类方案。现有通用多模态模型缺乏针对复杂兽装场景的专项优化，且该领域尚无公开基准数据集。为填补这一空白，我们构建了专门的兽装图像数据集，并提出三阶段混合流水线Fursee用于兽装身份检索与聚类。首先，采用YOLO检测并裁剪高分辨率兽装头部区域，以提升对小尺度及重叠目标的定位精度；其次，运用ArcFace优化DINOv3嵌入特征，在特征超球面上扩大不同身份间的角度间距；最后，通过DBSCAN进行无监督聚类，借助轮廓系数驱动搜索自动选择最优超参数，替代固定的人工半径设定。检索与聚类实验验证，本流水线在所有评估指标上均超越GPT5.5、Claude Opus 4.8、Qwen3.7-Plus等主流多模态模型，在兽装头部检索与分组任务中取得了具有竞争力的性能。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.22872v1)

---

> ### 4. NegAS: Negative Label Guided Attention and Scoring for Out\-of\-Distribution Object Detection with Vision\-Language Models
> **🔹 中文标题：** NegAS：基于负标签引导的注意力与评分机制的视觉语言模型分布外物体检测
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
> 分布外检测对于保障安全关键应用中目标检测系统的鲁棒性与可靠性至关重要。现有研究主要聚焦于单模态检测器或基于视觉语言模型的分类器，而视觉语言模型目标检测器在分布外场景中的潜力尚未得到充分探索。本文首次尝试在视觉语言模型基础上构建分布外目标检测方法。我们识别出视觉语言模型检测器面临的两大挑战：其文本引导注意力机制虽能通过ID标签增强前景区域，但对背景区域采取统一处理，未能利用潜在分布外区域来区分分布内实例与分布外实例；其基于sigmoid的多标签输出与基于softmax的分布外评分存在不兼容性，亟需开发与视觉语言模型概率输出一致的评分函数。为此，我们提出负标签引导注意力与评分方法。针对第一项挑战，我们设计负标签引导注意力模块，利用大语言模型生成的视觉相似但语义相异的负标签，引导注意力关注潜在分布外背景区域。针对第二项挑战，我们提出新型基于sigmoid的分布外评分函数，综合利用ID标签与负标签，使分布内实例产生强响应而分布外实例受到抑制。大量实验表明，该方法在显著提升分布外检测性能的同时保持了分布内精度，例如与基线模型相比，在COCO数据集上将FPR95降低11.4%，在OpenImages数据集上降低25.5%。尽管最初为YOLO-World等密集型视觉语言模型检测器设计，我们成功将该方法适配至基于查询的视觉语言模型Transformer架构Grounding DINO并取得显著提升，证明了该框架的泛化能力。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.22537v2)

---

> ### 5. A Smart Classroom Behavior Analysis Framework with a New Highly Congested Classroom Dataset
> **🔹 中文标题：** 智能课堂行为分析框架；一种新型高密度课堂数据集
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
> 学生行为检测对智能课堂分析至关重要，但在大班授课场景下仍面临密集实例共存、非对称遮挡、深度方向尺度变化以及远距离目标细粒度语义退化等挑战。现有课堂行为数据集与通用检测器难以有效表征并解决这些难题。本文构建了高度拥挤课堂行为数据集，包含50,229个学生行为实例，涵盖阅读、书写、抬头、睡觉、四处张望、低头和使用手机七类行为。该数据集融合了密集分布、严重遮挡、尺度变化与细粒度行为语义等多重挑战特性。为应对此类问题，我们提出面向高密度课堂的YOLO检测框架ODER-HSFNet。其核心创新包含三个任务专用模块：可感知遮挡的可变形边缘修正器用于强化遮挡条件下的边界特征；超图状态空间融合模块整合局部结构增强、状态空间上下文建模与高阶关系聚合；遮挡校准检测头则通过抑制低质量预非极大值抑制候选框，降低遮挡边界与相邻实例导致的误检。在两个课堂行为检测数据集上的实验表明，ODER-HSFNet显著优于主流YOLO系列方法，在HCCB数据集上取得60.60%/80.12%的mAP50:95/mAP50指标，在SCB-D3-S数据集上达到57.36%/74.65%。消融实验进一步验证了所提设计对高密度课堂行为检测的有效性。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.21568v1)

---

<div align="center">

*由 [YOLO-Arxiv-Daily](https://github.com/WangQvQ/YOLO-Arxiv-Daily) 自动生成*

</div>