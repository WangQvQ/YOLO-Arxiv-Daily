<div align="center">

# YOLO ArXiv Daily

[![Daily Papers](https://img.shields.io/badge/📅-每日更新-blue)]()
[![arXiv](https://img.shields.io/badge/arXiv-最新论文-red)](https://arxiv.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)

*自动追踪 YOLO 相关最新论文，提供中英文双语摘要*

</div>

---

## 📑 论文列表

> ### 1. Application of Machine Learning for the Identification of 2D Colloidal Assemblies: A Case Study on Particles of Distinct Shapes
> **🔹 中文标题：** 机器学习在二维胶体组装体识别中的应用：以不同形状的粒子为例
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
> 本研究致力于解决使用不同形状的二维涂层粒子——球体、椭球体、长方体与棒状体——识别胶体单层组装体的问题。文中对以下组装体分类进行探讨：孤立粒子、二聚体、链状结构、团簇及环状结构。研究选取YOLO模型作为识别方法，针对四种粒子形状分别构建合成数据集以训练模型。文章重点讨论了基于合成数据训练的模型在实验图像上的应用，并分析了此类模型识别真实图像中构型的可行性。虽然在人工图像上的识别近乎完美，但对实验图像的测试结果显示出显著偏差：所有粒子类型的平均误差达43.1%，且数值离散度较大——球体为20%，长方体则高达58.5%，表明该算法对几何形状存在选择性敏感度。所创建的数据集与训练模型已开源供公众使用，相应模块已集成至前期开发的信息系统（https://isanm.space/）。为提升预测精度，亟需基于实验图像构建更完善的数据集。
>
> **💻 代码链接：** https://isanm.space/).
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.23639v1)

---

> ### 2. Fursee: Hybrid YOLO\-DINOv3 Framework for Fursuit Identity Retrieval and Clustering
> **🔹 中文标题：** 毛皮服：基于YOLO-DINOv3混合框架的身份检索与聚类方法
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
> 全球兽装聚会产生海量兽装摄影图像，而人工分类面临高昂的人力成本，亟需自动化身份检索与聚类解决方案。通用多模态模型缺乏针对复杂兽装场景的专项优化，且该领域尚无公开基准数据集。为填补这一空白，我们构建了专用兽装图像数据集，并提出名为"Fursee"的三阶段混合流程，用于兽装身份检索与聚类：首先，采用YOLO检测并裁剪高分辨率兽装头部区域，以提升对小型重叠目标的定位精度；其次，通过ArcFace优化DINOv3嵌入向量，在特征超球面上扩大不同身份间的角度分离度；最后，利用DBSCAN进行无监督聚类，通过轮廓系数驱动的搜索机制自动选择最优超参数，替代固定的人工设定半径。检索与聚类实验表明，该流程在所有评估指标上均优于GPT5.5、Claude Opus 4.8和Qwen3.7-Plus等主流多模态模型，在兽装头部检索与分组任务中展现出卓越性能。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.22872v1)

---

> ### 3. NegAS: Negative Label Guided Attention and Scoring for Out\-of\-Distribution Object Detection with Vision\-Language Models
> **🔹 中文标题：** NegAS：基于视觉语言模型的分布外目标检测的负标签引导注意力与评分机制
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
> 分布外检测对于确保部署在安全关键应用中的目标检测系统的鲁棒性和可靠性至关重要。现有研究主要集中于单模态检测器或基于视觉语言模型的分类器，而基于VLM的目标检测器在OOD场景中的潜力尚未得到充分探索。本研究首次尝试在VLM基础上构建分布外目标检测方法。我们发现VLM检测器面临两个特定挑战：（i）其文本引导注意力机制虽能增强具有分布内标签的前景区域，却对背景采取统一处理，未能有效利用潜在OOD区域实现分布内与分布外实例的分离；（ii）基于sigmoid的多标签输出与基于softmax的OOD评分机制存在不兼容性，亟需与VLM概率输出特性相匹配的评分函数。

为此，我们提出负标签引导注意力与评分框架。针对挑战（i），设计负标签引导注意力模块：通过大语言模型生成视觉相似但语义相异的负标签，引导注意力聚焦潜在OOD背景区域。针对挑战（ii），创新性地构建基于sigmoid的OOD评分函数，同时利用ID与负标签信息，使分布内实例产生强响应而分布外实例受到抑制。

大量实验证明，该方法在保持分布内检测精度的同时显著提升OOD检测性能。相比基线模型，其在COCO数据集降低FPR95达11.4%，在OpenImages数据集降低达25.5%。尽管最初为YOLO-World等稠密型VLM检测器设计，NegAS框架在基于查询的VLM转换器Grounding DINO上同样取得显著改进，验证了该框架的普适性。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.22537v1)

---

> ### 4. A Smart Classroom Behavior Analysis Framework with a New Highly Congested Classroom Dataset
> **🔹 中文标题：** 智能课堂行为分析框架与新建高拥挤度课堂数据集
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
> 学生行为检测对智能课堂分析至关重要，但在大班教学场景中仍面临诸多挑战，包括密集实例共现、非对称遮挡、深度尺度变化以及远距离目标的细粒度语义退化等问题。现有的课堂行为数据集和通用检测器难以充分刻画并解决这些挑战。本文构建了高度拥挤课堂行为数据集，包含阅读、书写、抬头、睡觉、四处张望、低头和使用手机七类共50,229个学生行为实例。该数据集通过整合密集分布、严重遮挡、尺度变化与细粒度行为语义，提供了具有挑战性的基准平台。针对上述问题，本文提出ODER-HSFNet——一种专为高度拥挤课堂设计的基于YOLO的检测框架。其核心创新包含三个任务导向模块：遮挡感知可变形边缘校正器能增强遮挡条件下的边界证据；超图状态空间融合模块融合了局部结构增强、状态空间上下文建模与高阶关系聚合；遮挡校准检测头则能抑制非极大值抑制前的低质量候选框，减少由遮挡边界与邻近实例产生的误检。在两个课堂行为检测数据集上的实验表明，ODER-HSFNet优于主流YOLO系列方法，在HCCB数据集上实现60.60%/80.12%的mAP50:95/mAP50指标，在SCB-D3-S数据集上达到57.36%/74.65%的检测精度。消融研究进一步验证了所提方法在高度拥挤课堂行为检测中的有效性。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.21568v1)

---

> ### 5. NeoLoc\-68: End\-to\-end 68\-point neonatal facial landmark localisation in neonatal clinical environments
> **🔹 中文标题：** NeoLoc-68:在新生儿临床环境中实现端到端的68点新生儿面部关键点定位
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-06-18 |
> | 👤 作者 | Abdullah Bin\-Obaid |
>
> **📄 英文摘要：**
> Facial landmark localisation is a prerequisite for developing automated, non\-contact neonatal pain assessment methods. Clinicians use pain scales to judge the severity of pain, many of which rely on facial expression. However, facial landmark detectors trained on adult faces perform poorly in neonatal clinical environments due to frequent occlusions caused by medical equipment, varied head poses, and challenging imaging conditions, including motion blur triggered by sudden pain\-related movements. We propose an end\-to\-end facial landmark detector capable of predicting 68 landmarks on neonatal faces in clinical environments. We combined 37,459 single\-face images from 11 public datasets, standardised to 68\-point markup, with 1,123 manually annotated frames from a neonatal research dataset \(totalling over 76,000 landmarks\). A YOLO\-based keypoint model was adapted to regress the facial landmarks, initialised with weights from a pretrained neonatal face detector. On public datasets, our proposed model achieved state\-of\-the\-art performance: Normalised Mean Error \(NME\) = 5.37, Failure Rate \(FR\) = 12.5%, Area Under the Cumulative Error Curve \(AUC\) at AUC0.08 = 38.00% and AUC0.1 = 48.70%. On the clinical neonatal test set, before fine\-tuning, the model achieved the lowest Detection Failure Rate \(DFR\) = 5.3% among all baselines and showed strong generalisation. After fine\-tuning, performance improved further to NME = 6.36, FR = 22.30%, DFR = 1.77%, AUC0.08 = 29.24% and AUC0.1 = 40.25%. To the best of our knowledge, this represents the first end\-to\-end 68\-point neonatal facial landmark detection model. With further dataset expansion and refinement, it could support downstream tasks in neonatal health monitoring and pain\-related facial analysis.
>
> **📝 中文摘要：**
> 面部关键点定位是开发自动化、非接触式新生儿疼痛评估方法的先决条件。临床医生通过疼痛量表评估疼痛程度，许多量表依赖面部表情判断。然而，基于成人面部训练的关键点检测器在新生儿临床环境中表现欠佳，这源于医疗设备造成的频繁遮挡、多变的头部姿态以及复杂的成像条件（包括疼痛相关动作引发的运动模糊）。本文提出一种端到端的面部关键点检测器，能够在临床环境中预测新生儿面部的68个关键点。我们整合了11个公开数据集中的37,459张单人面部图像（标准化为68点标注）与新生儿研究数据集中1,123帧人工标注图像（总计超过76,000个关键点），并改进基于YOLO的关键点模型进行面部关键点回归，采用预训练新生儿面部检测器的权重进行初始化。在公开数据集上，本模型达到最优性能：归一化平均误差为5.37，失败率为12.5%，累积误差曲线下面积在AUC0.08标准下为38.00%，AUC0.1标准下为48.70%。在临床新生儿测试集上，微调前模型以5.3%的检测失败率在所有基准模型中表现最佳，展现出强大的泛化能力；微调后性能进一步提升至归一化平均误差6.36、失败率22.30%、检测失败率1.77%、AUC0.08标准下29.24%、AUC0.1标准下40.25%。据我们所知，这是首个端到端的68点新生儿面部关键点检测模型。随着数据集的扩展与完善，该模型有望支持新生儿健康监测及疼痛相关面部分析的下游任务。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.20823v1)

---

<div align="center">

*由 [YOLO-Arxiv-Daily](https://github.com/WangQvQ/YOLO-Arxiv-Daily) 自动生成*

</div>