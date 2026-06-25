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
> **🔹 中文标题：** 基于机器学习的二维胶体组装识别：不同形状粒子的案例研究
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
> 本研究针对采用不同形状粒子（二维涂层）识别胶体单层组装体的问题展开，涉及球形、椭球形、长方体及棒状粒子，并将组装体分类为：孤立粒子、二聚体、链状结构、团簇和环状结构。研究采用YOLO模型作为识别方法，为四种粒子形状分别构建合成数据集以训练模型。论文探讨了基于合成数据训练模型在实验图像中的应用，并分析了该模型用于识别真实图像构型的可行性。虽然在人工图像上的识别近乎完美，但实验图像测试显示显著偏差：所有粒子类型的平均误差为43.1%，且数值分布差异明显——球体误差为20%，长方体误差达58.5%，表明算法对物体几何形状具有选择性敏感度。所构建的数据集与训练模型已开放获取，相应模块已集成至前期开发的信息系统（https://isanm.space/）。为进一步提升预测效果，需基于实验图像构建数据集。
>
> **💻 代码链接：** https://isanm.space/).
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.23639v1)

---

> ### 2. Fursee: Hybrid YOLO\-DINOv3 Framework for Fursuit Identity Retrieval and Clustering
> **🔹 中文标题：** Fursee：用于兽装身份检索与聚类的混合YOLO-DINOv3框架
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
> 全球兽装展会产出海量兽装照片，而人工分类带来沉重劳动成本，亟需开发自动身份检索与聚类方案。现有通用多模态模型缺乏针对复杂兽装场景的专项优化，且该领域尚无公开基准数据集。为填补这一空白，我们构建了专用兽装图像数据集，并提出三阶段混合流程Fursee，用于兽装身份检索与聚类。首先，YOLO检测并裁剪高分辨率兽装头部区域，提升小型重叠目标的定位精度；其次，ArcFace优化DINOv3嵌入向量，扩大特征超球面上不同身份的角间距；最后，DBSCAN进行无监督聚类，通过轮廓系数驱动搜索自动选择最优超参数，避免固定手动半径设定。检索与聚类实验表明，该流程在所有评估指标上均优于GPT5.5、Claude Opus 4.8和Qwen3.7-Plus等主流多模态模型，在兽装头部检索与分组任务中展现出优异性能。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.22872v1)

---

> ### 3. NegAS: Negative Label Guided Attention and Scoring for Out\-of\-Distribution Object Detection with Vision\-Language Models
> **🔹 中文标题：** NegAS：基于负标签引导的视觉-语言模型分布外目标检测注意力与评分机制
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
> 分布外（OOD）检测对于确保部署在安全关键应用中的目标检测系统的鲁棒性和可靠性至关重要。以往研究主要集中于单模态检测器或基于视觉语言模型（VLM）的分类器，而VLM目标检测器在OOD场景中的潜力仍待深入探索。本研究首次尝试基于VLM构建OOD目标检测方法。我们发现VLM检测器面临两个特定挑战：（i）其文本引导注意力机制会增强带有ID标签的前景区域，但对背景区域处理方式单一，未能有效利用潜在OOD区域来区分分布内（ID）与OOD样本；（ii）其基于Sigmoid的多标签输出机制与基于Softmax的OOD评分函数不兼容，亟需与VLM概率输出相匹配的评分函数。

为此，我们提出负标签引导注意力与评分（NegAS）框架。针对挑战（i），设计了负标签引导注意力模块（NegA），利用大语言模型生成的视觉相似但语义相异的负标签，引导注意力聚焦于潜在OOD背景区域。针对挑战（ii），提出新型基于Sigmoid的OOD评分函数（NegS），该函数同时利用ID标签与负标签，对ID样本产生强响应，对OOD样本产生抑制响应。大量实验表明，该方法在显著提升OOD检测性能的同时保持了ID检测精度——例如相比基线模型，在COCO数据集上将FPR95降低11.4%，在OpenImages数据集上降低25.5%。该方法虽最初面向YOLO-World等密集型VLM检测器设计，但我们成功将其迁移至基于查询的VLM Transformer架构Grounding DINO，并取得显著性能提升，验证了本框架的广泛适用性。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.22537v1)

---

> ### 4. A Smart Classroom Behavior Analysis Framework with a New Highly Congested Classroom Dataset
> **🔹 中文标题：** 基于新高密度教室数据集的智能教室行为分析框架
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
> 学生行为检测对智慧课堂分析至关重要，但在大班教学场景中，由于密集实例共现、非对称遮挡、深度尺度变化以及远距离目标细粒度语义退化等问题，仍面临严峻挑战。现有课堂行为数据集与通用检测器难以有效刻画并解决这些挑战。本文构建了高密度课堂行为数据集，包含阅读、书写、抬头、睡觉、左顾右盼、低头、使用手机七类共50,229个学生行为实例。HCCB数据集整合了密集分布、严重遮挡、尺度变化及细粒度行为语义等挑战特性，为研究提供基准测试平台。针对上述问题，本文提出基于YOLO架构的遮挡感知超图空间融合网络，其核心创新包含三个任务导向模块：遮挡感知可变形边缘修正器用于强化遮挡条件下的边界证据；超图状态空间融合模块通过局部结构增强、状态空间上下文建模与高阶关系聚合实现特征融合；遮挡校准检测头则能抑制低质量预NMS候选框，减少遮挡边界与相邻实例引发的误检。在两组课堂行为检测数据集上的实验表明，该网络性能超越主流YOLO系列方法，在HCCB数据集上达到60.60%/80.12%的mAP50:95/mAP50，在SCB-D3-S数据集上达到57.36%/74.65%。消融实验进一步验证了所提设计对高密度课堂行为检测的有效性。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.21568v1)

---

> ### 5. NeoLoc\-68: End\-to\-end 68\-point neonatal facial landmark localisation in neonatal clinical environments
> **🔹 中文标题：** NeoLoc-68：新生儿临床环境下的端到端68点面部关键点定位系统
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
> 面部关键点定位是实现自动化、非接触式新生儿疼痛评估方法的先决条件。临床医生使用疼痛量表来判断疼痛程度，其中许多量表依赖面部表情。然而，基于成人面部训练的关键点检测器在新生儿临床环境中表现不佳，原因包括医疗设备造成的频繁遮挡、多样的头部姿态以及具有挑战性的成像条件（如疼痛引发突然动作导致的运动模糊）。我们提出了一种端到端的面部关键点检测器，能够在临床环境中预测新生儿面部的68个关键点。我们整合了来自11个公开数据集的37,459张单人脸图像（标准化为68点标注），以及来自新生儿研究数据集的1,123帧手动标注帧（总计超过76,000个关键点）。该模型基于YOLO关键点架构进行适配以回归面部关键点，并使用预训练的新生儿面部检测器权重初始化。在公开数据集上，我们提出的模型实现了最先进的性能：归一化平均误差（NME）=5.37，失败率（FR）=12.5%，累积误差曲线下面积（AUC）在AUC0.08处为38.00%和AUC0.1处为48.70%。在临床新生儿测试集上，微调前模型在所有基线中取得了最低的检测失败率（DFR）=5.3%，并表现出强大的泛化能力。微调后性能进一步提升至NME=6.36，FR=22.30%，DFR=1.77%，AUC0.08=29.24%和AUC0.1=40.25%。据我们所知，这是首个端到端的68点新生儿面部关键点检测模型。随着数据集的进一步扩展和优化，该模型有望支持新生儿健康监测和疼痛相关面部分析的下游任务。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.20823v1)

---

<div align="center">

*由 [YOLO-Arxiv-Daily](https://github.com/WangQvQ/YOLO-Arxiv-Daily) 自动生成*

</div>