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
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-06-25 |
> | 👤 作者 | Philipp Allgeuer |
>
> **📄 英文摘要：**
> Robots operating in real\-world environments must in general be able to recognize previously unseen objects. As robotic systems move toward open\-world autonomy, there is a growing, yet largely unmet, need for open vocabulary object detectors that are prompt\-free and efficient enough for continuous deployment. We present AnomNOVIC, a two\-stage known\-workspace framework that combines a masked autoencoder \(MAE\) trained for anomaly detection, with NOVIC, a powerful real\-time prompt\-free open vocabulary image classifier. The MAE produces generic object\-agnostic bounding boxes, allowing NOVIC to classify salient image regions without requiring a predefined candidate class list. We evaluate AnomNOVIC against strong open vocabulary baselines in a tabletop robot\-object environment featuring the NICOL humanoid robot, reaching 47.1% AP / 57.5% AP50 for prompt\-free recognition, and 59.0% AP / 72.5% AP50 if class candidates are provided. Across additional datasets, including an in\-the\-wild test set with 48 unique objects, AnomNOVIC reaches up to 82.6% prompt\-free detection and classification accuracy. These results significantly surpass all tested open vocabulary baselines, including YOLO\-World\-v2, OWLv2, and YOLOE.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.26829v1)

---

> ### 2. Application of Machine Learning for the Identification of 2D Colloidal Assemblies: A Case Study on Particles of Distinct Shapes
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-06-22 |
> | 👤 作者 | L. T. Khusainova |
>
> **📄 英文摘要：**
> This work addresses the problem of identifying colloidal monolayer assemblies using particles of various shapes \(two\-dimensional coatings\): spheres, ellipsoids, cuboids, and rods. The following classification of assemblies is considered: isolated particles, dimers, chains, clusters, and loops. The YOLO model was chosen as the identification method. Synthetic datasets were prepared for each of the four particle shapes to train the models. The paper discusses the application of models trained on synthetic data to experimental images. An analysis was carried out on the feasibility of using such models for recognizing configurations in real images. While recognition on artificial images is nearly perfect, tests on experimental images showed a significant deviation. The average error across all particle types was 43.1%, but a considerable spread in values is observed: from 20% for spheres to 58.5% for cuboids, indicating the algorithm's selective sensitivity to object geometry. The created datasets and trained models are freely available for use. The corresponding modules have been integrated into the previously developed information system \(https://isanm.space/\). To further improve prediction results, it is necessary to prepare datasets based on experimental images.
>
> **💻 代码链接：** https://isanm.space/).
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.23639v1)

---

> ### 3. Fursee: Hybrid YOLO\-DINOv3 Framework for Fursuit Identity Retrieval and Clustering
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-06-22 |
> | 👤 作者 | Jundi Wu |
>
> **📄 英文摘要：**
> Global furry conventions produce massive fursuit photographs, while manual sorting brings heavy labor costs and calls for automatic identity retrieval and clustering solutions. General multimodal models lack dedicated optimization for complex fursuit scenes, and no public benchmark dataset exists for this task. To fill this gap, we build a specialized fursuit image dataset and present a three\-stage hybrid pipeline Fursee for fursuit identity retrieval and clustering. First, YOLO detects and crops high\-resolution fursuit head patches to improve localization of small and overlapping targets. Second, ArcFace optimizes DINOv3 embeddings to enlarge angular separation between different identities on the feature hypersphere. Third, DBSCAN performs unsupervised clustering, with silhouette\-coefficient\-driven search automatically selecting optimal hyperparameters rather than fixed manual radius. Retrieval and clustering experiments verify that our pipeline outperforms mainstream multimodal models including GPT5.5, Claude Opus 4.8 and Qwen3.7\-Plus on all evaluation metrics, achieving competitive performance for fursuit head retrieval and grouping.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.22872v1)

---

> ### 4. NegAS: Negative Label Guided Attention and Scoring for Out\-of\-Distribution Object Detection with Vision\-Language Models
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-06-21 |
> | 👤 作者 | Yingjie Zhang |
>
> **📄 英文摘要：**
> Out\-of\-Distribution \(OOD\) detection is essential for ensuring the robustness and reliability of object detection systems deployed in safety\-critical applications. While prior research has mainly focused on uni\-modal detectors or vision\-language model \(VLM\) based classifiers, the potential of VLM\-based object detectors in OOD scenarios remains underexplored. In this work, we take the first step toward building OOD object detection methods upon VLMs. We identify two challenges specific to VLM detectors: \(i\) their text\-guided attention enhances foreground with ID labels but treats background uniformly, leaving potential OOD regions unexploited for separating in\-distribution \(ID\) from OOD instances; and \(ii\) their sigmoid\-based multi\-label outputs are incompatible with softmax\-based OOD scores, calling for scoring functions consistent with VLM probabilistic outputs. Hence, we introduce Negative Label Guided Attention and Scoring \(NegAS\). To address \(i\), we propose a negative label guided attention module \(NegA\), where LLM\-generated, visually\-similar but semantically\-different negative labels are used to guide attention toward potential OOD background regions. To address \(ii\), we introduce a novel sigmoid\-based OOD scoring function \(NegS\) that leverages both ID and negative labels, producing strong responses for ID instances and suppressed responses for OOD ones. Extensive experiments demonstrate that our approach improves OOD detection performance by a large margin while maintaining ID accuracy, e.g., reducing the FPR95 by 11.4% on the COCO dataset and 25.5% on the OpenImages dataset compared to the baseline model. While initially designed for dense VLM detectors like YOLO\-World, we successfully adapt NegAS to Grounding DINO, a query\-based VLM transformer and achieve significant improvements, demonstrating the generalizability of our framework.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.22537v2)

---

> ### 5. A Smart Classroom Behavior Analysis Framework with a New Highly Congested Classroom Dataset
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-06-19 |
> | 👤 作者 | Wei Xu |
>
> **📄 英文摘要：**
> Student behavior detection is important for intelligent classroom analysis but remains challenging in large\-class scenarios due to dense instance co\-occurrence, asymmetric occlusion, depth\-wise scale variation, and fine\-grained semantic degradation in distant targets. Existing classroom behavior datasets and general\-purpose detectors are insufficient to characterize and address these challenges. This paper constructs the Highly Congested Classroom Behavior \(HCCB\) dataset, containing 50,229 student behavior instances across seven categories: reading, writing, heads up, sleeping, looking around, bowing head, and using phone. HCCB provides a challenging benchmark that integrates dense distributions, severe occlusion, scale variation, and fine\-grained behavioral semantics. To address these issues, we propose ODER\-HSFNet, a YOLO\-based detection framework tailored to highly crowded classrooms. At its core, ODER\-HSFNet introduces three task\-specific innovations: the Occlusion\-aware Deformable Edge Rectifier \(ODER\), which strengthens boundary evidence under occlusion; the Hypergraph\-State Spatial Fusion \(HSSF\) module, which integrates local structure enhancement, state\-space contextual modeling, and high\-order relation aggregation; and the Occlusion\-Calibrated Detection Head \(OCDetect\), which suppresses low\-quality Pre\-NMS candidates and reduces false positives from occlusion boundaries and neighboring instances. Experiments on two classroom behavior detection datasets show that ODER\-HSFNet outperforms mainstream YOLO\-series methods, achieving 60.60%/80.12% mAP50:95/mAP50 on HCCB and 57.36%/74.65% on SCB\-D3\-S. Ablation studies further verify the effectiveness of the proposed design for highly crowded classroom behavior detection.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.21568v1)

---

<div align="center">

*由 [YOLO-Arxiv-Daily](https://github.com/WangQvQ/YOLO-Arxiv-Daily) 自动生成*

</div>