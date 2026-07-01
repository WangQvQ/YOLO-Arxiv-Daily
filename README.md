<div align="center">

# YOLO ArXiv Daily

[![Daily Papers](https://img.shields.io/badge/📅-每日更新-blue)]()
[![arXiv](https://img.shields.io/badge/arXiv-最新论文-red)](https://arxiv.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)

*自动追踪 YOLO 相关最新论文，提供中英文双语摘要*

</div>

---

## 📑 论文列表

> ### 1. Character Recognition of Nepali Number Plate
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-06-27 |
> | 👤 作者 | Satyasa Khadka |
>
> **📄 英文摘要：**
> This paper presents a robust Automatic Number Plate Recognition \(ANPR\) system tailored for Nepali license plates written in Devanagari script. In this paper, a pipelined model was used that integrates YOLO\-based models for license plate and character detection, followed by a CNN classifier trained on 34 Devanagari characters. Two publicly available data sets were used that incorporate diverse lighting, fonts, and structural variations. Data augmentation and additional training on embossed plates enhanced the generalizability of the model. The system achieved a recognition accuracy of up to 93%, demonstrating strong performance under real\-world conditions and providing a scalable solution for traffic management in Nepal. Code: https://github.com/Satyasakhadka/Nepali\-NumberPlate\-Character\-Recognition
>
> **💻 代码链接：** https://github.com/Satyasakhadka/Nepali-NumberPlate-Character-Recognition
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.28946v1)

---

> ### 2. Virtual Ring Try\-On
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-06-27 |
> | 👤 作者 | Vishnu D. Burkhawala |
>
> **📄 英文摘要：**
> This paper presents an innovative approach that enables the users to capture their hand and try the jewel ring on their hand. The user captures the image of the hand using the React Native base GUI of the mobile application and selects the ring that the user wants to try, and the output image will have the user's hand with the ring image. This approach is implemented using a combination of MediaPipe hand point detection and YOLO\-V8 custom object detection. The hand image uploaded by the user first undergoes mediapipe hand point detection. It will give the hand points and a Region of Interest mask where the ring is going to be placed. Then the ring is passed through YOLO object detection, in which ring points are detected, and background is removed. After that, using vector algebra, the angular discrepancy between the finger's reference axis and the ring's principal axis is computed. Also, ring size is rescaled according to finger thickness, preserving the aspect ratio to maintain perceptual realism. Then the ring is placed on the hand image and the output image is generated and shown on the user screen.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.28792v1)

---

> ### 3. Identifying the Unknown: Prompt\-Free Open Vocabulary Anomaly Recognition for Robot\-Object Interaction
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

> ### 4. Application of Machine Learning for the Identification of 2D Colloidal Assemblies: A Case Study on Particles of Distinct Shapes
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

> ### 5. Fursee: Hybrid YOLO\-DINOv3 Framework for Fursuit Identity Retrieval and Clustering
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

<div align="center">

*由 [YOLO-Arxiv-Daily](https://github.com/WangQvQ/YOLO-Arxiv-Daily) 自动生成*

</div>