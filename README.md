<div align="center">

# YOLO ArXiv Daily

[![Daily Papers](https://img.shields.io/badge/📅-每日更新-blue)]()
[![arXiv](https://img.shields.io/badge/arXiv-最新论文-red)](https://arxiv.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)

*自动追踪 YOLO 相关最新论文，提供中英文双语摘要*

</div>

---

## 📑 论文列表

> ### 1. LiftNav: Path Planning via Semantic Lifting in TSDF\-Guided Gaussian Splatting
> **🔹 中文标题：** LiftNav：基于TSDF引导高斯喷溅的语义提升路径规划
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-05-29 |
> | 👤 作者 | Hannah Schieber |
>
> **📄 英文摘要：**
> Autonomous robots in unknown indoor environments require both reliable collision avoidance and object\-level understanding. Classical representations such as TSDF support safe planning but lack semantics, while photorealistic methods like Gaussian Splatting \(GS\) provide rich appearance yet suffer from soft geometry, limiting precise obstacle avoidance. We present LiftNav, a hybrid navigation framework built on GSFusion's TSDF\+GS dual map, augmented with a real\-time pipeline of YOLO\-based detection, TSDF\-based 3D lifting, and B\-spline trajectory optimization. This design enables flexible semantic navigation without dense 3D embeddings. We further introduce a hinge\-loss\-based collision penalty that improves trajectory smoothness and safety. We evaluate our approach in a simulation using the Replica dataset. Compared against a state\-of\-the\-art radiance field baseline we show a 100% feasibility rate and shorter trajectories.
>
> **📝 中文摘要：**
> 在未知室内环境中的自主机器人既需要可靠的避障能力，也需要对物体层面的理解能力。传统表示方法如TSDF支持安全规划但缺乏语义信息，而高保真视觉方法如高斯溅射（GS）虽提供丰富外观信息却存在几何模糊性，限制了精确避障。本文提出LiftNav混合导航框架，基于GSFusion的TSDF+GS双地图构建，通过集成基于YOLO的检测、TSDF三维提升和B样条轨迹优化的实时处理流程，实现无需密集三维嵌入的灵活语义导航。我们进一步引入基于铰链损失的碰撞惩罚机制，以提升轨迹平滑性与安全性。通过Replica数据集仿真实验验证，相较于最新辐射场基线方法，本方法实现了100%可行率与更短的轨迹路径。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2605.31376v1)

---

> ### 2. Intra\-YOLO: A Small Object Detection Model for Caries and Molar\-Incisor Hypomineralization in Intraoral Photography Based on Transfer Learning with Reinforcement Learning
> **🔹 中文标题：** 基于迁移学习与强化学习的口腔摄影龋齿及磨牙-切牙釉质发育不全小型目标检测模型：Intra-YOLO
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-05-27 |
> | 👤 作者 | Po\-Lun Chwang |
>
> **📄 英文摘要：**
> This study developed a computer\-aided diagnosis \(CAD\) system for detecting caries and molar\-incisor hypomineralization \(MIH\) in intraoral photographs. These lesions share similar appearances, making clinical differentiation challenging, especially given their small size and variability in imaging conditions.
>
> **📝 中文摘要：**
> 本研究开发了一种计算机辅助诊断系统，用于在口内照片中检测龋齿和磨牙切牙矿化不全症。这两种病变形态相似，临床鉴别困难，尤其是考虑到其病灶尺寸较小且拍摄条件存在差异。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2605.28157v1)

---

> ### 3. PlayClass: Automated Play Behaviour Classification in Poultry
> **🔹 中文标题：** PlayClass：家禽自动玩耍行为分类
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-05-26 |
> | 👤 作者 | Prince Ravi Leow |
>
> **📄 英文摘要：**
> Automated monitoring of animal welfare has largely targeted negative indicators, leaving positive welfare behaviours such as play underexplored. To address this gap, we present PlayClass, a pipeline for play\-behaviour classification in poultry from top\-down pen video. The pipeline leverages long\-duration tracking with SAM 3 via YOLO\-guided chunk boundaries to minimise identity errors in point\-based prompting, and frozen embeddings from image and video foundation models for play action classification. Although handcrafted motion features from tracked masks alone achieved competitive accuracy, V\-JEPA 2.1 consistently outperformed all other backbones across model scales, reaching 77.0 macro\-averaged F$\_1$ when combined with handcrafted features. Despite this result, the dataset remains challenging due to play sub\-types sharing similar kinematic profiles with non\-play and inter\-bird occlusion. Overall, our work provides encouraging evidence towards automated frameworks for play behaviour classification in poultry.
>
> **📝 中文摘要：**
> 动物福利自动监控长期以负面指标为主导，对游戏等积极福利行为的研究尚存空白。为填补这一缺口，我们推出PlayClass——一种基于俯视圈舍视频的禽类游戏行为分类流程。该流程通过YOLO引导的分段边界，结合SAM 3实现长期追踪，以降低基于点提示的身份识别误差，并利用图像与视频基础模型的冻结嵌入进行游戏动作分类。尽管仅从追踪掩膜提取的人工设计运动特征已能达到具有竞争力的准确率，但V-JEPA 2.1在不同模型规模下始终优于其他骨干网络，当其与人工设计特征结合时，宏观平均F1值达到77.0。尽管取得此成果，由于游戏子类型与非游戏行为具有相似的运动学特征，且存在个体间遮挡，该数据集仍具挑战性。总体而言，我们的研究为禽类游戏行为分类的自动化框架构建提供了积极证据。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2605.27304v1)

---

> ### 4. Small Object Detection in Industrial Recycling: A New Dataset and YOLO Performance Evaluation
> **🔹 中文标题：** 工业回收中的小目标检测：新型数据集与YOLO性能评估
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-05-26 |
> | 👤 作者 | Oussama Messai |
>
> **📄 英文摘要：**
> In this paper, we address the problem of detecting small, dense, and overlapping objects, a major challenge in computer vision. Our focus is on reviewing proposed methods based on deep learning supervised approaches. We provide a detailed comparison of these systems on a new dataset of more than 10k images and 120k instances, highlighting their performance, accuracy, and computational efficiency in the industrial recycling process use case. Through this comparative analysis, we identify the most reliable systems currently available and the specific challenges they are designed to tackle. Furthermore, we explore the benefits of data augmentation and synthetic images. Based on our analysis, we also propose potential future directions and innovative solutions that could enhance the effectiveness of small, dense and overlapped object detection systems. The scope of our investigations encompasses object detection, length measurement, and anomaly detection within the context of the recycling process. The anomaly detection strategy is robust against variations in image resolution and zoom levels, ensuring reliable performance in industrial applications. The repository of the proposed dataset, methods and evaluation codes can be found at: https://github.com/o\-messai/SDOOD
>
> **📝 中文摘要：**
> 本文针对小尺寸、高密度且重叠物体的检测难题展开研究，该问题在计算机视觉领域具有显著挑战性。研究重点聚焦于对基于深度学习监督方法的现有技术进行系统综述。通过在包含超过1万张图像及12万标注实例的新型数据集上开展对比实验，我们深入分析了各类检测系统在工业回收场景下的性能表现、检测精度与计算效率。基于比较分析结果，我们识别出当前最具可靠性的检测系统及其针对性解决的具体技术挑战，并进一步探讨了数据增强与合成图像技术的应用价值。在理论分析基础上，本文提出了提升小尺寸、高密度重叠物体检测系统效能的潜在研究方向与创新方案。研究范围涵盖回收流程中的物体检测、尺寸测量及异常检测，其中异常检测策略具有抗图像分辨率与缩放尺度变化的能力，可确保工业应用中的稳定性能。所提出的数据集、检测方法及评估代码库可通过以下链接获取：https://github.com/o-messai/SDOOD
>
> **💻 代码链接：** https://github.com/o-messai/SDOOD
>
> 🔗 [阅读论文](http://arxiv.org/abs/2605.26884v1)

---

> ### 5. Pixel\-Level Pavement Distress Assessment Using Instance Segmentation
> **🔹 中文标题：** 基于实例分割的像素级路面病害评估
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-05-25 |
> | 👤 作者 | Logan Dewick |
>
> **📄 英文摘要：**
> Automated pavement distress assessment requires more than image\-level classification or coarse bounding box detection, demanding precise localization of thin, branching, and irregular cracks to achieve the geometric precision necessary for maintenance\-relevant quantification. This paper presents a vision\-based pavement distress analysis system based on Mask R\-CNN instance segmentation and evaluates it on UWGB\-StreetCrack, a custom field\-collected roadway image dataset acquired with a vehicle\-mounted smartphone and manually annotated with polygon labels for longitudinal cracks, transverse cracks, alligator cracks, and potholes. Five Detectron2\-based Mask R\-CNN backbone variants were considered under a consistent fine\-tuning protocol. The best\-performing model, Mask R\-CNN with a ResNet\-101 FPN backbone, achieved 84.23% precision, 90.04% recall, and an F1 score of 87.04% under the project\-specific bounding\-box matching protocol. The same model produced an aggregate predicted crack\-area fraction of 2.164%, closely matching the 2.170% ground\-truth crack\-area fraction. To contextualize the segmentation system against a detector\-oriented alternative, a CSPDarknet53\-based YOLO detector was also adapted and retrained on the dataset, reaching 27.5% precision and 20.7% recall on the validation protocol. The results show that instance segmentation is a practical direction for field pavement imagery and aggregate crack\-area estimation, while also exposing open challenges in annotation consistency, class imbalance, confounder rejection, and mask\-level benchmarking.
>
> **📝 中文摘要：**
> 路面病害的自动化评估不仅需要图像级分类或粗边界框检测，更要求精确定位细长、分支状且形态不规则的裂缝，以满足养护量化所需的几何精度。本文提出一种基于Mask R-CNN实例分割的视觉化路面病害分析系统，并在定制采集的道路图像数据集UWGB-StreetCrack上进行评估。该数据集通过车载智能手机采集现场图像，对纵向裂缝、横向裂缝、网状裂缝及坑洼进行多边形标注。在统一微调框架下，对比了五种基于Detectron2的Mask R-CNN骨干网络变体。其中采用ResNet-101 FPN骨干网络的Mask R-CNN模型表现最优，在项目特定边界框匹配协议中达到84.23%精确率、90.04%召回率和87.04%的F1分数。该模型预测的裂缝面积占比为2.164%，与真实值2.170%高度吻合。为对比检测器方案，本文还采用基于CSPDarknet53的YOLO检测器在相同数据集上重新训练，其在验证协议中仅达到27.5%精确率和20.7%召回率。研究表明，实例分割技术是处理现场路面图像与整体裂缝面积估算的实用方向，同时也揭示了标注一致性、类别不平衡、干扰因素排除及掩膜级评估基准构建等待解难题。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2605.26095v1)

---

<div align="center">

*由 [YOLO-Arxiv-Daily](https://github.com/WangQvQ/YOLO-Arxiv-Daily) 自动生成*

</div>