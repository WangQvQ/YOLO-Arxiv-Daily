<div align="center">

# YOLO ArXiv Daily

[![Daily Papers](https://img.shields.io/badge/📅-每日更新-blue)]()
[![arXiv](https://img.shields.io/badge/arXiv-最新论文-red)](https://arxiv.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)

*自动追踪 YOLO 相关最新论文，提供中英文双语摘要*

</div>

---

## 📑 论文列表

> ### 1. Intra\-YOLO: A Small Object Detection Model for Caries and Molar\-Incisor Hypomineralization in Intraoral Photography Based on Transfer Learning with Reinforcement Learning
> **🔹 中文标题：** 基于迁移学习与强化学习的Intra-YOLO：口腔摄影中龋齿与磨牙-切牙釉质发育不全的小目标检测模型
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
> 本研究开发了一套计算机辅助诊断系统，用于检测口内照片中的龋齿和磨牙切牙矿化不全病变。这两种病变外观相似，临床鉴别困难，尤其考虑到其体积较小且成像条件多变。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2605.28157v1)

---

> ### 2. PlayClass: Automated Play Behaviour Classification in Poultry
> **🔹 中文标题：** PlayClass: 家禽游戏行为的自动分类
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
> 动物福利自动化监测主要针对负面指标，而玩耍等积极福利行为仍待深入研究。为此，我们提出PlayClass——一种基于俯视禽栏视频的玩耍行为分类流程。该流程采用YOLO引导分段结合SAM 3的长时跟踪技术，通过点提示机制最小化身份识别错误，并利用图像与视频基础模型的冻结嵌入特征进行玩耍动作分类。尽管仅基于跟踪掩模的手工运动特征已能获得相当精度，但V-JEPA 2.1在不同模型规模中始终优于其他骨干网络，当其与手工特征结合时达到77.0的宏观平均F1值。尽管如此，该数据集仍具挑战性，因为玩耍亚型与非玩耍行为具有相似的运动学特征，且存在禽体间遮挡问题。总体而言，本研究为禽类玩耍行为的自动化分类框架提供了令人鼓舞的实证支持。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2605.27304v1)

---

> ### 3. Small Object Detection in Industrial Recycling: A New Dataset and YOLO Performance Evaluation
> **🔹 中文标题：** 工业回收中的小物体检测：新数据集与YOLO性能评估
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
> 本文针对计算机视觉领域中小目标、密集目标及重叠目标检测这一重要挑战展开研究。文章系统梳理了基于监督式深度学习方法的相关进展，在包含超过1万张图像与12万目标实例的新数据集上对各类方法进行详细比较，重点分析了它们在工业回收流程应用场景中的性能表现、检测精度及计算效率。通过对比分析，我们评估了当前最可靠的检测系统及其针对的特定挑战，同时探讨了数据增强与合成图像技术带来的效益。基于研究结论，本文提出了未来研究方向与创新解决方案，以提升小目标、密集目标与重叠目标检测系统的效能。本研究的范畴涵盖回收流程中的目标检测、尺寸测量与异常检测任务，其中异常检测策略具备对图像分辨率及缩放尺度变化的强鲁棒性，可满足工业应用需求。所提出的数据集、方法框架及评估代码已开源于：https://github.com/o-messai/SDOOD
>
> **💻 代码链接：** https://github.com/o-messai/SDOOD
>
> 🔗 [阅读论文](http://arxiv.org/abs/2605.26884v1)

---

> ### 4. Pixel\-Level Pavement Distress Assessment Using Instance Segmentation
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
> 自动化路面损伤评估不仅需要图像级分类或粗略的边界框检测，还需精确识别细长、分支状及不规则裂缝的几何形态，以实现维护相关量化所需的几何精度。本文提出一种基于Mask R-CNN实例分割的视觉化路面损伤分析系统，并在定制化的UWGB-StreetCrack路面图像数据集上进行评估。该数据集通过车载智能手机采集，包含纵向裂缝、横向裂缝、鳄鱼裂纹及坑洼等多类损伤，并采用多边形标注进行精确标记。在统一微调框架下，对五种基于Detectron2的Mask R-CNN骨干网络变体进行测试。其中采用ResNet-101 FPN骨干网络的Mask R-CNN表现最优：在项目定制化边界框匹配协议下，其精确率达到84.23%，召回率为90.04%，F1分数为87.04%。该模型预测的裂缝面积占比为2.164%，与真实值2.170%高度吻合。为对比分割系统与检测器方案的差异，研究还采用基于CSPDarknet53的YOLO检测器进行适配训练，其在验证协议中精确率为27.5%，召回率为20.7%。结果表明，实例分割技术在实地路面影像分析与裂缝面积估算中具有实用价值，同时也揭示了标注一致性、类别不平衡、干扰因素排除及掩膜级基准测试等开放性挑战。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2605.26095v1)

---

> ### 5. SAM3\-Assisted Training of Lightweight YOLO Models for Precision Pig Farming
> **🔹 中文标题：** SAM3辅助的轻量化YOLO模型训练助力精准生猪养殖
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-05-25 |
> | 👤 作者 | Marcos Vinicius Mendes Faria |
>
> **📄 英文摘要：**
> Deep learning\-based object detection has revolutionized Precision Livestock Farming \(PLF\), yet a critical barrier remains: high\-performance Foundation Models \(such as SAM 3\) are too computationally intensive for edge deployment, while lightweight models \(like YOLO\) require prohibitive manual annotation efforts. This work proposes a fully automated knowledge distillation pipeline that leverages the Segment Anything Model 3 \(SAM 3\) to generate zero\-shot pseudo\-labels for training efficient YOLOv8 detectors. By treating SAM 3 as an offline auto\-annotator, we eliminate the manual labeling bottleneck, producing models capable of real\-time inference on resource\-constrained hardware. We systematically evaluate this approach on the PigLife dataset, comparing SAM 3\-supervised models against human\-annotated baselines. Results demonstrate that a SAM 3\-trained YOLOv8m achieves a mean Average Precision \(mAP\) of 79.4% without human intervention, while reducing inference latency by approximately 200$times$ compared to the teacher model. Furthermore, stratified analysis reveals that in low\-occlusion scenarios, the automated pipeline achieves detection rates comparable to human benchmarks \($AP\_\{50\} > 99%$\). These findings indicate that foundation models can serve as effective, zero\-annotation\-cost supervisors, enabling scalable edge computing solutions for smart agriculture.
>
> **📝 中文摘要：**
> 基于深度学习的目标检测技术为精准畜牧养殖带来了革命性变革，但当前仍存在一个关键瓶颈：高性能基础模型（如SAM 3）因计算成本过高而难以部署于边缘设备，而轻量化模型（如YOLO系列）则需要耗费大量人工标注成本。本研究提出了一种全自动知识蒸馏流程，利用分段一切模型3（SAM 3）生成零样本伪标签来训练高效的YOLOv8检测器。通过将SAM 3作为离线自动标注器，我们突破了人工标注的瓶颈，训练出的模型可在资源受限硬件上实现实时推理。我们在PigLife数据集上系统评估了该方法，对比了SAM 3监督模型与人工标注基准模型的性能。实验结果表明，经SAM 3训练的YOLOv8m在无需人工干预的情况下达到79.4%的平均精度均值（mAP），同时推理延迟相比教师模型降低约200倍。分层分析进一步揭示，在低遮挡场景下，该自动化流程的检测率（$AP_{50} > 99%$）已达到人工标注基准水平。这些发现表明，基础模型可作为高效且零标注成本的监督者，为智慧农业实现可扩展的边缘计算解决方案。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2605.25860v1)

---

<div align="center">

*由 [YOLO-Arxiv-Daily](https://github.com/WangQvQ/YOLO-Arxiv-Daily) 自动生成*

</div>