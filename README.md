<div align="center">

# YOLO ArXiv Daily

[![Daily Papers](https://img.shields.io/badge/📅-每日更新-blue)]()
[![arXiv](https://img.shields.io/badge/arXiv-最新论文-red)](https://arxiv.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)

*自动追踪 YOLO 相关最新论文，提供中英文双语摘要*

</div>

---

## 📑 论文列表

> ### 1. PlayClass: Automated Play Behaviour Classification in Poultry
> **🔹 中文标题：** PlayClass：家禽游戏行为的自动分类方法
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
> 动物福利的自动化监测主要针对负面指标，而诸如玩耍等积极福利行为尚未深入探索。为填补这一空白，我们提出PlayClass——一个基于俯视围栏视频对家禽玩耍行为进行分类的流程。该流程借助YOLO引导的分块边界实现SAM 3的长期跟踪，以最大程度减少基于点提示时的身份识别错误，并利用图像与视频基础模型的冻结嵌入进行玩耍动作分类。尽管仅从跟踪掩膜中提取手工运动特征已能达到有竞争力的准确率，但V-JEPA 2.1在不同模型规模上均优于其他主干网络，结合手工特征时宏平均F1值达到77.0。尽管取得该结果，由于玩耍子类型与非玩耍行为具有相似的运动特征，且存在个体间的遮挡，该数据集仍具有挑战性。总体而言，本研究为建立家禽玩耍行为的自动化分类框架提供了积极的技术启示。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2605.27304v1)

---

> ### 2. Small Object Detection in Industrial Recycling: A New Dataset and YOLO Performance Evaluation
> **🔹 中文标题：** 工业回收中的小目标检测：新数据集与YOLO性能评估
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
> 本文针对计算机视觉领域的核心挑战——小而密集重叠目标的检测问题，重点评述了基于深度学习监督方法的研究进展。通过构建包含逾万张图像与12万实例的新型数据集，我们对现有检测系统在工业回收场景中的性能表现、识别精度与计算效率进行了系统性对比分析，从而甄选出当前最可靠的技术方案及其特定应用场景。研究同时探讨了数据增强与合成图像技术的赋能效应，并据此提出提升小密集重叠目标检测系统效能的未来研究方向与创新解决方案。本研究范畴覆盖回收流程中的目标检测、尺寸测量与异常检测三大任务，其中异常检测策略具备图像分辨率与缩放级别自适应能力，可确保工业环境下的稳定运行。所提出的数据集、方法与评估代码开源地址为：https://github.com/o-messai/SDOOD
>
> **💻 代码链接：** https://github.com/o-messai/SDOOD
>
> 🔗 [阅读论文](http://arxiv.org/abs/2605.26884v1)

---

> ### 3. Pixel\-Level Pavement Distress Assessment Using Instance Segmentation
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
> 自动化路面病害评估不仅需要图像级分类或粗略的边界框检测，更需要精确定位细微的、分枝状的、不规则裂缝，以实现维护相关量化所需的几何精度。本文提出了一种基于Mask R-CNN实例分割的视觉化路面病害分析系统，并在UWGB-StreetCrack数据集上对其进行了评估。该数据集是通过车载智能手机采集的道路图像，由人工以多边形标注了纵向裂缝、横向裂缝、龟裂和坑洞。研究在统一的微调协议下，测试了五种基于Detectron2的Mask R-CNN骨干网络变体。其中，采用ResNet-101 FPN骨干网络的Mask R-CNN模型表现最佳，在项目特定的边界框匹配协议下，达到了84.23%的精确率、90.04%的召回率和87.04%的F1分数。该模型预测的裂缝面积占比为2.164%，与地面真值的2.170%裂缝面积占比高度吻合。为将实例分割系统与检测器导向的替代方案进行对比，本研究还采用了基于CSPDarknet53的YOLO检测器并在该数据集上重新训练，其在验证协议上的精确率为27.5%，召回率为20.7%。结果表明，实例分割是处理实地路面图像和估算总体裂缝面积的实用方向，同时也揭示了在标注一致性、类别不平衡、干扰项排除和掩码级基准测试等方面仍存在的开放挑战。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2605.26095v1)

---

> ### 4. SAM3\-Assisted Training of Lightweight YOLO Models for Precision Pig Farming
> **🔹 中文标题：** SAM3辅助训练轻量化YOLO模型用于精准养猪
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
> 基于深度学习的目标检测技术已彻底变革了精准畜牧业，但仍面临关键瓶颈：高性能基础模型（如SAM3）因计算量过大难以部署于边缘设备，而轻量化模型（如YOLO）又需消耗大量人工标注成本。本研究提出一种全自动知识蒸馏流程，利用分割万物模型3（SAM3）生成零样本伪标签，以训练高效的YOLOv8检测器。通过将SAM3作为离线自动标注器，我们消除了人工标注的瓶颈，使模型能够在资源受限的硬件上实现实时推理。我们在PigLife数据集上系统评估该方法，对比SAM3监督模型与人工标注基线的表现。实验表明，无需人工干预的SAM3训练YOLOv8m模型平均精度均值达79.4%，推理延迟相比教师模型降低约200倍。分层分析进一步显示，在低遮挡场景中，自动化流程的检测性能可达到人类基准水平（AP₅₀>99%）。这些发现证实基础模型能作为高效且零标注成本的监督器，为智慧农业提供可扩展的边缘计算解决方案。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2605.25860v1)

---

> ### 5. TinyFormer: Preserving Tiny Objects in YOLO\-DETRHybridReal\-time Detectors
> **🔹 中文标题：** 微型Transformer：在YOLO-DETR混合实时检测器中保留微小目标
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-05-24 |
> | 👤 作者 | Jun\-Wei Hsieh |
>
> **📄 英文摘要：**
> YOLO\-series and DETR\-based detectors struggle with tiny\-object detection. YOLO\-style models benefit from efficient dense prediction, but their large\-stride backbones may suppress tiny instances in deep feature maps and make grid assignment ambiguous. DETR\-based models remove hand\-crafted post\-processing through set prediction, yet they reason over coarse token grids, where tiny objects occupy only a few weak tokens and are easily overlooked during matching. To address these limitations, we propose TinyFormer, a unified YOLO\-\-DETR hybrid real\-time detector that combines ViT representations, NMS\-free set prediction, and a YOLO\-style pyramid neck for accurate small\-object detection. TinyFormer introduces a Parallel Bi\-fusion Module \(PBM\), which builds high\-resolution shortcuts from shallow stages to the feature pyramid, preserving fine spatial details during multi\-scale fusion. We further design a Spatial Semantic Adapter \(SSA\) to compensate for the spatial loss caused by coarse tokenization. SSA extracts high\-resolution cues from early stages and injects them into transformer token embeddings, improving tiny\-object localization without sacrificing the global modeling ability of DETR. Experiments on MS COCO show that TinyFormer consistently outperforms recent YOLO\-series detectors and the strong DEIMv2 baseline. TinyFormer\-X achieves 58.4% AP even without PBM, while adding PBM improves the overall AP to 58.5% and brings a 1.6% AP gain on small objects. With Objects365 pre\-training, TinyFormer\-X\-PBM reaches 60.2% AP, surpassing RF\-DETR and other Objects365\-pretrained detectors with fewer parameters and lower computation. These results demonstrate that TinyFormer bridges dense YOLO\-style feature fusion and DETR\-style set prediction, providing a strong accuracy\-efficiency trade\-off for real\-time tiny\-object detection. Code is available at https://github.com/mmpmmpmmpjosh/TinyFormer.
>
> **📝 中文摘要：**
> YOLO系列与基于DETR的检测器在小目标检测上均存在瓶颈。YOLO类模型虽受益于高效密集预测机制，但其大步幅主干网络可能在深层特征图中抑制微小实例，且网格分配存在模糊性；DETR类模型虽通过集合预测消除了人工后处理，却需在粗糙标记网格上进行推理，其中微小目标仅占据少数弱特征信号，在匹配过程中极易被忽视。为解决上述局限，我们提出TinyFormer——一种融合ViT表征、无NMS集合预测与YOLO式金字塔颈部结构的统一混合实时检测器，以实现精准的小目标检测。TinyFormer引入并行双融合模块，通过构建从浅层阶段到特征金字塔的高分辨率捷径，在多尺度融合过程中保留精细空间细节。我们进一步设计空间语义适配器，以补偿粗糙标记化导致的空间信息损失：该适配器从早期阶段提取高分辨率线索并注入Transformer标记嵌入，在保持DETR全局建模能力的同时提升小目标定位精度。MS COCO实验表明，TinyFormer持续优于近期YOLO系列检测器与强基线DEIMv2。即使未使用并行双融合模块，TinyFormer-X仍达58.4%平均精度；加入该模块后整体平均精度提升至58.5%，并在小目标上获得1.6%的精度增益。经Objects365预训练的TinyFormer-X-PBM以更少参数量与计算成本达到60.2%平均精度，超越RF-DETR及其他Objects365预训练检测器。实验结果证实，TinyFormer成功桥接了密集型YOLO风格特征融合与集合预测式DETR范式，为实时小目标检测提供了精度与效率的卓越平衡。代码已开源：https://github.com/mmpmmpmmpjosh/TinyFormer。
>
> **💻 代码链接：** https://github.com/mmpmmpmmpjosh/TinyFormer.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2605.25046v1)

---

<div align="center">

*由 [YOLO-Arxiv-Daily](https://github.com/WangQvQ/YOLO-Arxiv-Daily) 自动生成*

</div>