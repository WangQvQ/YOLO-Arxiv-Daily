# 每日从arXiv中获取最新YOLO相关论文


## AnnoPage Dataset: Dataset of Non\-Textual Elements in Documents with Fine\-Grained Categorization / AnnoPage数据集：具有细粒度分类的文档中非文本元素的数据集

发布日期：2025-03-28

作者：Martin Kišš

摘要：We introduce the AnnoPage Dataset, a novel collection of 7550 pages from historical documents, primarily in Czech and German, spanning from 1485 to the present, focusing on the late 19th and early 20th centuries. The dataset is designed to support research in document layout analysis and object detection. Each page is annotated with axis\-aligned bounding boxes \(AABB\) representing elements of 25 categories of non\-textual elements, such as images, maps, decorative elements, or charts, following the Czech Methodology of image document processing. The annotations were created by expert librarians to ensure accuracy and consistency. The dataset also incorporates pages from multiple, mainly historical, document datasets to enhance variability and maintain continuity. The dataset is divided into development and test subsets, with the test set carefully selected to maintain the category distribution. We provide baseline results using YOLO and DETR object detectors, offering a reference point for future research. The AnnoPage Dataset is publicly available on Zenodo \(https://doi.org/10.5281/zenodo.12788419\), along with ground\-truth annotations in YOLO format.

中文摘要：我们介绍了AnnoPage数据集，这是一个7550页的历史文献集，主要来自捷克语和德语，从1485年到现在，主要集中在19世纪末和20世纪初。该数据集旨在支持文档布局分析和对象检测的研究。根据捷克图像文档处理方法，每页都用轴对齐的边界框（AABB）进行注释，表示25类非文本元素的元素，如图像、地图、装饰元素或图表。注释由专业图书馆员创建，以确保准确性和一致性。该数据集还整合了来自多个（主要是历史）文档数据集的页面，以增强可变性并保持连续性。数据集分为开发和测试子集，测试集经过精心选择以保持类别分布。我们使用YOLO和DETR物体探测器提供了基线结果，为未来的研究提供了参考点。AnnoPage数据集在Zenodo上公开可用(https://doi.org/10.5281/zenodo.12788419)，以及YOLO格式的地面实况注释。


代码链接：https://doi.org/10.5281/zenodo.12788419),

论文链接：[阅读更多](http://arxiv.org/abs/2503.22526v1)

---


## BiblioPage: A Dataset of Scanned Title Pages for Bibliographic Metadata Extraction / BiblioPage:用于书目元数据提取的扫描标题页数据集

发布日期：2025-03-25

作者：Jan Kohút

摘要：Manual digitization of bibliographic metadata is time consuming and labor intensive, especially for historical and real\-world archives with highly variable formatting across documents. Despite advances in machine learning, the absence of dedicated datasets for metadata extraction hinders automation. To address this gap, we introduce BiblioPage, a dataset of scanned title pages annotated with structured bibliographic metadata. The dataset consists of approximately 2,000 monograph title pages collected from 14 Czech libraries, spanning a wide range of publication periods, typographic styles, and layout structures. Each title page is annotated with 16 bibliographic attributes, including title, contributors, and publication metadata, along with precise positional information in the form of bounding boxes. To extract structured information from this dataset, we valuated object detection models such as YOLO and DETR combined with transformer\-based OCR, achieving a maximum mAP of 52 and an F1 score of 59. Additionally, we assess the performance of various visual large language models, including LlamA 3.2\-Vision and GPT\-4o, with the best model reaching an F1 score of 67. BiblioPage serves as a real\-world benchmark for bibliographic metadata extraction, contributing to document understanding, document question answering, and document information extraction. Dataset and evaluation scripts are availible at: https://github.com/DCGM/biblio\-dataset

中文摘要：书目元数据的手动数字化既费时又费力，特别是对于文档格式高度可变的历史和现实世界档案。尽管机器学习取得了进步，但缺乏用于元数据提取的专用数据集阻碍了自动化。为了解决这一差距，我们引入了BiblioPage，这是一个用结构化书目元数据注释的扫描标题页数据集。该数据集由从14个捷克图书馆收集的约2000个专著标题页组成，涵盖了广泛的出版时期、排版风格和布局结构。每个标题页都有16个书目属性注释，包括标题、贡献者和出版物元数据，以及以边界框形式的精确位置信息。为了从该数据集中提取结构化信息，我们评估了YOLO和DETR等对象检测模型与基于变换器的OCR的结合，获得了52的最大mAP和59的F1分数。此外，我们评估了各种视觉大型语言模型的性能，包括LlamA 3.2-Vision和GPT-4o，其中最佳模型的F1得分为67。BiblioPage是书目元数据提取的真实基准，有助于文档理解、文档问答和文档信息提取。数据集和评估脚本可在以下网址获得：https://github.com/DCGM/biblio-dataset


代码链接：https://github.com/DCGM/biblio-dataset

论文链接：[阅读更多](http://arxiv.org/abs/2503.19658v1)

---


## You Only Look Once at Anytime \(AnytimeYOLO\): Analysis and Optimization of Early\-Exits for Object\-Detection / 你随时只看一次（AnytimeYOLO）：目标检测早期退出的分析和优化

发布日期：2025-03-21

作者：Daniel Kuhse

摘要：We introduce AnytimeYOLO, a family of variants of the YOLO architecture that enables anytime object detection. Our AnytimeYOLO networks allow for interruptible inference, i.e., they provide a prediction at any point in time, a property desirable for safety\-critical real\-time applications.   We present structured explorations to modify the YOLO architecture, enabling early termination to obtain intermediate results. We focus on providing fine\-grained control through high granularity of available termination points. First, we formalize Anytime Models as a special class of prediction models that offer anytime predictions. Then, we discuss a novel transposed variant of the YOLO architecture, that changes the architecture to enable better early predictions and greater freedom for the order of processing stages. Finally, we propose two optimization algorithms that, given an anytime model, can be used to determine the optimal exit execution order and the optimal subset of early\-exits to select for deployment in low\-resource environments. We evaluate the anytime performance and trade\-offs of design choices, proposing a new anytime quality metric for this purpose. In particular, we also discuss key challenges for anytime inference that currently make its deployment costly.

中文摘要：我们介绍AnytimeYOLO，这是YOLO架构的一系列变体，可以实现任何时间的对象检测。我们的AnytimeYOLO网络允许可中断推理，即它们在任何时间点提供预测，这是安全关键实时应用所需的特性。我们提出了修改YOLO架构的结构化探索，使早期终止能够获得中间结果。我们专注于通过高粒度的可用终止点提供细粒度的控制。首先，我们将Anytime模型形式化为一类特殊的预测模型，提供随时预测。然后，我们讨论了YOLO架构的一种新的转置变体，该变体改变了架构，以实现更好的早期预测和处理阶段顺序的更大自由度。最后，我们提出了两种优化算法，在给定随时模型的情况下，可用于确定最佳退出执行顺序和早期退出的最佳子集，以选择在低资源环境中部署。我们评估设计选择的随时性能和权衡，为此提出了一种新的随时质量指标。特别是，我们还讨论了目前使其部署成本高昂的随时推理的关键挑战。


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2503.17497v1)

---


## UltraFlwr \-\- An Efficient Federated Medical and Surgical Object Detection Framework / UltraFlwr——一种高效的联合医疗和手术目标检测框架

发布日期：2025-03-19

作者：Yang Li

摘要：Object detection shows promise for medical and surgical applications such as cell counting and tool tracking. However, its faces multiple real\-world edge deployment challenges including limited high\-quality annotated data, data sharing restrictions, and computational constraints. In this work, we introduce UltraFlwr, a framework for federated medical and surgical object detection. By leveraging Federated Learning \(FL\), UltraFlwr enables decentralized model training across multiple sites without sharing raw data. To further enhance UltraFlwr's efficiency, we propose YOLO\-PA, a set of novel Partial Aggregation \(PA\) strategies specifically designed for YOLO models in FL. YOLO\-PA significantly reduces communication overhead by up to 83% per round while maintaining performance comparable to Full Aggregation \(FA\) strategies. Our extensive experiments on BCCD and m2cai16\-tool\-locations datasets demonstrate that YOLO\-PA not only provides better client models compared to client\-wise centralized training and FA strategies, but also facilitates efficient training and deployment across resource\-constrained edge devices. Further, we also establish one of the first benchmarks in federated medical and surgical object detection. This paper advances the feasibility of training and deploying detection models on the edge, making federated object detection more practical for time\-critical and resource\-constrained medical and surgical applications. UltraFlwr is publicly available at https://github.com/KCL\-BMEIS/UltraFlwr.

中文摘要：物体检测在细胞计数和工具跟踪等医疗和外科应用中显示出前景。然而，它面临着多个现实世界的边缘部署挑战，包括有限的高质量注释数据、数据共享限制和计算约束。在这项工作中，我们介绍了UltraFlwr，这是一个用于联合医疗和手术对象检测的框架。通过利用联合学习（FL），UltraFlwr实现了跨多个站点的分散模型训练，而无需共享原始数据。为了进一步提高UltraFlwr的效率，我们提出了YOLO-PA，这是一组专门为FL中的YOLO模型设计的新型部分聚合（PA）策略。YOLO-PA每轮可显著降低高达83%的通信开销，同时保持与完全聚合（FA）策略相当的性能。我们在BCCD和m2cai16工具位置数据集上的广泛实验表明，与客户端集中训练和FA策略相比，YOLO-PA不仅提供了更好的客户端模型，而且促进了跨资源受限边缘设备的高效训练和部署。此外，我们还建立了联邦医疗和手术对象检测的首批基准之一。本文提出了在边缘训练和部署检测模型的可行性，使联邦对象检测在时间关键和资源受限的医疗和外科应用中更加实用。UltraFlwr可在以下网址公开获取https://github.com/KCL-BMEIS/UltraFlwr.


代码链接：https://github.com/KCL-BMEIS/UltraFlwr.

论文链接：[阅读更多](http://arxiv.org/abs/2503.15161v1)

---


## YOLO\-LLTS: Real\-Time Low\-Light Traffic Sign Detection via Prior\-Guided Enhancement and Multi\-Branch Feature Interaction / YOLO-LLTS：通过先验引导增强和多分支特征交互进行实时低光交通标志检测

发布日期：2025-03-18

作者：Ziyu Lin

摘要：Detecting traffic signs effectively under low\-light conditions remains a significant challenge. To address this issue, we propose YOLO\-LLTS, an end\-to\-end real\-time traffic sign detection algorithm specifically designed for low\-light environments. Firstly, we introduce the High\-Resolution Feature Map for Small Object Detection \(HRFM\-TOD\) module to address indistinct small\-object features in low\-light scenarios. By leveraging high\-resolution feature maps, HRFM\-TOD effectively mitigates the feature dilution problem encountered in conventional PANet frameworks, thereby enhancing both detection accuracy and inference speed. Secondly, we develop the Multi\-branch Feature Interaction Attention \(MFIA\) module, which facilitates deep feature interaction across multiple receptive fields in both channel and spatial dimensions, significantly improving the model's information extraction capabilities. Finally, we propose the Prior\-Guided Enhancement Module \(PGFE\) to tackle common image quality challenges in low\-light environments, such as noise, low contrast, and blurriness. This module employs prior knowledge to enrich image details and enhance visibility, substantially boosting detection performance. To support this research, we construct a novel dataset, the Chinese Nighttime Traffic Sign Sample Set \(CNTSSS\), covering diverse nighttime scenarios, including urban, highway, and rural environments under varying weather conditions. Experimental evaluations demonstrate that YOLO\-LLTS achieves state\-of\-the\-art performance, outperforming the previous best methods by 2.7% mAP50 and 1.6% mAP50:95 on TT100K\-night, 1.3% mAP50 and 1.9% mAP50:95 on CNTSSS, and achieving superior results on the CCTSDB2021 dataset. Moreover, deployment experiments on edge devices confirm the real\-time applicability and effectiveness of our proposed approach.

中文摘要：在低光照条件下有效检测交通标志仍然是一个重大挑战。为了解决这个问题，我们提出了YOLO-LLTS，这是一种专为低光环境设计的端到端实时交通标志检测算法。首先，我们引入了小目标检测的高分辨率特征图（HRFM-TOD）模块，以解决低光场景中模糊的小目标特征。通过利用高分辨率特征图，HRFM-TOD有效地缓解了传统PANet框架中遇到的特征稀释问题，从而提高了检测精度和推理速度。其次，我们开发了多分支特征交互注意力（MFIA）模块，该模块促进了通道和空间维度上多个感受野之间的深度特征交互，显著提高了模型的信息提取能力。最后，我们提出了先验引导增强模块（PGFE）来解决低光环境中常见的图像质量挑战，如噪声、低对比度和模糊。该模块利用先验知识来丰富图像细节并提高可见度，从而大大提高了检测性能。为了支持这项研究，我们构建了一个新的数据集，即中国夜间交通标志样本集（CNTSSS），涵盖了不同天气条件下的不同夜间场景，包括城市、高速公路和农村环境。实验评估表明，YOLO-LLTS达到了最先进的性能，在TT100K夜晚的表现优于之前的最佳方法2.7%mAP50和1.6%mAP50:95，在CNTSSS上的表现优于1.3%mAP50，在CCTSDB2021数据集上的表现更为出色。此外，在边缘设备上的部署实验证实了我们提出的方法的实时适用性和有效性。


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2503.13883v1)

---

