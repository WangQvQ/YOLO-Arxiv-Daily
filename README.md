# 每日从arXiv中获取最新YOLO相关论文


## RepVGG\-GELAN: Enhanced GELAN with VGG\-STYLE ConvNets for Brain Tumour Detection

**发布日期**：2024-05-06

**作者**：Thennarasi Balakrishnan

**摘要**：Object detection algorithms particularly those based on YOLO have
demonstrated remarkable efficiency in balancing speed and accuracy. However,
their application in brain tumour detection remains underexplored. This study
proposes RepVGG\-GELAN, a novel YOLO architecture enhanced with RepVGG, a
reparameterized convolutional approach for object detection tasks particularly
focusing on brain tumour detection within medical images. RepVGG\-GELAN
leverages the RepVGG architecture to improve both speed and accuracy in
detecting brain tumours. Integrating RepVGG into the YOLO framework aims to
achieve a balance between computational efficiency and detection performance.
This study includes a spatial pyramid pooling\-based Generalized Efficient Layer
Aggregation Network \(GELAN\) architecture which further enhances the capability
of RepVGG. Experimental evaluation conducted on a brain tumour dataset
demonstrates the effectiveness of RepVGG\-GELAN surpassing existing RCS\-YOLO in
terms of precision and speed. Specifically, RepVGG\-GELAN achieves an increased
precision of 4.91% and an increased AP50 of 2.54% over the latest existing
approach while operating at 240.7 GFLOPs. The proposed RepVGG\-GELAN with GELAN
architecture presents promising results establishing itself as a
state\-of\-the\-art solution for accurate and efficient brain tumour detection in
medical images. The implementation code is publicly available at
https://github.com/ThensiB/RepVGG\-GELAN.


**代码链接**：https://github.com/ThensiB/RepVGG-GELAN.

**论文链接**：[阅读更多](http://arxiv.org/abs/2405.03541v1)

---


## Performance Evaluation of Real\-Time Object Detection for Electric Scooters

**发布日期**：2024-05-05

**作者**：Dong Chen

**摘要**：Electric scooters \(e\-scooters\) have rapidly emerged as a popular mode of
transportation in urban areas, yet they pose significant safety challenges. In
the United States, the rise of e\-scooters has been marked by a concerning
increase in related injuries and fatalities. Recently, while deep\-learning
object detection holds paramount significance in autonomous vehicles to avoid
potential collisions, its application in the context of e\-scooters remains
relatively unexplored. This paper addresses this gap by assessing the
effectiveness and efficiency of cutting\-edge object detectors designed for
e\-scooters. To achieve this, the first comprehensive benchmark involving 22
state\-of\-the\-art YOLO object detectors, including five versions \(YOLOv3,
YOLOv5, YOLOv6, YOLOv7, and YOLOv8\), has been established for real\-time traffic
object detection using a self\-collected dataset featuring e\-scooters. The
detection accuracy, measured in terms of mAP@0.5, ranges from 27.4%
\(YOLOv7\-E6E\) to 86.8% \(YOLOv5s\). All YOLO models, particularly YOLOv3\-tiny,
have displayed promising potential for real\-time object detection in the
context of e\-scooters. Both the traffic scene dataset
\(https://zenodo.org/records/10578641\) and software program codes
\(https://github.com/DongChen06/ScooterDet\) for model benchmarking in this study
are publicly available, which will not only improve e\-scooter safety with
advanced object detection but also lay the groundwork for tailored solutions,
promising a safer and more sustainable urban micromobility landscape.


**代码链接**：https://zenodo.org/records/10578641)，https://github.com/DongChen06/ScooterDet)

**论文链接**：[阅读更多](http://arxiv.org/abs/2405.03039v1)

---


## Fused attention mechanism\-based ore sorting network

**发布日期**：2024-05-05

**作者**：Junjiang Zhen

**摘要**：Deep learning has had a significant impact on the identification and
classification of mineral resources, especially playing a key role in
efficiently and accurately identifying different minerals, which is important
for improving the efficiency and accuracy of mining. However, traditional ore
sorting meth\- ods often suffer from inefficiency and lack of accuracy,
especially in complex mineral environments. To address these challenges, this
study proposes a method called OreYOLO, which incorporates an attentional
mechanism and a multi\-scale feature fusion strategy, based on ore data from
gold and sul\- fide ores. By introducing the progressive feature pyramid
structure into YOLOv5 and embedding the attention mechanism in the feature
extraction module, the detection performance and accuracy of the model are
greatly improved. In order to adapt to the diverse ore sorting scenarios and
the deployment requirements of edge devices, the network structure is designed
to be lightweight, which achieves a low number of parameters \(3.458M\) and
computational complexity \(6.3GFLOPs\) while maintaining high accuracy \(99.3% and
99.2%, respectively\). In the experimental part, a target detection dataset
containing 6000 images of gold and sulfuric iron ore is constructed for gold
and sulfuric iron ore classification training, and several sets of comparison
experiments are set up, including the YOLO series, EfficientDet, Faster\-RCNN,
and CenterNet, etc., and the experiments prove that OreYOLO outperforms the
commonly used high\-performance object detection of these architectures


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2405.02785v1)

---


## Iterative Filter Pruning for Concatenation\-based CNN Architectures

**发布日期**：2024-05-04

**作者**：Svetlana Pavlitska

**摘要**：Model compression and hardware acceleration are essential for the
resource\-efficient deployment of deep neural networks. Modern object detectors
have highly interconnected convolutional layers with concatenations. In this
work, we study how pruning can be applied to such architectures, exemplary for
YOLOv7. We propose a method to handle concatenation layers, based on the
connectivity graph of convolutional layers. By automating iterative sensitivity
analysis, pruning, and subsequent model fine\-tuning, we can significantly
reduce model size both in terms of the number of parameters and FLOPs, while
keeping comparable model accuracy. Finally, we deploy pruned models to FPGA and
NVIDIA Jetson Xavier AGX. Pruned models demonstrate a 2x speedup for the
convolutional layers in comparison to the unpruned counterparts and reach
real\-time capability with 14 FPS on FPGA. Our code is available at
https://github.com/fzi\-forschungszentrum\-informatik/iterative\-yolo\-pruning.


**代码链接**：https://github.com/fzi-forschungszentrum-informatik/iterative-yolo-pruning.

**论文链接**：[阅读更多](http://arxiv.org/abs/2405.03715v1)

---


## Better YOLO with Attention\-Augmented Network and Enhanced Generalization Performance for Safety Helmet Detection

**发布日期**：2024-05-04

**作者**：Shuqi Shen

**摘要**：Safety helmets play a crucial role in protecting workers from head injuries
in construction sites, where potential hazards are prevalent. However,
currently, there is no approach that can simultaneously achieve both model
accuracy and performance in complex environments. In this study, we utilized a
Yolo\-based model for safety helmet detection, achieved a 2% improvement in mAP
\(mean Average Precision\) performance while reducing parameters and Flops count
by over 25%. YOLO\(You Only Look Once\) is a widely used, high\-performance,
lightweight model architecture that is well suited for complex environments. We
presents a novel approach by incorporating a lightweight feature extraction
network backbone based on GhostNetv2, integrating attention modules such as
Spatial Channel\-wise Attention Net\(SCNet\) and Coordination Attention
Net\(CANet\), and adopting the Gradient Norm Aware optimizer \(GAM\) for improved
generalization ability. In safety\-critical environments, the accurate detection
and speed of safety helmets plays a pivotal role in preventing occupational
hazards and ensuring compliance with safety protocols. This work addresses the
pressing need for robust and efficient helmet detection methods, offering a
comprehensive framework that not only enhances accuracy but also improves the
adaptability of detection models to real\-world conditions. Our experimental
results underscore the synergistic effects of GhostNetv2, attention modules,
and the GAM optimizer, presenting a compelling solution for safety helmet
detection that achieves superior performance in terms of accuracy,
generalization, and efficiency.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2405.02591v1)

---


## FER\-YOLO\-Mamba: Facial Expression Detection and Classification Based on Selective State Space

**发布日期**：2024-05-03

**作者**：Hui Ma

**摘要**：Facial Expression Recognition \(FER\) plays a pivotal role in understanding
human emotional cues. However, traditional FER methods based on visual
information have some limitations, such as preprocessing, feature extraction,
and multi\-stage classification procedures. These not only increase
computational complexity but also require a significant amount of computing
resources. Considering Convolutional Neural Network \(CNN\)\-based FER schemes
frequently prove inadequate in identifying the deep, long\-distance dependencies
embedded within facial expression images, and the Transformer's inherent
quadratic computational complexity, this paper presents the FER\-YOLO\-Mamba
model, which integrates the principles of Mamba and YOLO technologies to
facilitate efficient coordination in facial expression image recognition and
localization. Within the FER\-YOLO\-Mamba model, we further devise a FER\-YOLO\-VSS
dual\-branch module, which combines the inherent strengths of convolutional
layers in local feature extraction with the exceptional capability of State
Space Models \(SSMs\) in revealing long\-distance dependencies. To the best of our
knowledge, this is the first Vision Mamba model designed for facial expression
detection and classification. To evaluate the performance of the proposed
FER\-YOLO\-Mamba model, we conducted experiments on two benchmark datasets,
RAF\-DB and SFEW. The experimental results indicate that the FER\-YOLO\-Mamba
model achieved better results compared to other models. The code is available
from https://github.com/SwjtuMa/FER\-YOLO\-Mamba.


**代码链接**：https://github.com/SwjtuMa/FER-YOLO-Mamba.

**论文链接**：[阅读更多](http://arxiv.org/abs/2405.01828v1)

---


## Active Learning Enabled Low\-cost Cell Image Segmentation Using Bounding Box Annotation

**发布日期**：2024-05-02

**作者**：Yu Zhu

**摘要**：Cell image segmentation is usually implemented using fully supervised deep
learning methods, which heavily rely on extensive annotated training data. Yet,
due to the complexity of cell morphology and the requirement for specialized
knowledge, pixel\-level annotation of cell images has become a highly
labor\-intensive task. To address the above problems, we propose an active
learning framework for cell segmentation using bounding box annotations, which
greatly reduces the data annotation cost of cell segmentation algorithms.
First, we generate a box\-supervised learning method \(denoted as YOLO\-SAM\) by
combining the YOLOv8 detector with the Segment Anything Model \(SAM\), which
effectively reduces the complexity of data annotation. Furthermore, it is
integrated into an active learning framework that employs the MC DropBlock
method to train the segmentation model with fewer box\-annotated samples.
Extensive experiments demonstrate that our model saves more than ninety percent
of data annotation time compared to mask\-supervised deep learning methods.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2405.01701v1)

---


## SOAR: Advancements in Small Body Object Detection for Aerial Imagery Using State Space Models and Programmable Gradients

**发布日期**：2024-05-02

**作者**：Tushar Verma

**摘要**：Small object detection in aerial imagery presents significant challenges in
computer vision due to the minimal data inherent in small\-sized objects and
their propensity to be obscured by larger objects and background noise.
Traditional methods using transformer\-based models often face limitations
stemming from the lack of specialized databases, which adversely affect their
performance with objects of varying orientations and scales. This underscores
the need for more adaptable, lightweight models. In response, this paper
introduces two innovative approaches that significantly enhance detection and
segmentation capabilities for small aerial objects. Firstly, we explore the use
of the SAHI framework on the newly introduced lightweight YOLO v9 architecture,
which utilizes Programmable Gradient Information \(PGI\) to reduce the
substantial information loss typically encountered in sequential feature
extraction processes. The paper employs the Vision Mamba model, which
incorporates position embeddings to facilitate precise location\-aware visual
understanding, combined with a novel bidirectional State Space Model \(SSM\) for
effective visual context modeling. This State Space Model adeptly harnesses the
linear complexity of CNNs and the global receptive field of Transformers,
making it particularly effective in remote sensing image classification. Our
experimental results demonstrate substantial improvements in detection accuracy
and processing efficiency, validating the applicability of these approaches for
real\-time small object detection across diverse aerial scenarios. This paper
also discusses how these methodologies could serve as foundational models for
future advancements in aerial object recognition technologies. The source code
will be made accessible here.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2405.01699v2)

---


## Gallbladder Cancer Detection in Ultrasound Images based on YOLO and Faster R\-CNN

**发布日期**：2024-04-23

**作者**：Sara Dadjouy

**摘要**：Medical image analysis is a significant application of artificial
intelligence for disease diagnosis. A crucial step in this process is the
identification of regions of interest within the images. This task can be
automated using object detection algorithms. YOLO and Faster R\-CNN are renowned
for such algorithms, each with its own strengths and weaknesses. This study
aims to explore the advantages of both techniques to select more accurate
bounding boxes for gallbladder detection from ultrasound images, thereby
enhancing gallbladder cancer classification. A fusion method that leverages the
benefits of both techniques is presented in this study. The proposed method
demonstrated superior classification performance, with an accuracy of 92.62%,
compared to the individual use of Faster R\-CNN and YOLOv8, which yielded
accuracies of 90.16% and 82.79%, respectively.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2404.15129v1)

---


## A Nasal Cytology Dataset for Object Detection and Deep Learning

**发布日期**：2024-04-21

**作者**：Mauro Camporeale

**摘要**：Nasal Cytology is a new and efficient clinical technique to diagnose rhinitis
and allergies that is not much widespread due to the time\-consuming nature of
cell counting; that is why AI\-aided counting could be a turning point for the
diffusion of this technique. In this article we present the first dataset of
rhino\-cytological field images: the NCD \(Nasal Cytology Dataset\), aimed to
train and deploy Object Detection models to support physicians and biologists
during clinical practice. The real distribution of the cytotypes, populating
the nasal mucosa has been replicated, sampling images from slides of clinical
patients, and manually annotating each cell found on them. The correspondent
object detection task presents non'trivial issues associated with the strong
class imbalancement, involving the rarest cell types. This work contributes to
some of open challenges by presenting a novel machine learning\-based approach
to aid the automated detection and classification of nasal mucosa cells: the
DETR and YOLO models shown good performance in detecting cells and classifying
them correctly, revealing great potential to accelerate the work of rhinology
experts.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2404.13745v1)

---

