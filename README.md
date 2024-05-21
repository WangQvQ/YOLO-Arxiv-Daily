# 每日从arXiv中获取最新YOLO相关论文


## Bangladeshi Native Vehicle Detection in Wild

**发布日期**：2024-05-20

**作者**：Bipin Saha

**摘要**：The success of autonomous navigation relies on robust and precise vehicle
recognition, hindered by the scarcity of region\-specific vehicle detection
datasets, impeding the development of context\-aware systems. To advance
terrestrial object detection research, this paper proposes a native vehicle
detection dataset for the most commonly appeared vehicle classes in Bangladesh.
17 distinct vehicle classes have been taken into account, with fully annotated
81542 instances of 17326 images. Each image width is set to at least 1280px.
The dataset's average vehicle bounding box\-to\-image ratio is 4.7036. This
Bangladesh Native Vehicle Dataset \(BNVD\) has accounted for several
geographical, illumination, variety of vehicle sizes, and orientations to be
more robust on surprised scenarios. In the context of examining the BNVD
dataset, this work provides a thorough assessment with four successive You Only
Look Once \(YOLO\) models, namely YOLO v5, v6, v7, and v8. These dataset's
effectiveness is methodically evaluated and contrasted with other vehicle
datasets already in use. The BNVD dataset exhibits mean average precision\(mAP\)
at 50% intersection over union \(IoU\) is 0.848 corresponding precision and
recall values of 0.841 and 0.774. The research findings indicate a mAP of 0.643
at an IoU range of 0.5 to 0.95. The experiments show that the BNVD dataset
serves as a reliable representation of vehicle distribution and presents
considerable complexities.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2405.12150v1)

---


## Drone\-type\-Set: Drone types detection benchmark for drone detection and tracking

**发布日期**：2024-05-16

**作者**：Kholoud AlDosari

**摘要**：The Unmanned Aerial Vehicles \(UAVs\) market has been significantly growing and
Considering the availability of drones at low\-cost prices the possibility of
misusing them, for illegal purposes such as drug trafficking, spying, and
terrorist attacks posing high risks to national security, is rising. Therefore,
detecting and tracking unauthorized drones to prevent future attacks that
threaten lives, facilities, and security, become a necessity. Drone detection
can be performed using different sensors, while image\-based detection is one of
them due to the development of artificial intelligence techniques. However,
knowing unauthorized drone types is one of the challenges due to the lack of
drone types datasets. For that, in this paper, we provide a dataset of various
drones as well as a comparison of recognized object detection models on the
proposed dataset including YOLO algorithms with their different versions, like,
v3, v4, and v5 along with the Detectronv2. The experimental results of
different models are provided along with a description of each method. The
collected dataset can be found in
https://drive.google.com/drive/folders/1EPOpqlF4vG7hp4MYnfAecVOsdQ2JwBEd?usp=share\_link


**代码链接**：https://drive.google.com/drive/folders/1EPOpqlF4vG7hp4MYnfAecVOsdQ2JwBEd?usp=share_link

**论文链接**：[阅读更多](http://arxiv.org/abs/2405.10398v1)

---


## WeedScout: Real\-Time Autonomous blackgrass Classification and Mapping using dedicated hardware

**发布日期**：2024-05-12

**作者**：Matthew Gazzard

**摘要**：Blackgrass \(Alopecurus myosuroides\) is a competitive weed that has
wide\-ranging impacts on food security by reducing crop yields and increasing
cultivation costs. In addition to the financial burden on agriculture, the
application of herbicides as a preventive to blackgrass can negatively affect
access to clean water and sanitation. The WeedScout project introduces a
Real\-Rime Autonomous Black\-Grass Classification and Mapping \(RT\-ABGCM\), a
cutting\-edge solution tailored for real\-time detection of blackgrass, for
precision weed management practices. Leveraging Artificial Intelligence \(AI\)
algorithms, the system processes live image feeds, infers blackgrass density,
and covers two stages of maturation. The research investigates the deployment
of You Only Look Once \(YOLO\) models, specifically the streamlined YOLOv8 and
YOLO\-NAS, accelerated at the edge with the NVIDIA Jetson Nano \(NJN\). By
optimising inference speed and model performance, the project advances the
integration of AI into agricultural practices, offering potential solutions to
challenges such as herbicide resistance and environmental impact. Additionally,
two datasets and model weights are made available to the research community,
facilitating further advancements in weed detection and precision farming
technologies.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2405.07349v1)

---


## Differentiable Model Scaling using Differentiable Topk

**发布日期**：2024-05-12

**作者**：Kai Liu

**摘要**：Over the past few years, as large language models have ushered in an era of
intelligence emergence, there has been an intensified focus on scaling
networks. Currently, many network architectures are designed manually, often
resulting in sub\-optimal configurations. Although Neural Architecture Search
\(NAS\) methods have been proposed to automate this process, they suffer from low
search efficiency. This study introduces Differentiable Model Scaling \(DMS\),
increasing the efficiency for searching optimal width and depth in networks.
DMS can model both width and depth in a direct and fully differentiable way,
making it easy to optimize. We have evaluated our DMS across diverse tasks,
ranging from vision tasks to NLP tasks and various network architectures,
including CNNs and Transformers. Results consistently indicate that our DMS can
find improved structures and outperforms state\-of\-the\-art NAS methods.
Specifically, for image classification on ImageNet, our DMS improves the top\-1
accuracy of EfficientNet\-B0 and Deit\-Tiny by 1.4% and 0.6%, respectively, and
outperforms the state\-of\-the\-art zero\-shot NAS method, ZiCo, by 1.3% while
requiring only 0.4 GPU days for searching. For object detection on COCO, DMS
improves the mAP of Yolo\-v8\-n by 2.0%. For language modeling, our pruned
Llama\-7B outperforms the prior method with lower perplexity and higher
zero\-shot classification accuracy. We will release our code in the future.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2405.07194v1)

---


## Common Corruptions for Enhancing and Evaluating Robustness in Air\-to\-Air Visual Object Detection

**发布日期**：2024-05-10

**作者**：Anastasios Arsenos

**摘要**：The main barrier to achieving fully autonomous flights lies in autonomous
aircraft navigation. Managing non\-cooperative traffic presents the most
important challenge in this problem. The most efficient strategy for handling
non\-cooperative traffic is based on monocular video processing through deep
learning models. This study contributes to the vision\-based deep learning
aircraft detection and tracking literature by investigating the impact of data
corruption arising from environmental and hardware conditions on the
effectiveness of these methods. More specifically, we designed $7$ types of
common corruptions for camera inputs taking into account real\-world flight
conditions. By applying these corruptions to the Airborne Object Tracking \(AOT\)
dataset we constructed the first robustness benchmark dataset named AOT\-C for
air\-to\-air aerial object detection. The corruptions included in this dataset
cover a wide range of challenging conditions such as adverse weather and sensor
noise. The second main contribution of this letter is to present an extensive
experimental evaluation involving $8$ diverse object detectors to explore the
degradation in the performance under escalating levels of corruptions \(domain
shifts\). Based on the evaluation results, the key observations that emerge are
the following: 1\) One\-stage detectors of the YOLO family demonstrate better
robustness, 2\) Transformer\-based and multi\-stage detectors like Faster R\-CNN
are extremely vulnerable to corruptions, 3\) Robustness against corruptions is
related to the generalization ability of models. The third main contribution is
to present that finetuning on our augmented synthetic data results in
improvements in the generalisation ability of the object detector in real\-world
flight experiments.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2405.06765v2)

---


## Real\-Time Pill Identification for the Visually Impaired Using Deep Learning

**发布日期**：2024-05-08

**作者**：Bo Dang

**摘要**：The prevalence of mobile technology offers unique opportunities for
addressing healthcare challenges, especially for individuals with visual
impairments. This paper explores the development and implementation of a deep
learning\-based mobile application designed to assist blind and visually
impaired individuals in real\-time pill identification. Utilizing the YOLO
framework, the application aims to accurately recognize and differentiate
between various pill types through real\-time image processing on mobile
devices. The system incorporates Text\-to\- Speech \(TTS\) to provide immediate
auditory feedback, enhancing usability and independence for visually impaired
users. Our study evaluates the application's effectiveness in terms of
detection accuracy and user experience, highlighting its potential to improve
medication management and safety among the visually impaired community.
Keywords\-Deep Learning; YOLO Framework; Mobile Application; Visual Impairment;
Pill Identification; Healthcare


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2405.05983v1)

---


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

