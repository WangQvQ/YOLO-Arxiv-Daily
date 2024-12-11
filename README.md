# 每日从arXiv中获取最新YOLO相关论文


## 3A\-YOLO: New Real\-Time Object Detectors with Triple Discriminative Awareness and Coordinated Representations

**发布日期**：2024-12-10

**作者**：Xuecheng Wu

**摘要**：Recent research on real\-time object detectors \(e.g., YOLO series\) has
demonstrated the effectiveness of attention mechanisms for elevating model
performance. Nevertheless, existing methods neglect to unifiedly deploy
hierarchical attention mechanisms to construct a more discriminative YOLO head
which is enriched with more useful intermediate features. To tackle this gap,
this work aims to leverage multiple attention mechanisms to hierarchically
enhance the triple discriminative awareness of the YOLO detection head and
complementarily learn the coordinated intermediate representations, resulting
in a new series detectors denoted 3A\-YOLO. Specifically, we first propose a new
head denoted TDA\-YOLO Module, which unifiedly enhance the representations
learning of scale\-awareness, spatial\-awareness, and task\-awareness. Secondly,
we steer the intermediate features to coordinately learn the inter\-channel
relationships and precise positional information. Finally, we perform neck
network improvements followed by introducing various tricks to boost the
adaptability of 3A\-YOLO. Extensive experiments across COCO and VOC benchmarks
indicate the effectiveness of our detectors.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2412.07168v1)

---


## DEYOLO: Dual\-Feature\-Enhancement YOLO for Cross\-Modality Object Detection

**发布日期**：2024-12-06

**作者**：Yishuo Chen

**摘要**：Object detection in poor\-illumination environments is a challenging task as
objects are usually not clearly visible in RGB images. As infrared images
provide additional clear edge information that complements RGB images, fusing
RGB and infrared images has potential to enhance the detection ability in
poor\-illumination environments. However, existing works involving both visible
and infrared images only focus on image fusion, instead of object detection.
Moreover, they directly fuse the two kinds of image modalities, which ignores
the mutual interference between them. To fuse the two modalities to maximize
the advantages of cross\-modality, we design a dual\-enhancement\-based
cross\-modality object detection network DEYOLO, in which semantic\-spatial cross
modality and novel bi\-directional decoupled focus modules are designed to
achieve the detection\-centered mutual enhancement of RGB\-infrared \(RGB\-IR\).
Specifically, a dual semantic enhancing channel weight assignment module \(DECA\)
and a dual spatial enhancing pixel weight assignment module \(DEPA\) are firstly
proposed to aggregate cross\-modality information in the feature space to
improve the feature representation ability, such that feature fusion can aim at
the object detection task. Meanwhile, a dual\-enhancement mechanism, including
enhancements for two\-modality fusion and single modality, is designed in both
DECAand DEPAto reduce interference between the two kinds of image modalities.
Then, a novel bi\-directional decoupled focus is developed to enlarge the
receptive field of the backbone network in different directions, which improves
the representation quality of DEYOLO. Extensive experiments on M3FD and LLVIP
show that our approach outperforms SOTA object detection algorithms by a clear
margin. Our code is available at https://github.com/chips96/DEYOLO.


**代码链接**：https://github.com/chips96/DEYOLO.

**论文链接**：[阅读更多](http://arxiv.org/abs/2412.04931v1)

---


## YOLO\-CCA: A Context\-Based Approach for Traffic Sign Detection

**发布日期**：2024-12-05

**作者**：Linfeng Jiang

**摘要**：Traffic sign detection is crucial for improving road safety and advancing
autonomous driving technologies. Due to the complexity of driving environments,
traffic sign detection frequently encounters a range of challenges, including
low resolution, limited feature information, and small object sizes. These
challenges significantly hinder the effective extraction of features from
traffic signs, resulting in false positives and false negatives in object
detection. To address these challenges, it is essential to explore more
efficient and accurate approaches for traffic sign detection. This paper
proposes a context\-based algorithm for traffic sign detection, which utilizes
YOLOv7 as the baseline model. Firstly, we propose an adaptive local context
feature enhancement \(LCFE\) module using multi\-scale dilation convolution to
capture potential relationships between the object and surrounding areas. This
module supplements the network with additional local context information.
Secondly, we propose a global context feature collection \(GCFC\) module to
extract key location features from the entire image scene as global context
information. Finally, we build a Transformer\-based context collection
augmentation \(CCA\) module to process the collected local context and global
context, which achieves superior multi\-level feature fusion results for YOLOv7
without bringing in additional complexity. Extensive experimental studies
performed on the Tsinghua\-Tencent 100K dataset show that the mAP of our method
is 92.1\\%. Compared with YOLOv7, our approach improves 3.9\\% in mAP, while the
amount of parameters is reduced by 2.7M. On the CCTSDB2021 dataset the mAP is
improved by 0.9\\%. These results show that our approach achieves higher
detection accuracy with fewer parameters. The source code is available at
\\url\{https://github.com/zippiest/yolo\-cca\}.


**代码链接**：https://github.com/zippiest/yolo-cca}.

**论文链接**：[阅读更多](http://arxiv.org/abs/2412.04289v1)

---


## Machine learning enhanced multi\-particle tracking in solid fuel combustion

**发布日期**：2024-12-05

**作者**：Haowen Chen

**摘要**：Particle velocimetry is essential in solid fuel combustion studies, however,
the accurate detection and tracking of particles in high Particle Number
Density \(PND\) combustion scenario remain challenging. The current study
advances the machine\-learning approaches for precise velocity measurements of
solid particles. For this, laser imaging experiments were performed for
high\-volatile bituminous coal particles burning in a laminar flow reactor.
Particle positions were imaged using time\-resolved Mie scattering. Various
detection methods, including conventional blob detection and Machine Learning
\(ML\) based You Only Look Once \(YOLO\) and Realtime Detection Transformer
\(RT\-DETR\) were employed and bench marked.~Particle tracking was performed using
the Simple Online Realtime Tracking \(SORT\) algorithm. The results demonstrated
the capability of machine learning models trained on low\-PND data for
prediction of high\-PND data. Slicing Aided Hyper Inference \(SAHI\) algorithm is
important for the better performance of the used models. By evaluating the
velocity statistics, it is found that the mean particle velocity decreases with
increasing PND, primarily due to stronger particle interactions. The particle
dynamics are closely related to the position of combustion zone observed in the
previous study. Thus, PND is considered as the dominant factor for the particle
group combustion behavior of high\-volatile solid fuels.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2412.04091v1)

---


## HyperDefect\-YOLO: Enhance YOLO with HyperGraph Computation for Industrial Defect Detection

**发布日期**：2024-12-05

**作者**：Zuo Zuo

**摘要**：In the manufacturing industry, defect detection is an essential but
challenging task aiming to detect defects generated in the process of
production. Though traditional YOLO models presents a good performance in
defect detection, they still have limitations in capturing high\-order feature
interrelationships, which hurdles defect detection in the complex scenarios and
across the scales. To this end, we introduce hypergraph computation into YOLO
framework, dubbed HyperDefect\-YOLO \(HD\-YOLO\), to improve representative ability
and semantic exploitation. HD\-YOLO consists of Defect Aware Module \(DAM\) and
Mixed Graph Network \(MGNet\) in the backbone, which specialize for perception
and extraction of defect features. To effectively aggregate multi\-scale
features, we propose HyperGraph Aggregation Network \(HGANet\) which combines
hypergraph and attention mechanism to aggregate multi\-scale features.
Cross\-Scale Fusion \(CSF\) is proposed to adaptively fuse and handle features
instead of simple concatenation and convolution. Finally, we propose Semantic
Aware Module \(SAM\) in the neck to enhance semantic exploitation for accurately
localizing defects with different sizes in the disturbed background. HD\-YOLO
undergoes rigorous evaluation on public HRIPCB and NEU\-DET datasets with
significant improvements compared to state\-of\-the\-art methods. We also evaluate
HD\-YOLO on self\-built MINILED dataset collected in real industrial scenarios to
demonstrate the effectiveness of the proposed method. The source codes are at
https://github.com/Jay\-zzcoder/HD\-YOLO.


**代码链接**：https://github.com/Jay-zzcoder/HD-YOLO.

**论文链接**：[阅读更多](http://arxiv.org/abs/2412.03969v1)

---


## Deep Learning and Hybrid Approaches for Dynamic Scene Analysis, Object Detection and Motion Tracking

**发布日期**：2024-12-05

**作者**：Shahran Rahman Alve

**摘要**：This project aims to develop a robust video surveillance system, which can
segment videos into smaller clips based on the detection of activities. It uses
CCTV footage, for example, to record only major events\-like the appearance of a
person or a thief\-so that storage is optimized and digital searches are easier.
It utilizes the latest techniques in object detection and tracking, including
Convolutional Neural Networks \(CNNs\) like YOLO, SSD, and Faster R\-CNN, as well
as Recurrent Neural Networks \(RNNs\) and Long Short\-Term Memory networks
\(LSTMs\), to achieve high accuracy in detection and capture temporal
dependencies. The approach incorporates adaptive background modeling through
Gaussian Mixture Models \(GMM\) and optical flow methods like Lucas\-Kanade to
detect motions. Multi\-scale and contextual analysis are used to improve
detection across different object sizes and environments. A hybrid motion
segmentation strategy combines statistical and deep learning models to manage
complex movements, while optimizations for real\-time processing ensure
efficient computation. Tracking methods, such as Kalman Filters and Siamese
networks, are employed to maintain smooth tracking even in cases of occlusion.
Detection is improved on various\-sized objects for multiple scenarios by
multi\-scale and contextual analysis. Results demonstrate high precision and
recall in detecting and tracking objects, with significant improvements in
processing times and accuracy due to real\-time optimizations and
illumination\-invariant features. The impact of this research lies in its
potential to transform video surveillance, reducing storage requirements and
enhancing security through reliable and efficient object detection and
tracking.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2412.05331v1)

---


## Traffic Co\-Simulation Framework Empowered by Infrastructure Camera Sensing and Reinforcement Learning

**发布日期**：2024-12-05

**作者**：Talha Azfar

**摘要**：Traffic simulations are commonly used to optimize traffic flow, with
reinforcement learning \(RL\) showing promising potential for automated traffic
signal control. Multi\-agent reinforcement learning \(MARL\) is particularly
effective for learning control strategies for traffic lights in a network using
iterative simulations. However, existing methods often assume perfect vehicle
detection, which overlooks real\-world limitations related to infrastructure
availability and sensor reliability. This study proposes a co\-simulation
framework integrating CARLA and SUMO, which combines high\-fidelity 3D modeling
with large\-scale traffic flow simulation. Cameras mounted on traffic light
poles within the CARLA environment use a YOLO\-based computer vision system to
detect and count vehicles, providing real\-time traffic data as input for
adaptive signal control in SUMO. MARL agents, trained with four different
reward structures, leverage this visual feedback to optimize signal timings and
improve network\-wide traffic flow. Experiments in the test\-bed demonstrate the
effectiveness of the proposed MARL approach in enhancing traffic conditions
using real\-time camera\-based detection. The framework also evaluates the
robustness of MARL under faulty or sparse sensing and compares the performance
of YOLOv5 and YOLOv8 for vehicle detection. Results show that while better
accuracy improves performance, MARL agents can still achieve significant
improvements with imperfect detection, demonstrating adaptability for
real\-world scenarios.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2412.03925v1)

---


## Smart Parking with Pixel\-Wise ROI Selection for Vehicle Detection Using YOLOv8, YOLOv9, YOLOv10, and YOLOv11

**发布日期**：2024-12-02

**作者**：Gustavo P. C. P. da Luz

**摘要**：The increasing urbanization and the growing number of vehicles in cities have
underscored the need for efficient parking management systems. Traditional
smart parking solutions often rely on sensors or cameras for occupancy
detection, each with its limitations. Recent advancements in deep learning have
introduced new YOLO models \(YOLOv8, YOLOv9, YOLOv10, and YOLOv11\), but these
models have not been extensively evaluated in the context of smart parking
systems, particularly when combined with Region of Interest \(ROI\) selection for
object detection. Existing methods still rely on fixed polygonal ROI selections
or simple pixel\-based modifications, which limit flexibility and precision.
This work introduces a novel approach that integrates Internet of Things, Edge
Computing, and Deep Learning concepts, by using the latest YOLO models for
vehicle detection. By exploring both edge and cloud computing, it was found
that inference times on edge devices ranged from 1 to 92 seconds, depending on
the hardware and model version. Additionally, a new pixel\-wise post\-processing
ROI selection method is proposed for accurately identifying regions of interest
to count vehicles in parking lot images. The proposed system achieved 99.68%
balanced accuracy on a custom dataset of 3,484 images, offering a
cost\-effective smart parking solution that ensures precise vehicle detection
while preserving data privacy


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2412.01983v2)

---


## Research on Cervical Cancer p16/Ki\-67 Immunohistochemical Dual\-Staining Image Recognition Algorithm Based on YOLO

**发布日期**：2024-12-02

**作者**：Xiao\-Jun Wu

**摘要**：The p16/Ki\-67 dual staining method is a new approach for cervical cancer
screening with high sensitivity and specificity. However, there are issues of
mis\-detection and inaccurate recognition when the YOLOv5s algorithm is directly
applied to dual\-stained cell images. This paper Proposes a novel cervical
cancer dual\-stained image recognition \(DSIR\-YOLO\) model based on an YOLOv5. By
fusing the Swin\-Transformer module, GAM attention mechanism, multi\-scale
feature fusion, and EIoU loss function, the detection performance is
significantly improved, with mAP@0.5 and mAP@0.5:0.95 reaching 92.6% and 70.5%,
respectively. Compared with YOLOv5s in five\-fold cross\-validation, the
accuracy, recall, mAP@0.5, and mAP@0.5:0.95 of the improved algorithm are
increased by 2.3%, 4.1%, 4.3%, and 8.0%, respectively, with smaller variances
and higher stability. Compared with other detection algorithms, DSIR\-YOLO in
this paper sacrifices some performance requirements to improve the network
recognition effect. In addition, the influence of dataset quality on the
detection results is studied. By controlling the sealing property of pixels,
scale difference, unlabelled cells, and diagonal annotation, the model
detection accuracy, recall, mAP@0.5, and mAP@0.5:0.95 are improved by 13.3%,
15.3%, 18.3%, and 30.5%, respectively.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2412.01372v1)

---


## Thermal Vision: Pioneering Non\-Invasive Temperature Tracking in Congested Spaces

**发布日期**：2024-12-01

**作者**：Arijit Samal

**摘要**：Non\-invasive temperature monitoring of individuals plays a crucial role in
identifying and isolating symptomatic individuals. Temperature monitoring
becomes particularly vital in settings characterized by close human proximity,
often referred to as dense settings. However, existing research on non\-invasive
temperature estimation using thermal cameras has predominantly focused on
sparse settings. Unfortunately, the risk of disease transmission is
significantly higher in dense settings like movie theaters or classrooms.
Consequently, there is an urgent need to develop robust temperature estimation
methods tailored explicitly for dense settings.
  Our study proposes a non\-invasive temperature estimation system that combines
a thermal camera with an edge device. Our system employs YOLO models for face
detection and utilizes a regression framework for temperature estimation. We
evaluated the system on a diverse dataset collected in dense and sparse
settings. Our proposed face detection model achieves an impressive mAP score of
over 84 in both in\-dataset and cross\-dataset evaluations. Furthermore, the
regression framework demonstrates remarkable performance with a mean square
error of 0.18$^\{\\circ\}$C and an impressive $R^2$ score of 0.96. Our
experiments' results highlight the developed system's effectiveness,
positioning it as a promising solution for continuous temperature monitoring in
real\-world applications. With this paper, we release our dataset and
programming code publicly.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2412.00863v1)

---

