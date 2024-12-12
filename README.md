# 每日从arXiv中获取最新YOLO相关论文


## SPACE\-SUIT: An Artificial Intelligence based chromospheric feature extractor and classifier for SUIT

**发布日期**：2024-12-11

**作者**：Pranava Seth

**摘要**：The Solar Ultraviolet Imaging Telescope\(SUIT\) onboard Aditya\-L1 is an imager
that observes the solar photosphere and chromosphere through observations in
the wavelength range of 200\-400 nm. A comprehensive understanding of the plasma
and thermodynamic properties of chromospheric and photospheric morphological
structures requires a large sample statistical study, necessitating the
development of automatic feature detection methods. To this end, we develop the
feature detection algorithm SPACE\-SUIT: Solar Phenomena Analysis and
Classification using Enhanced vision techniques for SUIT, to detect and
classify the solar chromospheric features to be observed from SUIT's Mg II k
filter. Specifically, we target plage regions, sunspots, filaments, and
off\-limb structures. SPACE uses You Only Look Once\(YOLO\), a neural
network\-based model to identify regions of interest. We train and validate
SPACE using mock\-SUIT images developed from Interface Region Imaging
Spectrometer\(IRIS\) full\-disk mosaic images in Mg II k line, while we also
perform detection on Level\-1 SUIT data. SPACE achieves an approximate precision
of 0.788, recall 0.863 and MAP of 0.874 on the validation mock SUIT FITS
dataset. Given the manual labeling of our dataset, we perform "self\-validation"
by applying statistical measures and Tamura features on the ground truth and
predicted bounding boxes. We find the distributions of entropy, contrast,
dissimilarity, and energy to show differences in the features. These
differences are qualitatively captured by the detected regions predicted by
SPACE and validated with the observed SUIT images, even in the absence of
labeled ground truth. This work not only develops a chromospheric feature
extractor but also demonstrates the effectiveness of statistical metrics and
Tamura features for distinguishing chromospheric features, offering independent
validation for future detection schemes.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2412.08589v1)

---


## DynamicPAE: Generating Scene\-Aware Physical Adversarial Examples in Real\-Time

**发布日期**：2024-12-11

**作者**：Jin Hu

**摘要**：Physical adversarial examples \(PAEs\) are regarded as "whistle\-blowers" of
real\-world risks in deep\-learning applications. However, current PAE generation
studies show limited adaptive attacking ability to diverse and varying scenes.
The key challenges in generating dynamic PAEs are exploring their patterns
under noisy gradient feedback and adapting the attack to agnostic scenario
natures. To address the problems, we present DynamicPAE, the first generative
framework that enables scene\-aware real\-time physical attacks beyond static
attacks. Specifically, to train the dynamic PAE generator under noisy gradient
feedback, we introduce the residual\-driven sample trajectory guidance
technique, which redefines the training task to break the limited feedback
information restriction that leads to the degeneracy problem. Intuitively, it
allows the gradient feedback to be passed to the generator through a low\-noise
auxiliary task, thereby guiding the optimization away from degenerate solutions
and facilitating a more comprehensive and stable exploration of feasible PAEs.
To adapt the generator to agnostic scenario natures, we introduce the
context\-aligned scene expectation simulation process, consisting of the
conditional\-uncertainty\-aligned data module and the skewness\-aligned objective
re\-weighting module. The former enhances robustness in the context of
incomplete observation by employing a conditional probabilistic model for
domain randomization, while the latter facilitates consistent stealth control
across different attack targets by automatically reweighting losses based on
the skewness indicator. Extensive digital and physical evaluations demonstrate
the superior attack performance of DynamicPAE, attaining a 1.95 $\\times$ boost
\(65.55% average AP drop under attack\) on representative object detectors \(e.g.,
Yolo\-v8\) over state\-of\-the\-art static PAE generating methods.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2412.08053v1)

---


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

