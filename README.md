# 每日从arXiv中获取最新YOLO相关论文


## Technical Report of Mobile Manipulator Robot for Industrial Environments

**发布日期**：2024-09-10

**作者**：Erfan Amoozad Khalili

**摘要**：This paper presents the development of the Auriga @Work robot, designed by
the Robotics and Intelligent Automation Lab at Shahid Beheshti University,
Department of Electrical Engineering, for the RoboCup 2024 competition. The
robot is tailored for industrial applications, focusing on enhancing efficiency
in repetitive or hazardous environments. It is equipped with a 4\-wheel Mecanum
drive system for omnidirectional mobility and a 5\-degree\-of\-freedom manipulator
arm with a custom 3D\-printed gripper for object manipulation and navigation
tasks. The robot's electronics are powered by custom\-designed boards utilizing
ESP32 microcontrollers and an Nvidia Jetson Nano for real\-time control and
decision\-making. The key software stack integrates Hector SLAM for mapping, the
A\* algorithm for path planning, and YOLO for object detection, along with
advanced sensor fusion for improved navigation and collision avoidance.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2409.06693v1)

---


## A Semantic Segmentation Approach on Sweet Orange Leaf Diseases Detection Utilizing YOLO

**发布日期**：2024-09-10

**作者**：Sabit Ahamed Preanto

**摘要**：This research introduces an advanced method for diagnosing diseases in sweet
orange leaves by utilising advanced artificial intelligence models like YOLOv8
. Due to their significance as a vital agricultural product, sweet oranges
encounter significant threats from a variety of diseases that harmfully affect
both their yield and quality. Conventional methods for disease detection
primarily depend on manual inspection which is ineffective and frequently leads
to errors, resulting in delayed treatment and increased financial losses. In
response to this challenge, the research utilized YOLOv8 , harnessing their
proficiencies in detecting objects and analyzing images. YOLOv8 is recognized
for its rapid and precise performance, while VIT is acknowledged for its
detailed feature extraction abilities. Impressively, during both the training
and validation stages, YOLOv8 exhibited a perfect accuracy of 80.4%, while VIT
achieved an accuracy of 99.12%, showcasing their potential to transform disease
detection in agriculture. The study comprehensively examined the practical
challenges related to the implementation of AI technologies in agriculture,
encompassing the computational demands and user accessibility, and offering
viable solutions for broader usage. Moreover, it underscores the environmental
considerations, particularly the potential for reduced pesticide usage, thereby
promoting sustainable farming and environmental conservation. These findings
provide encouraging insights into the application of AI in agriculture,
suggesting a transition towards more effective, sustainable, and
technologically advanced farming methods. This research not only highlights the
efficacy of YOLOv8 within a specific agricultural domain but also lays the
foundation for further studies that encompass a broader application in crop
management and sustainable agricultural practices.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2409.06671v1)

---


## An Attribute\-Enriched Dataset and Auto\-Annotated Pipeline for Open Detection

**发布日期**：2024-09-10

**作者**：Pengfei Qi

**摘要**：Detecting objects of interest through language often presents challenges,
particularly with objects that are uncommon or complex to describe, due to
perceptual discrepancies between automated models and human annotators. These
challenges highlight the need for comprehensive datasets that go beyond
standard object labels by incorporating detailed attribute descriptions. To
address this need, we introduce the Objects365\-Attr dataset, an extension of
the existing Objects365 dataset, distinguished by its attribute annotations.
This dataset reduces inconsistencies in object detection by integrating a broad
spectrum of attributes, including color, material, state, texture and tone. It
contains an extensive collection of 5.6M object\-level attribute descriptions,
meticulously annotated across 1.4M bounding boxes. Additionally, to validate
the dataset's effectiveness, we conduct a rigorous evaluation of YOLO\-World at
different scales, measuring their detection performance and demonstrating the
dataset's contribution to advancing object detection.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2409.06300v1)

---


## ALSS\-YOLO: An Adaptive Lightweight Channel Split and Shuffling Network for TIR Wildlife Detection in UAV Imagery

**发布日期**：2024-09-10

**作者**：Ang He

**摘要**：Unmanned aerial vehicles \(UAVs\) equipped with thermal infrared \(TIR\) cameras
play a crucial role in combating nocturnal wildlife poaching. However, TIR
images often face challenges such as jitter, and wildlife overlap,
necessitating UAVs to possess the capability to identify blurred and
overlapping small targets. Current traditional lightweight networks deployed on
UAVs struggle to extract features from blurry small targets. To address this
issue, we developed ALSS\-YOLO, an efficient and lightweight detector optimized
for TIR aerial images. Firstly, we propose a novel Adaptive Lightweight Channel
Split and Shuffling \(ALSS\) module. This module employs an adaptive channel
split strategy to optimize feature extraction and integrates a channel
shuffling mechanism to enhance information exchange between channels. This
improves the extraction of blurry features, crucial for handling jitter\-induced
blur and overlapping targets. Secondly, we developed a Lightweight Coordinate
Attention \(LCA\) module that employs adaptive pooling and grouped convolution to
integrate feature information across dimensions. This module ensures
lightweight operation while maintaining high detection precision and robustness
against jitter and target overlap. Additionally, we developed a single\-channel
focus module to aggregate the width and height information of each channel into
four\-dimensional channel fusion, which improves the feature representation
efficiency of infrared images. Finally, we modify the localization loss
function to emphasize the loss value associated with small objects to improve
localization accuracy. Extensive experiments on the BIRDSAI and ISOD TIR UAV
wildlife datasets show that ALSS\-YOLO achieves state\-of\-the\-art performance,
Our code is openly available at
https://github.com/helloworlder8/computer\_vision.


**代码链接**：https://github.com/helloworlder8/computer_vision.

**论文链接**：[阅读更多](http://arxiv.org/abs/2409.06259v2)

---


## BFA\-YOLO: Balanced multiscale object detection network for multi\-view building facade attachments detection

**发布日期**：2024-09-06

**作者**：Yangguang Chen

**摘要**：Detection of building facade attachments such as doors, windows, balconies,
air conditioner units, billboards, and glass curtain walls plays a pivotal role
in numerous applications. Building facade attachments detection aids in
vbuilding information modeling \(BIM\) construction and meeting Level of Detail 3
\(LOD3\) standards. Yet, it faces challenges like uneven object distribution,
small object detection difficulty, and background interference. To counter
these, we propose BFA\-YOLO, a model for detecting facade attachments in
multi\-view images. BFA\-YOLO incorporates three novel innovations: the Feature
Balanced Spindle Module \(FBSM\) for addressing uneven distribution, the Target
Dynamic Alignment Task Detection Head \(TDATH\) aimed at improving small object
detection, and the Position Memory Enhanced Self\-Attention Mechanism \(PMESA\) to
combat background interference, with each component specifically designed to
solve its corresponding challenge. Detection efficacy of deep network models
deeply depends on the dataset's characteristics. Existing open source datasets
related to building facades are limited by their single perspective, small
image pool, and incomplete category coverage. We propose a novel method for
building facade attachments detection dataset construction and construct the
BFA\-3D dataset for facade attachments detection. The BFA\-3D dataset features
multi\-view, accurate labels, diverse categories, and detailed classification.
BFA\-YOLO surpasses YOLOv8 by 1.8% and 2.9% in mAP@0.5 on the multi\-view BFA\-3D
and street\-view Facade\-WHU datasets, respectively. These results underscore
BFA\-YOLO's superior performance in detecting facade attachments.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2409.04025v1)

---


## YOLO\-CL cluster detection in the Rubin/LSST DC2 simulation

**发布日期**：2024-09-05

**作者**：Kirill Grishin

**摘要**：LSST will provide galaxy cluster catalogs up to z$\\sim$1 that can be used to
constrain cosmological models once their selection function is well\-understood.
We have applied the deep convolutional network YOLO for CLuster detection
\(YOLO\-CL\) to LSST simulations from the Dark Energy Science Collaboration Data
Challenge 2 \(DC2\), and characterized the LSST YOLO\-CL cluster selection
function. We have trained and validated the network on images from a hybrid
sample of \(1\) clusters observed in the Sloan Digital Sky Survey and detected
with the red\-sequence Matched\-filter Probabilistic Percolation, and \(2\)
simulated DC2 dark matter haloes with masses $M\_\{200c\} > 10^\{14\} M\_\{\\odot\}$. We
quantify the completeness and purity of the YOLO\-CL cluster catalog with
respect to DC2 haloes with $M\_\{200c\} > 10^\{14\} M\_\{\\odot\}$. The YOLO\-CL cluster
catalog is 100% and 94% complete for halo mass $M\_\{200c\} > 10^\{14.6\} M\_\{\\odot\}$
at $0.2<z<0.8$, and $M\_\{200c\} > 10^\{14\} M\_\{\\odot\}$ and redshift $z \\lesssim 1$,
respectively, with only 6% false positive detections. All the false positive
detections are dark matter haloes with $ 10^\{13.4\} M\_\{\\odot\} \\lesssim M\_\{200c\}
\\lesssim 10^\{14\} M\_\{\\odot\}$. The YOLO\-CL selection function is almost flat with
respect to the halo mass at $0.2 \\lesssim z \\lesssim 0.9$. The overall
performance of YOLO\-CL is comparable or better than other cluster detection
methods used for current and future optical and infrared surveys. YOLO\-CL shows
better completeness for low mass clusters when compared to current detections
in surveys using the Sunyaev Zel'dovich effect, and detects clusters at higher
redshifts than X\-ray\-based catalogs. The strong advantage of YOLO\-CL over
traditional galaxy cluster detection techniques is that it works directly on
images and does not require photometric and photometric redshift catalogs, nor
does it need to mask stellar sources and artifacts.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2409.03333v1)

---


## YOLO\-PPA based Efficient Traffic Sign Detection for Cruise Control in Autonomous Driving

**发布日期**：2024-09-05

**作者**：Jingyu Zhang

**摘要**：It is very important to detect traffic signs efficiently and accurately in
autonomous driving systems. However, the farther the distance, the smaller the
traffic signs. Existing object detection algorithms can hardly detect these
small scaled signs.In addition, the performance of embedded devices on vehicles
limits the scale of detection models.To address these challenges, a YOLO PPA
based traffic sign detection algorithm is proposed in this paper.The
experimental results on the GTSDB dataset show that compared to the original
YOLO, the proposed method improves inference efficiency by 11.2%. The mAP 50 is
also improved by 93.2%, which demonstrates the effectiveness of the proposed
YOLO PPA.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2409.03320v1)

---


## YoloTag: Vision\-based Robust UAV Navigation with Fiducial Markers

**发布日期**：2024-09-03

**作者**：Sourav Raxit

**摘要**：By harnessing fiducial markers as visual landmarks in the environment,
Unmanned Aerial Vehicles \(UAVs\) can rapidly build precise maps and navigate
spaces safely and efficiently, unlocking their potential for fluent
collaboration and coexistence with humans. Existing fiducial marker methods
rely on handcrafted feature extraction, which sacrifices accuracy. On the other
hand, deep learning pipelines for marker detection fail to meet real\-time
runtime constraints crucial for navigation applications. In this work, we
propose YoloTag \\textemdash a real\-time fiducial marker\-based localization
system. YoloTag uses a lightweight YOLO v8 object detector to accurately detect
fiducial markers in images while meeting the runtime constraints needed for
navigation. The detected markers are then used by an efficient
perspective\-n\-point algorithm to estimate UAV states. However, this
localization system introduces noise, causing instability in trajectory
tracking. To suppress noise, we design a higher\-order Butterworth filter that
effectively eliminates noise through frequency domain analysis. We evaluate our
algorithm through real\-robot experiments in an indoor environment, comparing
the trajectory tracking performance of our method against other approaches in
terms of several distance metrics.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2409.02334v1)

---


## DS MYOLO: A Reliable Object Detector Based on SSMs for Driving Scenarios

**发布日期**：2024-09-02

**作者**：Yang Li

**摘要**：Accurate real\-time object detection enhances the safety of advanced
driver\-assistance systems, making it an essential component in driving
scenarios. With the rapid development of deep learning technology, CNN\-based
YOLO real\-time object detectors have gained significant attention. However, the
local focus of CNNs results in performance bottlenecks. To further enhance
detector performance, researchers have introduced Transformer\-based
self\-attention mechanisms to leverage global receptive fields, but their
quadratic complexity incurs substantial computational costs. Recently, Mamba,
with its linear complexity, has made significant progress through global
selective scanning. Inspired by Mamba's outstanding performance, we propose a
novel object detector: DS MYOLO. This detector captures global feature
information through a simplified selective scanning fusion block \(SimVSS Block\)
and effectively integrates the network's deep features. Additionally, we
introduce an efficient channel attention convolution \(ECAConv\) that enhances
cross\-channel feature interaction while maintaining low computational
complexity. Extensive experiments on the CCTSDB 2021 and VLD\-45 driving
scenarios datasets demonstrate that DS MYOLO exhibits significant potential and
competitive advantage among similarly scaled YOLO series real\-time object
detectors.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2409.01093v1)

---


## A method for detecting dead fish on large water surfaces based on improved YOLOv10

**发布日期**：2024-08-31

**作者**：Qingbin Tian

**摘要**：Dead fish frequently appear on the water surface due to various factors. If
not promptly detected and removed, these dead fish can cause significant issues
such as water quality deterioration, ecosystem damage, and disease
transmission. Consequently, it is imperative to develop rapid and effective
detection methods to mitigate these challenges. Conventional methods for
detecting dead fish are often constrained by manpower and time limitations,
struggling to effectively manage the intricacies of aquatic environments. This
paper proposes an end\-to\-end detection model built upon an enhanced YOLOv10
framework, designed specifically to swiftly and precisely detect deceased fish
across extensive water surfaces.Key enhancements include: \(1\) Replacing
YOLOv10's backbone network with FasterNet to reduce model complexity while
maintaining high detection accuracy; \(2\) Improving feature fusion in the Neck
section through enhanced connectivity methods and replacing the original C2f
module with CSPStage modules; \(3\) Adding a compact target detection head to
enhance the detection performance of smaller objects. Experimental results
demonstrate significant improvements in P\(precision\), R\(recall\), and AP\(average
precision\) compared to the baseline model YOLOv10n. Furthermore, our model
outperforms other models in the YOLO series by significantly reducing model
size and parameter count, while sustaining high inference speed and achieving
optimal AP performance. The model facilitates rapid and accurate detection of
dead fish in large\-scale aquaculture systems. Finally, through ablation
experiments, we systematically analyze and assess the contribution of each
model component to the overall system performance.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2409.00388v1)

---

