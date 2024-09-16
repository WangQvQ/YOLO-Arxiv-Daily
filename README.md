# 每日从arXiv中获取最新YOLO相关论文


## Breaking reCAPTCHAv2

**发布日期**：2024-09-13

**作者**：Andreas Plesner

**摘要**：Our work examines the efficacy of employing advanced machine learning methods
to solve captchas from Google's reCAPTCHAv2 system. We evaluate the
effectiveness of automated systems in solving captchas by utilizing advanced
YOLO models for image segmentation and classification. Our main result is that
we can solve 100% of the captchas, while previous work only solved 68\-71%.
Furthermore, our findings suggest that there is no significant difference in
the number of challenges humans and bots must solve to pass the captchas in
reCAPTCHAv2. This implies that current AI technologies can exploit advanced
image\-based captchas. We also look under the hood of reCAPTCHAv2, and find
evidence that reCAPTCHAv2 is heavily based on cookie and browser history data
when evaluating whether a user is human or not. The code is provided alongside
this paper.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2409.08831v1)

---


## TapToTab : Video\-Based Guitar Tabs Generation using AI and Audio Analysis

**发布日期**：2024-09-13

**作者**：Ali Ghaleb

**摘要**：The automation of guitar tablature generation from video inputs holds
significant promise for enhancing music education, transcription accuracy, and
performance analysis. Existing methods face challenges with consistency and
completeness, particularly in detecting fretboards and accurately identifying
notes. To address these issues, this paper introduces an advanced approach
leveraging deep learning, specifically YOLO models for real\-time fretboard
detection, and Fourier Transform\-based audio analysis for precise note
identification. Experimental results demonstrate substantial improvements in
detection accuracy and robustness compared to traditional techniques. This
paper outlines the development, implementation, and evaluation of these
methodologies, aiming to revolutionize guitar instruction by automating the
creation of guitar tabs from video recordings.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2409.08618v1)

---


## Mamba\-YOLO\-World: Marrying YOLO\-World with Mamba for Open\-Vocabulary Detection

**发布日期**：2024-09-13

**作者**：Haoxuan Wang

**摘要**：Open\-vocabulary detection \(OVD\) aims to detect objects beyond a predefined
set of categories. As a pioneering model incorporating the YOLO series into
OVD, YOLO\-World is well\-suited for scenarios prioritizing speed and
efficiency.However, its performance is hindered by its neck feature fusion
mechanism, which causes the quadratic complexity and the limited guided
receptive fields.To address these limitations, we present Mamba\-YOLO\-World, a
novel YOLO\-based OVD model employing the proposed MambaFusion Path Aggregation
Network \(MambaFusion\-PAN\) as its neck architecture. Specifically, we introduce
an innovative State Space Model\-based feature fusion mechanism consisting of a
Parallel\-Guided Selective Scan algorithm and a Serial\-Guided Selective Scan
algorithm with linear complexity and globally guided receptive fields. It
leverages multi\-modal input sequences and mamba hidden states to guide the
selective scanning process.Experiments demonstrate that our model outperforms
the original YOLO\-World on the COCO and LVIS benchmarks in both zero\-shot and
fine\-tuning settings while maintaining comparable parameters and FLOPs.
Additionally, it surpasses existing state\-of\-the\-art OVD methods with fewer
parameters and FLOPs.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2409.08513v1)

---


## RT\-DETRv3: Real\-time End\-to\-End Object Detection with Hierarchical Dense Positive Supervision

**发布日期**：2024-09-13

**作者**：Shuo Wang

**摘要**：RT\-DETR is the first real\-time end\-to\-end transformer\-based object detector.
Its efficiency comes from the framework design and the Hungarian matching.
However, compared to dense supervision detectors like the YOLO series, the
Hungarian matching provides much sparser supervision, leading to insufficient
model training and difficult to achieve optimal results. To address these
issues, we proposed a hierarchical dense positive supervision method based on
RT\-DETR, named RT\-DETRv3. Firstly, we introduce a CNN\-based auxiliary branch
that provides dense supervision that collaborates with the original decoder to
enhance the encoder feature representation. Secondly, to address insufficient
decoder training, we propose a novel learning strategy involving self\-attention
perturbation. This strategy diversifies label assignment for positive samples
across multiple query groups, thereby enriching positive supervisions.
Additionally, we introduce a shared\-weight decoder branch for dense positive
supervision to ensure more high\-quality queries matching each ground truth.
Notably, all aforementioned modules are training\-only. We conduct extensive
experiments to demonstrate the effectiveness of our approach on COCO val2017.
RT\-DETRv3 significantly outperforms existing real\-time detectors, including the
RT\-DETR series and the YOLO series. For example, RT\-DETRv3\-R18 achieves 48.1%
AP \(\+1.6%/\+1.4%\) compared to RT\-DETR\-R18/RT\-DETRv2\-R18 while maintaining the
same latency. Meanwhile, it requires only half of epochs to attain a comparable
performance. Furthermore, RT\-DETRv3\-R101 can attain an impressive 54.6% AP
outperforming YOLOv10\-X. Code will be released soon.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2409.08475v1)

---


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

