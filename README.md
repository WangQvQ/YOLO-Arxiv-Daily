# 每日从arXiv中获取最新YOLO相关论文


## RMP\-YOLO: A Robust Motion Predictor for Partially Observable Scenarios even if You Only Look Once

**发布日期**：2024-09-18

**作者**：Jiawei Sun

**摘要**：We introduce RMP\-YOLO, a unified framework designed to provide robust motion
predictions even with incomplete input data. Our key insight stems from the
observation that complete and reliable historical trajectory data plays a
pivotal role in ensuring accurate motion prediction. Therefore, we propose a
new paradigm that prioritizes the reconstruction of intact historical
trajectories before feeding them into the prediction modules. Our approach
introduces a novel scene tokenization module to enhance the extraction and
fusion of spatial and temporal features. Following this, our proposed recovery
module reconstructs agents' incomplete historical trajectories by leveraging
local map topology and interactions with nearby agents. The reconstructed,
clean historical data is then integrated into the downstream prediction
modules. Our framework is able to effectively handle missing data of varying
lengths and remains robust against observation noise, while maintaining high
prediction accuracy. Furthermore, our recovery module is compatible with
existing prediction models, ensuring seamless integration. Extensive
experiments validate the effectiveness of our approach, and deployment in
real\-world autonomous vehicles confirms its practical utility. In the 2024
Waymo Motion Prediction Competition, our method, RMP\-YOLO, achieves
state\-of\-the\-art performance, securing third place.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2409.11696v1)

---


## ASMA: An Adaptive Safety Margin Algorithm for Vision\-Language Drone Navigation via Scene\-Aware Control Barrier Functions

**发布日期**：2024-09-16

**作者**：Sourav Sanyal

**摘要**：In the rapidly evolving field of vision\-language navigation \(VLN\), ensuring
robust safety mechanisms remains an open challenge. Control barrier functions
\(CBFs\) are efficient tools which guarantee safety by solving an optimal control
problem. In this work, we consider the case of a teleoperated drone in a VLN
setting, and add safety features by formulating a novel scene\-aware CBF using
ego\-centric observations obtained through an RGB\-D sensor. As a baseline, we
implement a vision\-language understanding module which uses the contrastive
language image pretraining \(CLIP\) model to query about a user\-specified \(in
natural language\) landmark. Using the YOLO \(You Only Look Once\) object
detector, the CLIP model is queried for verifying the cropped landmark,
triggering downstream navigation. To improve navigation safety of the baseline,
we propose ASMA \-\- an Adaptive Safety Margin Algorithm \-\- that crops the
drone's depth map for tracking moving object\(s\) to perform scene\-aware CBF
evaluation on\-the\-fly. By identifying potential risky observations from the
scene, ASMA enables real\-time adaptation to unpredictable environmental
conditions, ensuring optimal safety bounds on a VLN\-powered drone actions.
Using the robot operating system \(ROS\) middleware on a parrot bebop2 quadrotor
in the gazebo environment, ASMA offers 59.4% \- 61.8% increase in success rates
with insignificant 5.4% \- 8.2% increases in trajectory lengths compared to the
baseline CBF\-less VLN while recovering from unsafe situations.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2409.10283v1)

---


## Self\-Updating Vehicle Monitoring Framework Employing Distributed Acoustic Sensing towards Real\-World Settings

**发布日期**：2024-09-16

**作者**：Xi Wang

**摘要**：The recent emergence of Distributed Acoustic Sensing \(DAS\) technology has
facilitated the effective capture of traffic\-induced seismic data. The
traffic\-induced seismic wave is a prominent contributor to urban vibrations and
contain crucial information to advance urban exploration and governance.
However, identifying vehicular movements within massive noisy data poses a
significant challenge. In this study, we introduce a real\-time semi\-supervised
vehicle monitoring framework tailored to urban settings. It requires only a
small fraction of manual labels for initial training and exploits unlabeled
data for model improvement. Additionally, the framework can autonomously adapt
to newly collected unlabeled data. Before DAS data undergo object detection as
two\-dimensional images to preserve spatial information, we leveraged
comprehensive one\-dimensional signal preprocessing to mitigate noise.
Furthermore, we propose a novel prior loss that incorporates the shapes of
vehicular traces to track a single vehicle with varying speeds. To evaluate our
model, we conducted experiments with seismic data from the Stanford 2 DAS
Array. The results showed that our model outperformed the baseline model
Efficient Teacher and its supervised counterpart, YOLO \(You Only Look Once\), in
both accuracy and robustness. With only 35 labeled images, our model surpassed
YOLO's mAP 0.5:0.95 criterion by 18% and showed a 7% increase over Efficient
Teacher. We conducted comparative experiments with multiple update strategies
for self\-updating and identified an optimal approach. This approach surpasses
the performance of non\-overfitting training conducted with all data in a single
pass.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2409.10259v1)

---


## Tracking Virtual Meetings in the Wild: Re\-identification in Multi\-Participant Virtual Meetings

**发布日期**：2024-09-15

**作者**：Oriel Perl

**摘要**：In recent years, workplaces and educational institutes have widely adopted
virtual meeting platforms. This has led to a growing interest in analyzing and
extracting insights from these meetings, which requires effective detection and
tracking of unique individuals. In practice, there is no standardization in
video meetings recording layout, and how they are captured across the different
platforms and services. This, in turn, creates a challenge in acquiring this
data stream and analyzing it in a uniform fashion. Our approach provides a
solution to the most general form of video recording, usually consisting of a
grid of participants \(\\cref\{fig:videomeeting\}\) from a single video source with
no metadata on participant locations, while using the least amount of
constraints and assumptions as to how the data was acquired. Conventional
approaches often use YOLO models coupled with tracking algorithms, assuming
linear motion trajectories akin to that observed in CCTV footage. However, such
assumptions fall short in virtual meetings, where participant video feed window
can abruptly change location across the grid. In an organic video meeting
setting, participants frequently join and leave, leading to sudden, non\-linear
movements on the video grid. This disrupts optical flow\-based tracking methods
that depend on linear motion. Consequently, standard object detection and
tracking methods might mistakenly assign multiple participants to the same
tracker. In this paper, we introduce a novel approach to track and re\-identify
participants in remote video meetings, by utilizing the spatio\-temporal priors
arising from the data in our domain. This, in turn, increases tracking
capabilities compared to the use of general object tracking. Our approach
reduces the error rate by 95% on average compared to YOLO\-based tracking
methods as a baseline.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2409.09841v1)

---


## Stutter\-Solver: End\-to\-end Multi\-lingual Dysfluency Detection

**发布日期**：2024-09-15

**作者**：Xuanru Zhou

**摘要**：Current de\-facto dysfluency modeling methods utilize template matching
algorithms which are not generalizable to out\-of\-domain real\-world dysfluencies
across languages, and are not scalable with increasing amounts of training
data. To handle these problems, we propose Stutter\-Solver: an end\-to\-end
framework that detects dysfluency with accurate type and time transcription,
inspired by the YOLO object detection algorithm. Stutter\-Solver can handle
co\-dysfluencies and is a natural multi\-lingual dysfluency detector. To leverage
scalability and boost performance, we also introduce three novel dysfluency
corpora: VCTK\-Pro, VCTK\-Art, and AISHELL3\-Pro, simulating natural spoken
dysfluencies including repetition, block, missing, replacement, and
prolongation through articulatory\-encodec and TTS\-based methods. Our approach
achieves state\-of\-the\-art performance on all available dysfluency corpora. Code
and datasets are open\-sourced at https://github.com/eureka235/Stutter\-Solver


**代码链接**：https://github.com/eureka235/Stutter-Solver

**论文链接**：[阅读更多](http://arxiv.org/abs/2409.09621v1)

---


## Self\-Prompting Polyp Segmentation in Colonoscopy using Hybrid Yolo\-SAM 2 Model

**发布日期**：2024-09-14

**作者**：Mobina Mansoori

**摘要**：Early diagnosis and treatment of polyps during colonoscopy are essential for
reducing the incidence and mortality of Colorectal Cancer \(CRC\). However, the
variability in polyp characteristics and the presence of artifacts in
colonoscopy images and videos pose significant challenges for accurate and
efficient polyp detection and segmentation. This paper presents a novel
approach to polyp segmentation by integrating the Segment Anything Model \(SAM
2\) with the YOLOv8 model. Our method leverages YOLOv8's bounding box
predictions to autonomously generate input prompts for SAM 2, thereby reducing
the need for manual annotations. We conducted exhaustive tests on five
benchmark colonoscopy image datasets and two colonoscopy video datasets,
demonstrating that our method exceeds state\-of\-the\-art models in both image and
video segmentation tasks. Notably, our approach achieves high segmentation
accuracy using only bounding box annotations, significantly reducing annotation
time and effort. This advancement holds promise for enhancing the efficiency
and scalability of polyp detection in clinical settings
https://github.com/sajjad\-sh33/YOLO\_SAM2.


**代码链接**：https://github.com/sajjad-sh33/YOLO_SAM2.

**论文链接**：[阅读更多](http://arxiv.org/abs/2409.09484v1)

---


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
OVD, YOLO\-World is well\-suited for scenarios prioritizing speed and efficiency.
However, its performance is hindered by its neck feature fusion mechanism,
which causes the quadratic complexity and the limited guided receptive fields.
To address these limitations, we present Mamba\-YOLO\-World, a novel YOLO\-based
OVD model employing the proposed MambaFusion Path Aggregation Network
\(MambaFusion\-PAN\) as its neck architecture. Specifically, we introduce an
innovative State Space Model\-based feature fusion mechanism consisting of a
Parallel\-Guided Selective Scan algorithm and a Serial\-Guided Selective Scan
algorithm with linear complexity and globally guided receptive fields. It
leverages multi\-modal input sequences and mamba hidden states to guide the
selective scanning process. Experiments demonstrate that our model outperforms
the original YOLO\-World on the COCO and LVIS benchmarks in both zero\-shot and
fine\-tuning settings while maintaining comparable parameters and FLOPs.
Additionally, it surpasses existing state\-of\-the\-art OVD methods with fewer
parameters and FLOPs.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2409.08513v3)

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

