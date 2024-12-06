# 每日从arXiv中获取最新YOLO相关论文


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

**论文链接**：[阅读更多](http://arxiv.org/abs/2412.01983v1)

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


## Visual Modality Prompt for Adapting Vision\-Language Object Detectors

**发布日期**：2024-12-01

**作者**：Heitor R. Medeiros

**摘要**：The zero\-shot performance of object detectors degrades when tested on
different modalities, such as infrared and depth. While recent work has
explored image translation techniques to adapt detectors to new modalities,
these methods are limited to a single modality and apply only to traditional
detectors. Recently, vision\-language detectors, such as YOLO\-World and
Grounding DINO, have shown promising zero\-shot capabilities, however, they have
not yet been adapted for other visual modalities. Traditional fine\-tuning
approaches tend to compromise the zero\-shot capabilities of the detectors. The
visual prompt strategies commonly used for classification with vision\-language
models apply the same linear prompt translation to each image making them less
effective. To address these limitations, we propose ModPrompt, a visual prompt
strategy to adapt vision\-language detectors to new modalities without degrading
zero\-shot performance. In particular, an encoder\-decoder visual prompt strategy
is proposed, further enhanced by the integration of inference\-friendly task
residuals, facilitating more robust adaptation. Empirically, we benchmark our
method for modality adaptation on two vision\-language detectors, YOLO\-World and
Grounding DINO, and on challenging infrared \(LLVIP, FLIR\) and depth \(NYUv2\)
data, achieving performance comparable to full fine\-tuning while preserving the
model's zero\-shot capability. Our code is available at:
https://github.com/heitorrapela/ModPrompt


**代码链接**：https://github.com/heitorrapela/ModPrompt

**论文链接**：[阅读更多](http://arxiv.org/abs/2412.00622v1)

---


## Real\-Time Anomaly Detection in Video Streams

**发布日期**：2024-11-29

**作者**：Fabien Poirier

**摘要**：This thesis is part of a CIFRE agreement between the company Othello and the
LIASD laboratory. The objective is to develop an artificial intelligence system
that can detect real\-time dangers in a video stream. To achieve this, a novel
approach combining temporal and spatial analysis has been proposed. Several
avenues have been explored to improve anomaly detection by integrating object
detection, human pose detection, and motion analysis. For result
interpretability, techniques commonly used for image analysis, such as
activation and saliency maps, have been extended to videos, and an original
method has been proposed. The proposed architecture performs binary or
multiclass classification depending on whether an alert or the cause needs to
be identified. Numerous neural networkmodels have been tested, and three of
them have been selected. You Only Looks Once \(YOLO\) has been used for spatial
analysis, a Convolutional Recurrent Neuronal Network \(CRNN\) composed of VGG19
and a Gated Recurrent Unit \(GRU\) for temporal analysis, and a multi\-layer
perceptron for classification. These models handle different types of data and
can be combined in parallel or in series. Although the parallel mode is faster,
the serial mode is generally more reliable. For training these models,
supervised learning was chosen, and two proprietary datasets were created. The
first dataset focuses on objects that may play a potential role in anomalies,
while the second consists of videos containing anomalies or non\-anomalies. This
approach allows for the processing of both continuous video streams and finite
videos, providing greater flexibility in detection.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2411.19731v1)

---


## Data Augmentation with Diffusion Models for Colon Polyp Localization on the Low Data Regime: How much real data is enough?

**发布日期**：2024-11-28

**作者**：Adrian Tormos

**摘要**：The scarcity of data in medical domains hinders the performance of Deep
Learning models. Data augmentation techniques can alleviate that problem, but
they usually rely on functional transformations of the data that do not
guarantee to preserve the original tasks. To approximate the distribution of
the data using generative models is a way of reducing that problem and also to
obtain new samples that resemble the original data. Denoising Diffusion models
is a promising Deep Learning technique that can learn good approximations of
different kinds of data like images, time series or tabular data.
  Automatic colonoscopy analysis and specifically Polyp localization in
colonoscopy videos is a task that can assist clinical diagnosis and treatment.
The annotation of video frames for training a deep learning model is a time
consuming task and usually only small datasets can be obtained. The fine tuning
of application models using a large dataset of generated data could be an
alternative to improve their performance. We conduct a set of experiments
training different diffusion models that can generate jointly colonoscopy
images with localization annotations using a combination of existing open
datasets. The generated data is used on various transfer learning experiments
in the task of polyp localization with a model based on YOLO v9 on the low data
regime.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2411.18926v1)

---

