# 每日从arXiv中获取最新YOLO相关论文


## "Pass the butter": A study on desktop\-classic multitasking robotic arm based on advanced YOLOv7 and BERT

**发布日期**：2024-05-27

**作者**：Haohua Que

**摘要**：In recent years, various intelligent autonomous robots have begun to appear
in daily life and production. Desktop\-level robots are characterized by their
flexible deployment, rapid response, and suitability for light workload
environments. In order to meet the current societal demand for service robot
technology, this study proposes using a miniaturized desktop\-level robot \(by
ROS\) as a carrier, locally deploying a natural language model \(NLP\-BERT\), and
integrating visual recognition \(CV\-YOLO\) and speech recognition technology
\(ASR\-Whisper\) as inputs to achieve autonomous decision\-making and rational
action by the desktop robot. Three comprehensive experiments were designed to
validate the robotic arm, and the results demonstrate excellent performance
using this approach across all three experiments. In Task 1, the execution
rates for speech recognition and action performance were 92.6% and 84.3%,
respectively. In Task 2, the highest execution rates under the given conditions
reached 92.1% and 84.6%, while in Task 3, the highest execution rates were
95.2% and 80.8%, respectively. Therefore, it can be concluded that the proposed
solution integrating ASR, NLP, and other technologies on edge devices is
feasible and provides a technical and engineering foundation for realizing
multimodal desktop\-level robots.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2405.17250v1)

---


## Evaluation of Resource\-Efficient Crater Detectors on Embedded Systems

**发布日期**：2024-05-27

**作者**：Simon Vellas

**摘要**：Real\-time analysis of Martian craters is crucial for mission\-critical
operations, including safe landings and geological exploration. This work
leverages the latest breakthroughs for on\-the\-edge crater detection aboard
spacecraft. We rigorously benchmark several YOLO networks using a Mars craters
dataset, analyzing their performance on embedded systems with a focus on
optimization for low\-power devices. We optimize this process for a new wave of
cost\-effective, commercial\-off\-the\-shelf\-based smaller satellites.
Implementations on diverse platforms, including Google Coral Edge TPU, AMD
Versal SoC VCK190, Nvidia Jetson Nano and Jetson AGX Orin, undergo a detailed
trade\-off analysis. Our findings identify optimal network\-device pairings,
enhancing the feasibility of crater detection on resource\-constrained hardware
and setting a new precedent for efficient and resilient extraterrestrial
imaging. Code at: https://github.com/billpsomas/mars\_crater\_detection.


**代码链接**：https://github.com/billpsomas/mars_crater_detection.

**论文链接**：[阅读更多](http://arxiv.org/abs/2405.16953v1)

---


## Enhancing Pollinator Conservation towards Agriculture 4.0: Monitoring of Bees through Object Recognition

**发布日期**：2024-05-24

**作者**：Ajay John Alex

**摘要**：In an era of rapid climate change and its adverse effects on food production,
technological intervention to monitor pollinator conservation is of paramount
importance for environmental monitoring and conservation for global food
security. The survival of the human species depends on the conservation of
pollinators. This article explores the use of Computer Vision and Object
Recognition to autonomously track and report bee behaviour from images. A novel
dataset of 9664 images containing bees is extracted from video streams and
annotated with bounding boxes. With training, validation and testing sets
\(6722, 1915, and 997 images, respectively\), the results of the COCO\-based YOLO
model fine\-tuning approaches show that YOLOv5m is the most effective approach
in terms of recognition accuracy. However, YOLOv5s was shown to be the most
optimal for real\-time bee detection with an average processing and inference
time of 5.1ms per video frame at the cost of slightly lower ability. The
trained model is then packaged within an explainable AI interface, which
converts detection events into timestamped reports and charts, with the aim of
facilitating use by non\-technical users such as expert stakeholders from the
apiculture industry towards informing responsible consumption and production.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2405.15428v1)

---


## YOLOv10: Real\-Time End\-to\-End Object Detection

**发布日期**：2024-05-23

**作者**：Ao Wang

**摘要**：Over the past years, YOLOs have emerged as the predominant paradigm in the
field of real\-time object detection owing to their effective balance between
computational cost and detection performance. Researchers have explored the
architectural designs, optimization objectives, data augmentation strategies,
and others for YOLOs, achieving notable progress. However, the reliance on the
non\-maximum suppression \(NMS\) for post\-processing hampers the end\-to\-end
deployment of YOLOs and adversely impacts the inference latency. Besides, the
design of various components in YOLOs lacks the comprehensive and thorough
inspection, resulting in noticeable computational redundancy and limiting the
model's capability. It renders the suboptimal efficiency, along with
considerable potential for performance improvements. In this work, we aim to
further advance the performance\-efficiency boundary of YOLOs from both the
post\-processing and model architecture. To this end, we first present the
consistent dual assignments for NMS\-free training of YOLOs, which brings
competitive performance and low inference latency simultaneously. Moreover, we
introduce the holistic efficiency\-accuracy driven model design strategy for
YOLOs. We comprehensively optimize various components of YOLOs from both
efficiency and accuracy perspectives, which greatly reduces the computational
overhead and enhances the capability. The outcome of our effort is a new
generation of YOLO series for real\-time end\-to\-end object detection, dubbed
YOLOv10. Extensive experiments show that YOLOv10 achieves state\-of\-the\-art
performance and efficiency across various model scales. For example, our
YOLOv10\-S is 1.8$\\times$ faster than RT\-DETR\-R18 under the similar AP on COCO,
meanwhile enjoying 2.8$\\times$ smaller number of parameters and FLOPs. Compared
with YOLOv9\-C, YOLOv10\-B has 46\\% less latency and 25\\% fewer parameters for
the same performance.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2405.14458v1)

---


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

