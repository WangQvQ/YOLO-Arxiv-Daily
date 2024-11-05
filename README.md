# 每日从arXiv中获取最新YOLO相关论文


## A Visual Question Answering Method for SAR Ship: Breaking the Requirement for Multimodal Dataset Construction and Model Fine\-Tuning

**发布日期**：2024-11-03

**作者**：Fei Wang

**摘要**：Current visual question answering \(VQA\) tasks often require constructing
multimodal datasets and fine\-tuning visual language models, which demands
significant time and resources. This has greatly hindered the application of
VQA to downstream tasks, such as ship information analysis based on Synthetic
Aperture Radar \(SAR\) imagery. To address this challenge, this letter proposes a
novel VQA approach that integrates object detection networks with visual
language models, specifically designed for analyzing ships in SAR images. This
integration aims to enhance the capabilities of VQA systems, focusing on
aspects such as ship location, density, and size analysis, as well as risk
behavior detection. Initially, we conducted baseline experiments using YOLO
networks on two representative SAR ship detection datasets, SSDD and HRSID, to
assess each model's performance in terms of detection accuracy. Based on these
results, we selected the optimal model, YOLOv8n, as the most suitable detection
network for this task. Subsequently, leveraging the vision\-language model
Qwen2\-VL, we designed and implemented a VQA task specifically for SAR scenes.
This task employs the ship location and size information output by the
detection network to generate multi\-turn dialogues and scene descriptions for
SAR imagery. Experimental results indicate that this method not only enables
fundamental SAR scene question\-answering without the need for additional
datasets or fine\-tuning but also dynamically adapts to complex, multi\-turn
dialogue requirements, demonstrating robust semantic understanding and
adaptability.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2411.01445v1)

---


## Autobiasing Event Cameras

**发布日期**：2024-11-01

**作者**：Mehdi Sefidgar Dilmaghani

**摘要**：This paper presents an autonomous method to address challenges arising from
severe lighting conditions in machine vision applications that use event
cameras. To manage these conditions, the research explores the built in
potential of these cameras to adjust pixel functionality, named bias settings.
As cars are driven at various times and locations, shifts in lighting
conditions are unavoidable. Consequently, this paper utilizes the neuromorphic
YOLO\-based face tracking module of a driver monitoring system as the
event\-based application to study. The proposed method uses numerical metrics to
continuously monitor the performance of the event\-based application in
real\-time. When the application malfunctions, the system detects this through a
drop in the metrics and automatically adjusts the event cameras bias values.
The Nelder\-Mead simplex algorithm is employed to optimize this adjustment, with
finetuning continuing until performance returns to a satisfactory level. The
advantage of bias optimization lies in its ability to handle conditions such as
flickering or darkness without requiring additional hardware or software. To
demonstrate the capabilities of the proposed system, it was tested under
conditions where detecting human faces with default bias values was impossible.
These severe conditions were simulated using dim ambient light and various
flickering frequencies. Following the automatic and dynamic process of bias
modification, the metrics for face detection significantly improved under all
conditions. Autobiasing resulted in an increase in the YOLO confidence
indicators by more than 33 percent for object detection and 37 percent for face
detection highlighting the effectiveness of the proposed method.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2411.00729v1)

---


## Generative AI\-based Pipeline Architecture for Increasing Training Efficiency in Intelligent Weed Control Systems

**发布日期**：2024-11-01

**作者**：Sourav Modak

**摘要**：In automated crop protection tasks such as weed control, disease diagnosis,
and pest monitoring, deep learning has demonstrated significant potential.
However, these advanced models rely heavily on high\-quality, diverse datasets,
often limited and costly in agricultural settings. Traditional data
augmentation can increase dataset volume but usually lacks the real\-world
variability needed for robust training. This study presents a new approach for
generating synthetic images to improve deep learning\-based object detection
models for intelligent weed control. Our GenAI\-based image generation pipeline
integrates the Segment Anything Model \(SAM\) for zero\-shot domain adaptation
with a text\-to\-image Stable Diffusion Model, enabling the creation of synthetic
images that capture diverse real\-world conditions. We evaluate these synthetic
datasets using lightweight YOLO models, measuring data efficiency with mAP50
and mAP50\-95 scores across varying proportions of real and synthetic data.
Notably, YOLO models trained on datasets with 10% synthetic and 90% real images
generally demonstrate superior mAP50 and mAP50\-95 scores compared to those
trained solely on real images. This approach not only reduces dependence on
extensive real\-world datasets but also enhances predictive performance. The
integration of this approach opens opportunities for achieving continual
self\-improvement of perception modules in intelligent technical systems.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2411.00548v1)

---


## LAM\-YOLO: Drones\-based Small Object Detection on Lighting\-Occlusion Attention Mechanism YOLO

**发布日期**：2024-11-01

**作者**：Yuchen Zheng

**摘要**：Drone\-based target detection presents inherent challenges, such as the high
density and overlap of targets in drone\-based images, as well as the blurriness
of targets under varying lighting conditions, which complicates identification.
Traditional methods often struggle to recognize numerous densely packed small
targets under complex background. To address these challenges, we propose
LAM\-YOLO, an object detection model specifically designed for drone\-based.
First, we introduce a light\-occlusion attention mechanism to enhance the
visibility of small targets under different lighting conditions. Meanwhile, we
incroporate incorporate Involution modules to improve interaction among feature
layers. Second, we utilize an improved SIB\-IoU as the regression loss function
to accelerate model convergence and enhance localization accuracy. Finally, we
implement a novel detection strategy that introduces two auxiliary detection
heads for identifying smaller\-scale targets.Our quantitative results
demonstrate that LAM\-YOLO outperforms methods such as Faster R\-CNN, YOLOv9, and
YOLOv10 in terms of mAP@0.5 and mAP@0.5:0.95 on the VisDrone2019 public
dataset. Compared to the original YOLOv8, the average precision increases by
7.1\\%. Additionally, the proposed SIB\-IoU loss function shows improved faster
convergence speed during training and improved average precision over the
traditional loss function.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2411.00485v1)

---


## Evaluating the Evolution of YOLO \(You Only Look Once\) Models: A Comprehensive Benchmark Study of YOLO11 and Its Predecessors

**发布日期**：2024-10-31

**作者**：Nidhal Jegham

**摘要**：This study presents a comprehensive benchmark analysis of various YOLO \(You
Only Look Once\) algorithms, from YOLOv3 to the newest addition. It represents
the first research to comprehensively evaluate the performance of YOLO11, the
latest addition to the YOLO family. It evaluates their performance on three
diverse datasets: Traffic Signs \(with varying object sizes\), African Wildlife
\(with diverse aspect ratios and at least one instance of the object per image\),
and Ships and Vessels \(with small\-sized objects of a single class\), ensuring a
comprehensive assessment across datasets with distinct challenges. To ensure a
robust evaluation, we employ a comprehensive set of metrics, including
Precision, Recall, Mean Average Precision \(mAP\), Processing Time, GFLOPs count,
and Model Size. Our analysis highlights the distinctive strengths and
limitations of each YOLO version. For example: YOLOv9 demonstrates substantial
accuracy but struggles with detecting small objects and efficiency whereas
YOLOv10 exhibits relatively lower accuracy due to architectural choices that
affect its performance in overlapping object detection but excels in speed and
efficiency. Additionally, the YOLO11 family consistently shows superior
performance in terms of accuracy, speed, computational efficiency, and model
size. YOLO11m achieved a remarkable balance of accuracy and efficiency, scoring
mAP50\-95 scores of 0.795, 0.81, and 0.325 on the Traffic Signs, African
Wildlife, and Ships datasets, respectively, while maintaining an average
inference time of 2.4ms, a model size of 38.8Mb, and around 67.6 GFLOPs on
average. These results provide critical insights for both industry and
academia, facilitating the selection of the most suitable YOLO algorithm for
diverse applications and guiding future enhancements.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2411.00201v1)

---


## Whole\-Herd Elephant Pose Estimation from Drone Data for Collective Behavior Analysis

**发布日期**：2024-10-31

**作者**：Brody McNutt

**摘要**：This research represents a pioneering application of automated pose
estimation from drone data to study elephant behavior in the wild, utilizing
video footage captured from Samburu National Reserve, Kenya. The study
evaluates two pose estimation workflows: DeepLabCut, known for its application
in laboratory settings and emerging wildlife fieldwork, and YOLO\-NAS\-Pose, a
newly released pose estimation model not previously applied to wildlife
behavioral studies. These models are trained to analyze elephant herd behavior,
focusing on low\-resolution \($\\sim$50 pixels\) subjects to detect key points such
as the head, spine, and ears of multiple elephants within a frame. Both
workflows demonstrated acceptable quality of pose estimation on the test set,
facilitating the automated detection of basic behaviors crucial for studying
elephant herd dynamics. For the metrics selected for pose estimation evaluation
on the test set \-\- root mean square error \(RMSE\), percentage of correct
keypoints \(PCK\), and object keypoint similarity \(OKS\) \-\- the YOLO\-NAS\-Pose
workflow outperformed DeepLabCut. Additionally, YOLO\-NAS\-Pose exceeded
DeepLabCut in object detection evaluation. This approach introduces a novel
method for wildlife behavioral research, including the burgeoning field of
wildlife drone monitoring, with significant implications for wildlife
conservation.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2411.00196v1)

---


## YOLOv11 for Vehicle Detection: Advancements, Performance, and Applications in Intelligent Transportation Systems

**发布日期**：2024-10-30

**作者**：Mujadded Al Rabbani Alif

**摘要**：Accurate vehicle detection is essential for the development of intelligent
transportation systems, autonomous driving, and traffic monitoring. This paper
presents a detailed analysis of YOLO11, the latest advancement in the YOLO
series of deep learning models, focusing exclusively on vehicle detection
tasks. Building upon the success of its predecessors, YOLO11 introduces
architectural improvements designed to enhance detection speed, accuracy, and
robustness in complex environments. Using a comprehensive dataset comprising
multiple vehicle types\-cars, trucks, buses, motorcycles, and bicycles we
evaluate YOLO11's performance using metrics such as precision, recall, F1
score, and mean average precision \(mAP\). Our findings demonstrate that YOLO11
surpasses previous versions \(YOLOv8 and YOLOv10\) in detecting smaller and more
occluded vehicles while maintaining a competitive inference time, making it
well\-suited for real\-time applications. Comparative analysis shows significant
improvements in the detection of complex vehicle geometries, further
contributing to the development of efficient and scalable vehicle detection
systems. This research highlights YOLO11's potential to enhance autonomous
vehicle performance and traffic monitoring systems, offering insights for
future developments in the field.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2410.22898v1)

---


## From Explicit Rules to Implicit Reasoning in an Interpretable Violence Monitoring System

**发布日期**：2024-10-29

**作者**：Wen\-Dong Jiang

**摘要**：Recently, research based on pre\-trained models has demonstrated outstanding
performance in violence surveillance tasks. However, these black\-box systems
face challenges regarding explainability during training and inference
processes. An important question is how to incorporate explicit knowledge into
these implicit models, thereby designing expert\-driven and interpretable
violence surveillance systems. This paper proposes a new paradigm for weakly
supervised violence monitoring \(WSVM\) called Rule base Violence monitoring
\(RuleVM\). The proposed RuleVM uses a dual\-branch structure for different
designs for images and text. One of the branches is called the implicit branch,
which uses only visual features for coarse\-grained binary classification. In
this branch, image feature extraction is divided into two channels: one
responsible for extracting scene frames and the other focusing on extracting
actions. The other branch is called the explicit branch, which utilizes
language\-image alignment to perform fine\-grained classification. For the
language channel design in the explicit branch, the proposed RuleCLIP uses the
state\-of\-the\-art YOLO\-World model to detect objects and actions in video
frames, and association rules are identified through data mining methods as
descriptions of the video. Leveraging the dual\-branch architecture, RuleVM
achieves interpretable coarse\-grained and fine\-grained violence surveillance.
Extensive experiments were conducted on two commonly used benchmarks, and the
results show that RuleCLIP achieved the best performance in both coarse\-grained
and fine\-grained detection, significantly outperforming existing
state\-of\-the\-art methods. Moreover, interpretability experiments uncovered some
interesting rules, such as the observation that as the number of people
increases, the risk level of violent behavior also rises.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2410.21991v3)

---


## PK\-YOLO: Pretrained Knowledge Guided YOLO for Brain Tumor Detection in Multiplanar MRI Slices

**发布日期**：2024-10-29

**作者**：Ming Kang

**摘要**：Brain tumor detection in multiplane Magnetic Resonance Imaging \(MRI\) slices
is a challenging task due to the various appearances and relationships in the
structure of the multiplane images. In this paper, we propose a new You Only
Look Once \(YOLO\)\-based detection model that incorporates Pretrained Knowledge
\(PK\), called PK\-YOLO, to improve the performance for brain tumor detection in
multiplane MRI slices. To our best knowledge, PK\-YOLO is the first pretrained
knowledge guided YOLO\-based object detector. The main components of the new
method are a pretrained pure lightweight convolutional neural network\-based
backbone via sparse masked modeling, a YOLO architecture with the pretrained
backbone, and a regression loss function for improving small object detection.
The pretrained backbone allows for feature transferability of object queries on
individual plane MRI slices into the model encoders, and the learned domain
knowledge base can improve in\-domain detection. The improved loss function can
further boost detection performance on small\-size brain tumors in multiplanar
two\-dimensional MRI slices. Experimental results show that the proposed PK\-YOLO
achieves competitive performance on the multiplanar MRI brain tumor detection
datasets compared to state\-of\-the\-art YOLO\-like and DETR\-like object detectors.
The code is available at https://github.com/mkang315/PK\-YOLO.


**代码链接**：https://github.com/mkang315/PK-YOLO.

**论文链接**：[阅读更多](http://arxiv.org/abs/2410.21822v1)

---


## TACO: Adversarial Camouflage Optimization on Trucks to Fool Object Detectors

**发布日期**：2024-10-28

**作者**：Adonisz Dimitriu

**摘要**：Adversarial attacks threaten the reliability of machine learning models in
critical applications like autonomous vehicles and defense systems. As object
detectors become more robust with models like YOLOv8, developing effective
adversarial methodologies is increasingly challenging. We present Truck
Adversarial Camouflage Optimization \(TACO\), a novel framework that generates
adversarial camouflage patterns on 3D vehicle models to deceive
state\-of\-the\-art object detectors. Adopting Unreal Engine 5, TACO integrates
differentiable rendering with a Photorealistic Rendering Network to optimize
adversarial textures targeted at YOLOv8. To ensure the generated textures are
both effective in deceiving detectors and visually plausible, we introduce the
Convolutional Smooth Loss function, a generalized smooth loss function.
Experimental evaluations demonstrate that TACO significantly degrades YOLOv8's
detection performance, achieving an AP@0.5 of 0.0099 on unseen test data.
Furthermore, these adversarial patterns exhibit strong transferability to other
object detection models such as Faster R\-CNN and earlier YOLO versions.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2410.21443v1)

---

