# 每日从arXiv中获取最新YOLO相关论文


## Deep Learning Models for UAV\-Assisted Bridge Inspection: A YOLO Benchmark Analysis

**发布日期**：2024-11-07

**作者**：Trong\-Nhan Phan

**摘要**：Visual inspections of bridges are critical to ensure their safety and
identify potential failures early. This inspection process can be rapidly and
accurately automated by using unmanned aerial vehicles \(UAVs\) integrated with
deep learning models. However, choosing an appropriate model that is
lightweight enough to integrate into the UAV and fulfills the strict
requirements for inference time and accuracy is challenging. Therefore, our
work contributes to the advancement of this model selection process by
conducting a benchmark of 23 models belonging to the four newest YOLO variants
\(YOLOv5, YOLOv6, YOLOv7, YOLOv8\) on COCO\-Bridge\-2021\+, a dataset for bridge
details detection. Through comprehensive benchmarking, we identify YOLOv8n,
YOLOv7tiny, YOLOv6m, and YOLOv6m6 as the models offering an optimal balance
between accuracy and processing speed, with mAP@50 scores of 0.803, 0.837,
0.853, and 0.872, and inference times of 5.3ms, 7.5ms, 14.06ms, and 39.33ms,
respectively. Our findings accelerate the model selection process for UAVs,
enabling more efficient and reliable bridge inspections.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2411.04475v1)

---


## Self\-supervised cross\-modality learning for uncertainty\-aware object detection and recognition in applications which lack pre\-labelled training data

**发布日期**：2024-11-05

**作者**：Irum Mehboob

**摘要**：This paper shows how an uncertainty\-aware, deep neural network can be trained
to detect, recognise and localise objects in 2D RGB images, in applications
lacking annotated train\-ng datasets. We propose a self\-supervising
teacher\-student pipeline, in which a relatively simple teacher classifier,
trained with only a few labelled 2D thumbnails, automatically processes a
larger body of unlabelled RGB\-D data to teach a student network based on a
modified YOLOv3 architecture. Firstly, 3D object detection with back projection
is used to automatically extract and teach 2D detection and localisation
information to the student network. Secondly, a weakly supervised 2D thumbnail
classifier, with minimal training on a small number of hand\-labelled images, is
used to teach object category recognition. Thirdly, we use a Gaussian Process
GP to encode and teach a robust uncertainty estimation functionality, so that
the student can output confidence scores with each categorization. The
resulting student significantly outperforms the same YOLO architecture trained
directly on the same amount of labelled data. Our GP\-based approach yields
robust and meaningful uncertainty estimations for complex industrial object
classifications. The end\-to\-end network is also capable of real\-time
processing, needed for robotics applications. Our method can be applied to many
important industrial tasks, where labelled datasets are typically unavailable.
In this paper, we demonstrate an example of detection, localisation, and object
category recognition of nuclear mixed\-waste materials in highly cluttered and
unstructured scenes. This is critical for robotic sorting and handling of
legacy nuclear waste, which poses complex environmental remediation challenges
in many nuclearised nations.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2411.03082v1)

---


## ERUP\-YOLO: Enhancing Object Detection Robustness for Adverse Weather Condition by Unified Image\-Adaptive Processing

**发布日期**：2024-11-05

**作者**：Yuka Ogino

**摘要**：We propose an image\-adaptive object detection method for adverse weather
conditions such as fog and low\-light. Our framework employs differentiable
preprocessing filters to perform image enhancement suitable for later\-stage
object detections. Our framework introduces two differentiable filters: a
B\\'ezier curve\-based pixel\-wise \(BPW\) filter and a kernel\-based local \(KBL\)
filter. These filters unify the functions of classical image processing filters
and improve performance of object detection. We also propose a domain\-agnostic
data augmentation strategy using the BPW filter. Our method does not require
data\-specific customization of the filter combinations, parameter ranges, and
data augmentation. We evaluate our proposed approach, called Enhanced
Robustness by Unified Image Processing \(ERUP\)\-YOLO, by applying it to the
YOLOv3 detector. Experiments on adverse weather datasets demonstrate that our
proposed filters match or exceed the expressiveness of conventional methods and
our ERUP\-YOLO achieved superior performance in a wide range of adverse weather
conditions, including fog and low\-light conditions.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2411.02799v1)

---


## One\-Stage\-TFS: Thai One\-Stage Fingerspelling Dataset for Fingerspelling Recognition Frameworks

**发布日期**：2024-11-05

**作者**：Siriwiwat Lata

**摘要**：The Thai One\-Stage Fingerspelling \(One\-Stage\-TFS\) dataset is a comprehensive
resource designed to advance research in hand gesture recognition, explicitly
focusing on the recognition of Thai sign language. This dataset comprises 7,200
images capturing 15 one\-stage consonant gestures performed by undergraduate
students from Rajabhat Maha Sarakham University, Thailand. The contributors
include both expert students from the Special Education Department with
proficiency in Thai sign language and students from other departments without
prior sign language experience. Images were collected between July and December
2021 using a DSLR camera, with contributors demonstrating hand gestures against
both simple and complex backgrounds. The One\-Stage\-TFS dataset presents
challenges in detecting and recognizing hand gestures, offering opportunities
to develop novel end\-to\-end recognition frameworks. Researchers can utilize
this dataset to explore deep learning methods, such as YOLO, EfficientDet,
RetinaNet, and Detectron, for hand detection, followed by feature extraction
and recognition using techniques like convolutional neural networks,
transformers, and adaptive feature fusion networks. The dataset is accessible
via the Mendeley Data repository and supports a wide range of applications in
computer science, including deep learning, computer vision, and pattern
recognition, thereby encouraging further innovation and exploration in these
fields.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2411.02768v1)

---


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

