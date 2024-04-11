# 每日从arXiv中获取最新YOLO相关论文


## YOLO based Ocean Eddy Localization with AWS SageMaker

**发布日期**：2024-04-10

**作者**：Seraj Al Mahmud Mostafa

**摘要**：Ocean eddies play a significant role both on the sea surface and beneath it,
contributing to the sustainability of marine life dependent on oceanic
behaviors. Therefore, it is crucial to investigate ocean eddies to monitor
changes in the Earth, particularly in the oceans, and their impact on climate.
This study aims to pinpoint ocean eddies using AWS cloud services, specifically
SageMaker. The primary objective is to detect small\-scale \(<20km\) ocean eddies
from satellite remote images and assess the feasibility of utilizing SageMaker,
which offers tools for deploying AI applications. Moreover, this research not
only explores the deployment of cloud\-based services for remote sensing of
Earth data but also evaluates several YOLO \(You Only Look Once\) models using
single and multi\-GPU\-based services in the cloud. Furthermore, this study
underscores the potential of these services, their limitations, challenges
related to deployment and resource management, and their user\-riendliness for
Earth science projects.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2404.06744v1)

---


## Towards Improved Semiconductor Defect Inspection for high\-NA EUVL based on SEMI\-SuperYOLO\-NAS

**发布日期**：2024-04-08

**作者**：Ying\-Lin Chen

**摘要**：Due to potential pitch reduction, the semiconductor industry is adopting
High\-NA EUVL technology. However, its low depth of focus presents challenges
for High Volume Manufacturing. To address this, suppliers are exploring thinner
photoresists and new underlayers/hardmasks. These may suffer from poor SNR,
complicating defect detection. Vision\-based ML algorithms offer a promising
solution for semiconductor defect inspection. However, developing a robust ML
model across various image resolutions without explicit training remains a
challenge for nano\-scale defect inspection. This research's goal is to propose
a scale\-invariant ADCD framework capable to upscale images, addressing this
issue. We propose an improvised ADCD framework as SEMI\-SuperYOLO\-NAS, which
builds upon the baseline YOLO\-NAS architecture. This framework integrates a SR
assisted branch to aid in learning HR features by the defect detection
backbone, particularly for detecting nano\-scale defect instances from LR
images. Additionally, the SR\-assisted branch can recursively generate upscaled
images from their corresponding downscaled counterparts, enabling defect
detection inference across various image resolutions without requiring explicit
training. Moreover, we investigate improved data augmentation strategy aimed at
generating diverse and realistic training datasets to enhance model
performance. We have evaluated our proposed approach using two original FAB
datasets obtained from two distinct processes and captured using two different
imaging tools. Finally, we demonstrate zero\-shot inference for our model on a
new, originating from a process condition distinct from the training dataset
and possessing different Pitch characteristics. Experimental validation
demonstrates that our proposed ADCD framework aids in increasing the throughput
of imaging tools for defect inspection by reducing the required image pixel
resolutions.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2404.05862v1)

---


## FlightScope: A Deep Comprehensive Assessment of Aircraft Detection Algorithms in Satellite Imagery

**发布日期**：2024-04-03

**作者**：Safouane El Ghazouali

**摘要**：Object detection in remotely sensed satellite pictures is fundamental in many
fields such as biophysical, and environmental monitoring. While deep learning
algorithms are constantly evolving, they have been mostly implemented and
tested on popular ground\-based taken photos. This paper critically evaluates
and compares a suite of advanced object detection algorithms customized for the
task of identifying aircraft within satellite imagery. Using the large
HRPlanesV2 dataset, together with a rigorous validation with the GDIT dataset,
this research encompasses an array of methodologies including YOLO versions 5
and 8, Faster RCNN, CenterNet, RetinaNet, RTMDet, and DETR, all trained from
scratch. This exhaustive training and validation study reveal YOLOv5 as the
preeminent model for the specific case of identifying airplanes from remote
sensing data, showcasing high precision and adaptability across diverse imaging
conditions. This research highlight the nuanced performance landscapes of these
algorithms, with YOLOv5 emerging as a robust solution for aerial object
detection, underlining its importance through superior mean average precision,
Recall, and Intersection over Union scores. The findings described here
underscore the fundamental role of algorithm selection aligned with the
specific demands of satellite imagery analysis and extend a comprehensive
framework to evaluate model efficacy. The benchmark toolkit and codes,
available via https://github.com/toelt\-llc/FlightScope\_Bench, aims to further
exploration and innovation in the realm of remote sensing object detection,
paving the way for improved analytical methodologies in satellite imagery
applications.


**代码链接**：https://github.com/toelt-llc/FlightScope_Bench,

**论文链接**：[阅读更多](http://arxiv.org/abs/2404.02877v1)

---


## Leveraging YOLO\-World and GPT\-4V LMMs for Zero\-Shot Person Detection and Action Recognition in Drone Imagery

**发布日期**：2024-04-02

**作者**：Christian Limberg

**摘要**：In this article, we explore the potential of zero\-shot Large Multimodal
Models \(LMMs\) in the domain of drone perception. We focus on person detection
and action recognition tasks and evaluate two prominent LMMs, namely YOLO\-World
and GPT\-4V\(ision\) using a publicly available dataset captured from aerial
views. Traditional deep learning approaches rely heavily on large and
high\-quality training datasets. However, in certain robotic settings, acquiring
such datasets can be resource\-intensive or impractical within a reasonable
timeframe. The flexibility of prompt\-based Large Multimodal Models \(LMMs\) and
their exceptional generalization capabilities have the potential to
revolutionize robotics applications in these scenarios. Our findings suggest
that YOLO\-World demonstrates good detection performance. GPT\-4V struggles with
accurately classifying action classes but delivers promising results in
filtering out unwanted region proposals and in providing a general description
of the scenery. This research represents an initial step in leveraging LMMs for
drone perception and establishes a foundation for future investigations in this
area.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2404.01571v1)

---


## YOLOOC: YOLO\-based Open\-Class Incremental Object Detection with Novel Class Discovery

**发布日期**：2024-03-30

**作者**：Qian Wan

**摘要**：Because of its use in practice, open\-world object detection \(OWOD\) has gotten
a lot of attention recently. The challenge is how can a model detect novel
classes and then incrementally learn them without forgetting previously known
classes. Previous approaches hinge on strongly\-supervised or weakly\-supervised
novel\-class data for novel\-class detection, which may not apply to real
applications. We construct a new benchmark that novel classes are only
encountered at the inference stage. And we propose a new OWOD detector YOLOOC,
based on the YOLO architecture yet for the Open\-Class setup. We introduce label
smoothing to prevent the detector from over\-confidently mapping novel classes
to known classes and to discover novel classes. Extensive experiments conducted
on our more realistic setup demonstrate the effectiveness of our method for
discovering novel classes in our new benchmark.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2404.00257v1)

---


## SMOF: Streaming Modern CNNs on FPGAs with Smart Off\-Chip Eviction

**发布日期**：2024-03-27

**作者**：Petros Toupas

**摘要**：Convolutional Neural Networks \(CNNs\) have demonstrated their effectiveness in
numerous vision tasks. However, their high processing requirements necessitate
efficient hardware acceleration to meet the application's performance targets.
In the space of FPGAs, streaming\-based dataflow architectures are often adopted
by users, as significant performance gains can be achieved through layer\-wise
pipelining and reduced off\-chip memory access by retaining data on\-chip.
However, modern topologies, such as the UNet, YOLO, and X3D models, utilise
long skip connections, requiring significant on\-chip storage and thus limiting
the performance achieved by such system architectures. The paper addresses the
above limitation by introducing weight and activation eviction mechanisms to
off\-chip memory along the computational pipeline, taking into account the
available compute and memory resources. The proposed mechanism is incorporated
into an existing toolflow, expanding the design space by utilising off\-chip
memory as a buffer. This enables the mapping of such modern CNNs to devices
with limited on\-chip memory, under the streaming architecture design approach.
SMOF has demonstrated the capacity to deliver competitive and, in some cases,
state\-of\-the\-art performance across a spectrum of computer vision tasks,
achieving up to 10.65 X throughput improvement compared to previous works.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2403.18921v1)

---


## State of the art applications of deep learning within tracking and detecting marine debris: A survey

**发布日期**：2024-03-26

**作者**：Zoe Moorton

**摘要**：Deep learning techniques have been explored within the marine litter problem
for approximately 20 years but the majority of the research has developed
rapidly in the last five years. We provide an in\-depth, up to date, summary and
analysis of 28 of the most recent and significant contributions of deep
learning in marine debris. From cross referencing the research paper results,
the YOLO family significantly outperforms all other methods of object detection
but there are many respected contributions to this field that have
categorically agreed that a comprehensive database of underwater debris is not
currently available for machine learning. Using a small dataset curated and
labelled by us, we tested YOLOv5 on a binary classification task and found the
accuracy was low and the rate of false positives was high; highlighting the
importance of a comprehensive database. We conclude this survey with over 40
future research recommendations and open challenges.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2403.18067v1)

---


## VisionGPT: LLM\-Assisted Real\-Time Anomaly Detection for Safe Visual Navigation

**发布日期**：2024-03-19

**作者**：Hao Wang

**摘要**：This paper explores the potential of Large Language Models\(LLMs\) in zero\-shot
anomaly detection for safe visual navigation. With the assistance of the
state\-of\-the\-art real\-time open\-world object detection model Yolo\-World and
specialized prompts, the proposed framework can identify anomalies within
camera\-captured frames that include any possible obstacles, then generate
concise, audio\-delivered descriptions emphasizing abnormalities, assist in safe
visual navigation in complex circumstances. Moreover, our proposed framework
leverages the advantages of LLMs and the open\-vocabulary object detection model
to achieve the dynamic scenario switch, which allows users to transition
smoothly from scene to scene, which addresses the limitation of traditional
visual navigation. Furthermore, this paper explored the performance
contribution of different prompt components, provided the vision for future
improvement in visual accessibility, and paved the way for LLMs in video
anomaly detection and vision\-language understanding.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2403.12415v1)

---


## YOLOv9 for Fracture Detection in Pediatric Wrist Trauma X\-ray Images

**发布日期**：2024-03-17

**作者**：Chun\-Tse Chien

**摘要**：The introduction of YOLOv9, the latest version of the You Only Look Once
\(YOLO\) series, has led to its widespread adoption across various scenarios.
This paper is the first to apply the YOLOv9 algorithm model to the fracture
detection task as computer\-assisted diagnosis \(CAD\) to help radiologists and
surgeons to interpret X\-ray images. Specifically, this paper trained the model
on the GRAZPEDWRI\-DX dataset and extended the training set using data
augmentation techniques to improve the model performance. Experimental results
demonstrate that compared to the mAP 50\-95 of the current state\-of\-the\-art
\(SOTA\) model, the YOLOv9 model increased the value from 42.16% to 43.73%, with
an improvement of 3.7%. The implementation code is publicly available at
https://github.com/RuiyangJu/YOLOv9\-Fracture\-Detection.


**代码链接**：https://github.com/RuiyangJu/YOLOv9-Fracture-Detection.

**论文链接**：[阅读更多](http://arxiv.org/abs/2403.11249v1)

---


## Intelligent Railroad Grade Crossing: Leveraging Semantic Segmentation and Object Detection for Enhanced Safety

**发布日期**：2024-03-17

**作者**：Al Amin

**摘要**：Crashes and delays at Railroad Highway Grade Crossings \(RHGC\), where highways
and railroads intersect, pose significant safety concerns for the U.S. Federal
Railroad Administration \(FRA\). Despite the critical importance of addressing
accidents and traffic delays at highway\-railroad intersections, there is a
notable dearth of research on practical solutions for managing these issues. In
response to this gap in the literature, our study introduces an intelligent
system that leverages machine learning and computer vision techniques to
enhance safety at Railroad Highway Grade crossings \(RHGC\). This research
proposed a Non\-Maximum Suppression \(NMS\)\- based ensemble model that integrates
a variety of YOLO variants, specifically YOLOv5S, YOLOv5M, and YOLOv5L, for
grade\-crossing object detection, utilizes segmentation techniques from the UNet
architecture for detecting approaching rail at a grade crossing. Both methods
are implemented on a Raspberry Pi. Moreover, the strategy employs
high\-definition cameras installed at the RHGC. This framework enables the
system to monitor objects within the Region of Interest \(ROI\) at crossings,
detect the approach of trains, and clear the crossing area before a train
arrives. Regarding accuracy, precision, recall, and Intersection over Union
\(IoU\), the proposed state\-of\-the\-art NMS\-based object detection ensemble model
achieved 96% precision. In addition, the UNet segmentation model obtained a 98%
IoU value. This automated railroad grade crossing system powered by artificial
intelligence represents a promising solution for enhancing safety at
highway\-railroad intersections.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2403.11060v1)

---

