# 每日从arXiv中获取最新YOLO相关论文


## Gallbladder Cancer Detection in Ultrasound Images based on YOLO and Faster R\-CNN

**发布日期**：2024-04-23

**作者**：Sara Dadjouy

**摘要**：Medical image analysis is a significant application of artificial
intelligence for disease diagnosis. A crucial step in this process is the
identification of regions of interest within the images. This task can be
automated using object detection algorithms. YOLO and Faster R\-CNN are renowned
for such algorithms, each with its own strengths and weaknesses. This study
aims to explore the advantages of both techniques to select more accurate
bounding boxes for gallbladder detection from ultrasound images, thereby
enhancing gallbladder cancer classification. A fusion method that leverages the
benefits of both techniques is presented in this study. The proposed method
demonstrated superior classification performance, with an accuracy of 92.62%,
compared to the individual use of Faster R\-CNN and YOLOv8, which yielded
accuracies of 90.16% and 82.79%, respectively.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2404.15129v1)

---


## A Nasal Cytology Dataset for Object Detection and Deep Learning

**发布日期**：2024-04-21

**作者**：Mauro Camporeale

**摘要**：Nasal Cytology is a new and efficient clinical technique to diagnose rhinitis
and allergies that is not much widespread due to the time\-consuming nature of
cell counting; that is why AI\-aided counting could be a turning point for the
diffusion of this technique. In this article we present the first dataset of
rhino\-cytological field images: the NCD \(Nasal Cytology Dataset\), aimed to
train and deploy Object Detection models to support physicians and biologists
during clinical practice. The real distribution of the cytotypes, populating
the nasal mucosa has been replicated, sampling images from slides of clinical
patients, and manually annotating each cell found on them. The correspondent
object detection task presents non'trivial issues associated with the strong
class imbalancement, involving the rarest cell types. This work contributes to
some of open challenges by presenting a novel machine learning\-based approach
to aid the automated detection and classification of nasal mucosa cells: the
DETR and YOLO models shown good performance in detecting cells and classifying
them correctly, revealing great potential to accelerate the work of rhinology
experts.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2404.13745v1)

---


## BG\-YOLO: A Bidirectional\-Guided Method for Underwater Object Detection

**发布日期**：2024-04-13

**作者**：Jian Zhang

**摘要**：Degraded underwater images decrease the accuracy of underwater object
detection. However, existing methods for underwater image enhancement mainly
focus on improving the indicators in visual aspects, which may not benefit the
tasks of underwater image detection, and may lead to serious degradation in
performance. To alleviate this problem, we proposed a bidirectional\-guided
method for underwater object detection, referred to as BG\-YOLO. In the proposed
method, network is organized by constructing an enhancement branch and a
detection branch in a parallel way. The enhancement branch consists of a
cascade of an image enhancement subnet and an object detection subnet. And the
detection branch only consists of a detection subnet. A feature guided module
connects the shallow convolution layer of the two branches. When training the
enhancement branch, the object detection subnet in the enhancement branch
guides the image enhancement subnet to be optimized towards the direction that
is most conducive to the detection task. The shallow feature map of the trained
enhancement branch will be output to the feature guided module, constraining
the optimization of detection branch through consistency loss and prompting
detection branch to learn more detailed information of the objects. And hence
the detection performance will be refined. During the detection tasks, only
detection branch will be reserved so that no additional cost of computation
will be introduced. Extensive experiments demonstrate that the proposed method
shows significant improvement in performance of the detector in severely
degraded underwater scenes while maintaining a remarkable detection speed.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2404.08979v1)

---


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

**论文链接**：[阅读更多](http://arxiv.org/abs/2404.02877v2)

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

**论文链接**：[阅读更多](http://arxiv.org/abs/2404.00257v2)

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

