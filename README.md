# 每日从arXiv中获取最新YOLO相关论文


## FER\-YOLO\-Mamba: Facial Expression Detection and Classification Based on Selective State Space

**发布日期**：2024-05-03

**作者**：Hui Ma

**摘要**：Facial Expression Recognition \(FER\) plays a pivotal role in understanding
human emotional cues. However, traditional FER methods based on visual
information have some limitations, such as preprocessing, feature extraction,
and multi\-stage classification procedures. These not only increase
computational complexity but also require a significant amount of computing
resources. Considering Convolutional Neural Network \(CNN\)\-based FER schemes
frequently prove inadequate in identifying the deep, long\-distance dependencies
embedded within facial expression images, and the Transformer's inherent
quadratic computational complexity, this paper presents the FER\-YOLO\-Mamba
model, which integrates the principles of Mamba and YOLO technologies to
facilitate efficient coordination in facial expression image recognition and
localization. Within the FER\-YOLO\-Mamba model, we further devise a FER\-YOLO\-VSS
dual\-branch module, which combines the inherent strengths of convolutional
layers in local feature extraction with the exceptional capability of State
Space Models \(SSMs\) in revealing long\-distance dependencies. To the best of our
knowledge, this is the first Vision Mamba model designed for facial expression
detection and classification. To evaluate the performance of the proposed
FER\-YOLO\-Mamba model, we conducted experiments on two benchmark datasets,
RAF\-DB and SFEW. The experimental results indicate that the FER\-YOLO\-Mamba
model achieved better results compared to other models. The code is available
from https://github.com/SwjtuMa/FER\-YOLO\-Mamba.


**代码链接**：https://github.com/SwjtuMa/FER-YOLO-Mamba.

**论文链接**：[阅读更多](http://arxiv.org/abs/2405.01828v1)

---


## Active Learning Enabled Low\-cost Cell Image Segmentation Using Bounding Box Annotation

**发布日期**：2024-05-02

**作者**：Yu Zhu

**摘要**：Cell image segmentation is usually implemented using fully supervised deep
learning methods, which heavily rely on extensive annotated training data. Yet,
due to the complexity of cell morphology and the requirement for specialized
knowledge, pixel\-level annotation of cell images has become a highly
labor\-intensive task. To address the above problems, we propose an active
learning framework for cell segmentation using bounding box annotations, which
greatly reduces the data annotation cost of cell segmentation algorithms.
First, we generate a box\-supervised learning method \(denoted as YOLO\-SAM\) by
combining the YOLOv8 detector with the Segment Anything Model \(SAM\), which
effectively reduces the complexity of data annotation. Furthermore, it is
integrated into an active learning framework that employs the MC DropBlock
method to train the segmentation model with fewer box\-annotated samples.
Extensive experiments demonstrate that our model saves more than ninety percent
of data annotation time compared to mask\-supervised deep learning methods.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2405.01701v1)

---


## SOAR: Advancements in Small Body Object Detection for Aerial Imagery Using State Space Models and Programmable Gradients

**发布日期**：2024-05-02

**作者**：Tushar Verma

**摘要**：Small object detection in aerial imagery presents significant challenges in
computer vision due to the minimal data inherent in small\-sized objects and
their propensity to be obscured by larger objects and background noise.
Traditional methods using transformer\-based models often face limitations
stemming from the lack of specialized databases, which adversely affect their
performance with objects of varying orientations and scales. This underscores
the need for more adaptable, lightweight models. In response, this paper
introduces two innovative approaches that significantly enhance detection and
segmentation capabilities for small aerial objects. Firstly, we explore the use
of the SAHI framework on the newly introduced lightweight YOLO v9 architecture,
which utilizes Programmable Gradient Information \(PGI\) to reduce the
substantial information loss typically encountered in sequential feature
extraction processes. The paper employs the Vision Mamba model, which
incorporates position embeddings to facilitate precise location\-aware visual
understanding, combined with a novel bidirectional State Space Model \(SSM\) for
effective visual context modeling. This State Space Model adeptly harnesses the
linear complexity of CNNs and the global receptive field of Transformers,
making it particularly effective in remote sensing image classification. Our
experimental results demonstrate substantial improvements in detection accuracy
and processing efficiency, validating the applicability of these approaches for
real\-time small object detection across diverse aerial scenarios. This paper
also discusses how these methodologies could serve as foundational models for
future advancements in aerial object recognition technologies. The source code
will be made accessible here.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2405.01699v1)

---


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

