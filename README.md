# 每日从arXiv中获取最新YOLO相关论文


## Few\-shot target\-driven instance detection based on open\-vocabulary object detection models

**发布日期**：2024-10-21

**作者**：Ben Crulis

**摘要**：Current large open vision models could be useful for one and few\-shot object
recognition. Nevertheless, gradient\-based re\-training solutions are costly. On
the other hand, open\-vocabulary object detection models bring closer visual and
textual concepts in the same latent space, allowing zero\-shot detection via
prompting at small computational cost. We propose a lightweight method to turn
the latter into a one\-shot or few\-shot object recognition models without
requiring textual descriptions. Our experiments on the TEgO dataset using the
YOLO\-World model as a base show that performance increases with the model size,
the number of examples and the use of image augmentation.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2410.16028v1)

---


## How Important are Data Augmentations to Close the Domain Gap for Object Detection in Orbit?

**发布日期**：2024-10-21

**作者**：Maximilian Ulmer

**摘要**：We investigate the efficacy of data augmentations to close the domain gap in
spaceborne computer vision, crucial for autonomous operations like on\-orbit
servicing. As the use of computer vision in space increases, challenges such as
hostile illumination and low signal\-to\-noise ratios significantly hinder
performance. While learning\-based algorithms show promising results, their
adoption is limited by the need for extensive annotated training data and the
domain gap that arises from differences between synthesized and real\-world
imagery. This study explores domain generalization in terms of data
augmentations \-\- classical color and geometric transformations, corruptions,
and noise \-\- to enhance model performance across the domain gap. To this end,
we conduct an large scale experiment using a hyperparameter optimization
pipeline that samples hundreds of different configurations and searches for the
best set to bridge the domain gap. As a reference task, we use 2D object
detection and evaluate on the SPEED\+ dataset that contains real
hardware\-in\-the\-loop satellite images in its test set. Moreover, we evaluate
four popular object detectors, including Mask R\-CNN, Faster R\-CNN, YOLO\-v7, and
the open set detector GroundingDINO, and highlight their trade\-offs between
performance, inference speed, and training time. Our results underscore the
vital role of data augmentations in bridging the domain gap, improving model
performance, robustness, and reliability for critical space applications. As a
result, we propose two novel data augmentations specifically developed to
emulate the visual effects observed in orbital imagery. We conclude by
recommending the most effective augmentations for advancing computer vision in
challenging orbital environments. Code for training detectors and
hyperparameter search will be made publicly available.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2410.15766v1)

---


## Deep Learning and Machine Learning \-\- Object Detection and Semantic Segmentation: From Theory to Applications

**发布日期**：2024-10-21

**作者**：Jintao Ren

**摘要**：This book offers an in\-depth exploration of object detection and semantic
segmentation, combining theoretical foundations with practical applications. It
covers state\-of\-the\-art advancements in machine learning and deep learning,
with a focus on convolutional neural networks \(CNNs\), YOLO architectures, and
transformer\-based approaches like DETR. The book also delves into the
integration of artificial intelligence \(AI\) techniques and large language
models for enhanced object detection in complex environments. A thorough
discussion of big data analysis is presented, highlighting the importance of
data processing, model optimization, and performance evaluation metrics. By
bridging the gap between traditional methods and modern deep learning
frameworks, this book serves as a comprehensive guide for researchers, data
scientists, and engineers aiming to leverage AI\-driven methodologies in
large\-scale object detection tasks.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2410.15584v1)

---


## YOLO\-RD: Introducing Relevant and Compact Explicit Knowledge to YOLO by Retriever\-Dictionary

**发布日期**：2024-10-20

**作者**：Hao\-Tang Tsui

**摘要**：Identifying and localizing objects within images is a fundamental challenge,
and numerous efforts have been made to enhance model accuracy by experimenting
with diverse architectures and refining training strategies. Nevertheless, a
prevalent limitation in existing models is overemphasizing the current input
while ignoring the information from the entire dataset. We introduce an
innovative \{\\em \\textbf\{R\}etriever\}\-\{\\em\\textbf\{D\}ictionary\} \(RD\) module to
address this issue. This architecture enables YOLO\-based models to efficiently
retrieve features from a Dictionary that contains the insight of the dataset,
which is built by the knowledge from Visual Models \(VM\), Large Language Models
\(LLM\), or Visual Language Models \(VLM\). The flexible RD enables the model to
incorporate such explicit knowledge that enhances the ability to benefit
multiple tasks, specifically, segmentation, detection, and classification, from
pixel to image level. The experiments show that using the RD significantly
improves model performance, achieving more than a 3\\% increase in mean Average
Precision for object detection with less than a 1\\% increase in model
parameters. Beyond 1\-stage object detection models, the RD module improves the
effectiveness of 2\-stage models and DETR\-based architectures, such as Faster
R\-CNN and Deformable DETR


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2410.15346v1)

---


## Cutting\-Edge Detection of Fatigue in Drivers: A Comparative Study of Object Detection Models

**发布日期**：2024-10-19

**作者**：Amelia Jones

**摘要**：This research delves into the development of a fatigue detection system based
on modern object detection algorithms, particularly YOLO \(You Only Look Once\)
models, including YOLOv5, YOLOv6, YOLOv7, and YOLOv8. By comparing the
performance of these models, we evaluate their effectiveness in real\-time
detection of fatigue\-related behavior in drivers. The study addresses
challenges like environmental variability and detection accuracy and suggests a
roadmap for enhancing real\-time detection. Experimental results demonstrate
that YOLOv8 offers superior performance, balancing accuracy with speed. Data
augmentation techniques and model optimization have been key in enhancing
system adaptability to various driving conditions.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2410.15030v1)

---


## You Only Look Twice\! for Failure Causes Identification of Drill Bits

**发布日期**：2024-10-18

**作者**：Asma Yamani

**摘要**：Efficient identification of the root causes of drill bit failure is crucial
due to potential impacts such as operational losses, safety threats, and
delays. Early recognition of these failures enables proactive maintenance,
reducing risks and financial losses associated with unforeseen breakdowns and
prolonged downtime. Thus, our study investigates various causes of drill bit
failure using images of different blades. The process involves annotating
cutters with their respective locations and damage types, followed by the
development of two YOLO Location and Damage Cutter Detection models, as well as
multi\-class multi\-label Decision Tree and Random Forests models to identify the
causes of failure by assessing the cutters' location and damage type.
Additionally, RRFCI is proposed for the classification of failure causes.
Notably, the cutter location detection model achieved a high score of 0.97 mPA,
and the cutter damage detection model yielded a 0.49 mPA. The rule\-based
approach over\-performed both DT and RF in failure cause identification,
achieving a macro\-average F1\-score of 0.94 across all damage causes. The
integration of the complete automated pipeline successfully identified 100\\% of
the 24 failure causes when tested on independent sets of ten drill bits,
showcasing its potential to efficiently assist experts in identifying the root
causes of drill bit damages.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2410.14282v1)

---


## Spatiotemporal Object Detection for Improved Aerial Vehicle Detection in Traffic Monitoring

**发布日期**：2024-10-17

**作者**：Kristina Telegraph

**摘要**：This work presents advancements in multi\-class vehicle detection using UAV
cameras through the development of spatiotemporal object detection models. The
study introduces a Spatio\-Temporal Vehicle Detection Dataset \(STVD\) containing
6, 600 annotated sequential frame images captured by UAVs, enabling
comprehensive training and evaluation of algorithms for holistic spatiotemporal
perception. A YOLO\-based object detection algorithm is enhanced to incorporate
temporal dynamics, resulting in improved performance over single frame models.
The integration of attention mechanisms into spatiotemporal models is shown to
further enhance performance. Experimental validation demonstrates significant
progress, with the best spatiotemporal model exhibiting a 16.22% improvement
over single frame models, while it is demonstrated that attention mechanisms
hold the potential for additional performance gains.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2410.13616v1)

---


## DocLayout\-YOLO: Enhancing Document Layout Analysis through Diverse Synthetic Data and Global\-to\-Local Adaptive Perception

**发布日期**：2024-10-16

**作者**：Zhiyuan Zhao

**摘要**：Document Layout Analysis is crucial for real\-world document understanding
systems, but it encounters a challenging trade\-off between speed and accuracy:
multimodal methods leveraging both text and visual features achieve higher
accuracy but suffer from significant latency, whereas unimodal methods relying
solely on visual features offer faster processing speeds at the expense of
accuracy. To address this dilemma, we introduce DocLayout\-YOLO, a novel
approach that enhances accuracy while maintaining speed advantages through
document\-specific optimizations in both pre\-training and model design. For
robust document pre\-training, we introduce the Mesh\-candidate BestFit
algorithm, which frames document synthesis as a two\-dimensional bin packing
problem, generating the large\-scale, diverse DocSynth\-300K dataset.
Pre\-training on the resulting DocSynth\-300K dataset significantly improves
fine\-tuning performance across various document types. In terms of model
optimization, we propose a Global\-to\-Local Controllable Receptive Module that
is capable of better handling multi\-scale variations of document elements.
Furthermore, to validate performance across different document types, we
introduce a complex and challenging benchmark named DocStructBench. Extensive
experiments on downstream datasets demonstrate that DocLayout\-YOLO excels in
both speed and accuracy. Code, data, and models are available at
https://github.com/opendatalab/DocLayout\-YOLO.


**代码链接**：https://github.com/opendatalab/DocLayout-YOLO.

**论文链接**：[阅读更多](http://arxiv.org/abs/2410.12628v1)

---


## Development of Image Collection Method Using YOLO and Siamese Network

**发布日期**：2024-10-16

**作者**：Chan Young Shin

**摘要**：As we enter the era of big data, collecting high\-quality data is very
important. However, collecting data by humans is not only very time\-consuming
but also expensive. Therefore, many scientists have devised various methods to
collect data using computers. Among them, there is a method called web
crawling, but the authors found that the crawling method has a problem in that
unintended data is collected along with the user. The authors found that this
can be filtered using the object recognition model YOLOv10. However, there are
cases where data that is not properly filtered remains. Here, image
reclassification was performed by additionally utilizing the distance output
from the Siamese network, and higher performance was recorded than other
classification models. \(average \\\_f1 score YOLO\+MobileNet
0.678\->YOLO\+SiameseNet 0.772\)\) The user can specify a distance threshold to
adjust the balance between data deficiency and noise\-robustness. The authors
also found that the Siamese network can achieve higher performance with fewer
resources because the cropped images are used for object recognition when
processing images in the Siamese network. \(Class 20 mean\-based f1 score,
non\-crop\+Siamese\(MobileNetV3\-Small\) 80.94 \-> crop
preprocessing\+Siamese\(MobileNetV3\-Small\) 82.31\) In this way, the image
retrieval system that utilizes two consecutive models to reduce errors can save
users' time and effort, and build better quality data faster and with fewer
resources than before.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2410.12561v1)

---


## YOLO\-ELA: Efficient Local Attention Modeling for High\-Performance Real\-Time Insulator Defect Detection

**发布日期**：2024-10-15

**作者**：Olalekan Akindele

**摘要**：Existing detection methods for insulator defect identification from unmanned
aerial vehicles \(UAV\) struggle with complex background scenes and small
objects, leading to suboptimal accuracy and a high number of false positives
detection. Using the concept of local attention modeling, this paper proposes a
new attention\-based foundation architecture, YOLO\-ELA, to address this issue.
The Efficient Local Attention \(ELA\) blocks were added into the neck part of the
one\-stage YOLOv8 architecture to shift the model's attention from background
features towards features of insulators with defects. The SCYLLA
Intersection\-Over\-Union \(SIoU\) criterion function was used to reduce detection
loss, accelerate model convergence, and increase the model's sensitivity
towards small insulator defects, yielding higher true positive outcomes. Due to
a limited dataset, data augmentation techniques were utilized to increase the
diversity of the dataset. In addition, we leveraged the transfer learning
strategy to improve the model's performance. Experimental results on
high\-resolution UAV images show that our method achieved a state\-of\-the\-art
performance of 96.9% mAP0.5 and a real\-time detection speed of 74.63 frames per
second, outperforming the baseline model. This further demonstrates the
effectiveness of attention\-based convolutional neural networks \(CNN\) in object
detection tasks.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2410.11727v1)

---

