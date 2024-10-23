# 每日从arXiv中获取最新YOLO相关论文


## YOLO\-TS: Real\-Time Traffic Sign Detection with Enhanced Accuracy Using Optimized Receptive Fields and Anchor\-Free Fusion

**发布日期**：2024-10-22

**作者**：Junzhou Chen

**摘要**：Ensuring safety in both autonomous driving and advanced driver\-assistance
systems \(ADAS\) depends critically on the efficient deployment of traffic sign
recognition technology. While current methods show effectiveness, they often
compromise between speed and accuracy. To address this issue, we present a
novel real\-time and efficient road sign detection network, YOLO\-TS. This
network significantly improves performance by optimizing the receptive fields
of multi\-scale feature maps to align more closely with the size distribution of
traffic signs in various datasets. Moreover, our innovative feature\-fusion
strategy, leveraging the flexibility of Anchor\-Free methods, allows for
multi\-scale object detection on a high\-resolution feature map abundant in
contextual information, achieving remarkable enhancements in both accuracy and
speed. To mitigate the adverse effects of the grid pattern caused by dilated
convolutions on the detection of smaller objects, we have devised a unique
module that not only mitigates this grid effect but also widens the receptive
field to encompass an extensive range of spatial contextual information, thus
boosting the efficiency of information usage. Evaluation on challenging public
datasets, TT100K and CCTSDB2021, demonstrates that YOLO\-TS surpasses existing
state\-of\-the\-art methods in terms of both accuracy and speed. The code for our
method will be available.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2410.17144v1)

---


## Multi Kernel Estimation based Object Segmentation

**发布日期**：2024-10-22

**作者**：Haim Goldfisher

**摘要**：This paper presents a novel approach for multi\-kernel estimation by enhancing
the KernelGAN algorithm, which traditionally estimates a single kernel for the
entire image. We introduce Multi\-KernelGAN, which extends KernelGAN's
capabilities by estimating two distinct kernels based on object segmentation
masks. Our approach is validated through three distinct methods: texture\-based
patch Fast Fourier Transform \(FFT\) calculation, detail\-based segmentation, and
deep learning\-based object segmentation using YOLOv8 and the Segment Anything
Model \(SAM\). Among these methods, the combination of YOLO and SAM yields the
best results for kernel estimation. Experimental results demonstrate that our
multi\-kernel estimation technique outperforms conventional single\-kernel
methods in super\-resolution tasks.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2410.17064v1)

---


## DSORT\-MCU: Detecting Small Objects in Real\-Time on Microcontroller Units

**发布日期**：2024-10-22

**作者**：Liam Boyle

**摘要**：Advances in lightweight neural networks have revolutionized computer vision
in a broad range of IoT applications, encompassing remote monitoring and
process automation. However, the detection of small objects, which is crucial
for many of these applications, remains an underexplored area in current
computer vision research, particularly for low\-power embedded devices that host
resource\-constrained processors. To address said gap, this paper proposes an
adaptive tiling method for lightweight and energy\-efficient object detection
networks, including YOLO\-based models and the popular FOMO network. The
proposed tiling enables object detection on low\-power MCUs with no compromise
on accuracy compared to large\-scale detection models. The benefit of the
proposed method is demonstrated by applying it to FOMO and TinyissimoYOLO
networks on a novel RISC\-V\-based MCU with built\-in ML accelerators. Extensive
experimental results show that the proposed tiling method boosts the F1\-score
by up to 225% for both FOMO and TinyissimoYOLO networks while reducing the
average object count error by up to 76% with FOMO and up to 89% for
TinyissimoYOLO. Furthermore, the findings of this work indicate that using a
soft F1 loss over the popular binary cross\-entropy loss can serve as an
implicit non\-maximum suppression for the FOMO network. To evaluate the
real\-world performance, the networks are deployed on the RISC\-V based GAP9
microcontroller from GreenWaves Technologies, showcasing the proposed method's
ability to strike a balance between detection performance \($58% \- 95%$ F1
score\), low latency \(0.6 ms/Inference \- 16.2 ms/Inference\}\), and energy
efficiency \(31 uJ/Inference\} \- 1.27 mJ/Inference\) while performing multiple
predictions using high\-resolution images on a MCU.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2410.16769v1)

---


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

