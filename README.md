# 每日从arXiv中获取最新YOLO相关论文


## YOLO\-Vehicle\-Pro: A Cloud\-Edge Collaborative Framework for Object Detection in Autonomous Driving under Adverse Weather Conditions

**发布日期**：2024-10-23

**作者**：Xiguang Li

**摘要**：With the rapid advancement of autonomous driving technology, efficient and
accurate object detection capabilities have become crucial factors in ensuring
the safety and reliability of autonomous driving systems. However, in
low\-visibility environments such as hazy conditions, the performance of
traditional object detection algorithms often degrades significantly, failing
to meet the demands of autonomous driving. To address this challenge, this
paper proposes two innovative deep learning models: YOLO\-Vehicle and
YOLO\-Vehicle\-Pro. YOLO\-Vehicle is an object detection model tailored
specifically for autonomous driving scenarios, employing multimodal fusion
techniques to combine image and textual information for object detection.
YOLO\-Vehicle\-Pro builds upon this foundation by introducing an improved image
dehazing algorithm, enhancing detection performance in low\-visibility
environments. In addition to model innovation, this paper also designs and
implements a cloud\-edge collaborative object detection system, deploying models
on edge devices and offloading partial computational tasks to the cloud in
complex situations. Experimental results demonstrate that on the KITTI dataset,
the YOLO\-Vehicle\-v1s model achieved 92.1% accuracy while maintaining a
detection speed of 226 FPS and an inference time of 12ms, meeting the real\-time
requirements of autonomous driving. When processing hazy images, the
YOLO\-Vehicle\-Pro model achieved a high accuracy of 82.3% mAP@50 on the Foggy
Cityscapes dataset while maintaining a detection speed of 43 FPS.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2410.17734v1)

---


## YOLOv11: An Overview of the Key Architectural Enhancements

**发布日期**：2024-10-23

**作者**：Rahima Khanam

**摘要**：This study presents an architectural analysis of YOLOv11, the latest
iteration in the YOLO \(You Only Look Once\) series of object detection models.
We examine the models architectural innovations, including the introduction of
the C3k2 \(Cross Stage Partial with kernel size 2\) block, SPPF \(Spatial Pyramid
Pooling \- Fast\), and C2PSA \(Convolutional block with Parallel Spatial
Attention\) components, which contribute in improving the models performance in
several ways such as enhanced feature extraction. The paper explores YOLOv11's
expanded capabilities across various computer vision tasks, including object
detection, instance segmentation, pose estimation, and oriented object
detection \(OBB\). We review the model's performance improvements in terms of
mean Average Precision \(mAP\) and computational efficiency compared to its
predecessors, with a focus on the trade\-off between parameter count and
accuracy. Additionally, the study discusses YOLOv11's versatility across
different model sizes, from nano to extra\-large, catering to diverse
application needs from edge devices to high\-performance computing environments.
Our research provides insights into YOLOv11's position within the broader
landscape of object detection and its potential impact on real\-time computer
vision applications.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2410.17725v1)

---


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

