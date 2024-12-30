# 每日从arXiv中获取最新YOLO相关论文


## Optimizing Helmet Detection with Hybrid YOLO Pipelines: A Detailed Analysis

**发布日期**：2024-12-27

**作者**：Vaikunth M

**摘要**：Helmet detection is crucial for advancing protection levels in public road
traffic dynamics. This problem statement translates to an object detection
task. Therefore, this paper compares recent You Only Look Once \(YOLO\) models in
the context of helmet detection in terms of reliability and computational load.
Specifically, YOLOv8, YOLOv9, and the newly released YOLOv11 have been used.
Besides, a modified architectural pipeline that remarkably improves the overall
performance has been proposed in this manuscript. This hybridized YOLO model
\(h\-YOLO\) has been pitted against the independent models for analysis that
proves h\-YOLO is preferable for helmet detection over plain YOLO models. The
models were tested using a range of standard object detection benchmarks such
as recall, precision, and mAP \(Mean Average Precision\). In addition, training
and testing times were recorded to provide the overall scope of the models in a
real\-time detection scenario.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2412.19467v1)

---


## Efficient Detection Framework Adaptation for Edge Computing: A Plug\-and\-play Neural Network Toolbox Enabling Edge Deployment

**发布日期**：2024-12-24

**作者**：Jiaqi Wu

**摘要**：Edge computing has emerged as a key paradigm for deploying deep
learning\-based object detection in time\-sensitive scenarios. However, existing
edge detection methods face challenges: 1\) difficulty balancing detection
precision with lightweight models, 2\) limited adaptability of generalized
deployment designs, and 3\) insufficient real\-world validation. To address these
issues, we propose the Edge Detection Toolbox \(ED\-TOOLBOX\), which utilizes
generalizable plug\-and\-play components to adapt object detection models for
edge environments. Specifically, we introduce a lightweight Reparameterized
Dynamic Convolutional Network \(Rep\-DConvNet\) featuring weighted multi\-shape
convolutional branches to enhance detection performance. Additionally, we
design a Sparse Cross\-Attention \(SC\-A\) network with a
localized\-mapping\-assisted self\-attention mechanism, enabling a well\-crafted
joint module for adaptive feature transfer. For real\-world applications, we
incorporate an Efficient Head into the YOLO framework to accelerate edge model
optimization. To demonstrate practical impact, we identify a gap in helmet
detection \-\- overlooking band fastening, a critical safety factor \-\- and create
the Helmet Band Detection Dataset \(HBDD\). Using ED\-TOOLBOX\-optimized models, we
address this real\-world task. Extensive experiments validate the effectiveness
of ED\-TOOLBOX, with edge detection models outperforming six state\-of\-the\-art
methods in visual surveillance simulations, achieving real\-time and accurate
performance. These results highlight ED\-TOOLBOX as a superior solution for edge
object detection.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2412.18230v1)

---


## Detecting and Classifying Defective Products in Images Using YOLO

**发布日期**：2024-12-22

**作者**：Zhen Qi

**摘要**：With the continuous advancement of industrial automation, product quality
inspection has become increasingly important in the manufacturing process.
Traditional inspection methods, which often rely on manual checks or simple
machine vision techniques, suffer from low efficiency and insufficient
accuracy. In recent years, deep learning technology, especially the YOLO \(You
Only Look Once\) algorithm, has emerged as a prominent solution in the field of
product defect detection due to its efficient real\-time detection capabilities
and excellent classification performance. This study aims to use the YOLO
algorithm to detect and classify defects in product images. By constructing and
training a YOLO model, we conducted experiments on multiple industrial product
datasets. The results demonstrate that this method can achieve real\-time
detection while maintaining high detection accuracy, significantly improving
the efficiency and accuracy of product quality inspection. This paper further
analyzes the advantages and limitations of the YOLO algorithm in practical
applications and explores future research directions.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2412.16935v1)

---


## Texture\- and Shape\-based Adversarial Attacks for Vehicle Detection in Synthetic Overhead Imagery

**发布日期**：2024-12-20

**作者**：Mikael Yeghiazaryan

**摘要**：Detecting vehicles in aerial images can be very challenging due to complex
backgrounds, small resolution, shadows, and occlusions. Despite the
effectiveness of SOTA detectors such as YOLO, they remain vulnerable to
adversarial attacks \(AAs\), compromising their reliability. Traditional AA
strategies often overlook the practical constraints of physical implementation,
focusing solely on attack performance. Our work addresses this issue by
proposing practical implementation constraints for AA in texture and/or shape.
These constraints include pixelation, masking, limiting the color palette of
the textures, and constraining the shape modifications. We evaluated the
proposed constraints through extensive experiments using three widely used
object detector architectures, and compared them to previous works. The results
demonstrate the effectiveness of our solutions and reveal a trade\-off between
practicality and performance. Additionally, we introduce a labeled dataset of
overhead images featuring vehicles of various categories. We will make the
code/dataset public upon paper acceptance.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2412.16358v1)

---


## A Light\-Weight Framework for Open\-Set Object Detection with Decoupled Feature Alignment in Joint Space

**发布日期**：2024-12-19

**作者**：Yonghao He

**摘要**：Open\-set object detection \(OSOD\) is highly desirable for robotic manipulation
in unstructured environments. However, existing OSOD methods often fail to meet
the requirements of robotic applications due to their high computational burden
and complex deployment. To address this issue, this paper proposes a
light\-weight framework called Decoupled OSOD \(DOSOD\), which is a practical and
highly efficient solution to support real\-time OSOD tasks in robotic systems.
Specifically, DOSOD builds upon the YOLO\-World pipeline by integrating a
vision\-language model \(VLM\) with a detector. A Multilayer Perceptron \(MLP\)
adaptor is developed to transform text embeddings extracted by the VLM into a
joint space, within which the detector learns the region representations of
class\-agnostic proposals. Cross\-modality features are directly aligned in the
joint space, avoiding the complex feature interactions and thereby improving
computational efficiency. DOSOD operates like a traditional closed\-set detector
during the testing phase, effectively bridging the gap between closed\-set and
open\-set detection. Compared to the baseline YOLO\-World, the proposed DOSOD
significantly enhances real\-time performance while maintaining comparable
accuracy. The slight DOSOD\-S model achieves a Fixed AP of $26.7\\%$, compared to
$26.2\\%$ for YOLO\-World\-v1\-S and $22.7\\%$ for YOLO\-World\-v2\-S, using similar
backbones on the LVIS minival dataset. Meanwhile, the FPS of DOSOD\-S is
$57.1\\%$ higher than YOLO\-World\-v1\-S and $29.6\\%$ higher than YOLO\-World\-v2\-S.
Meanwhile, we demonstrate that the DOSOD model facilitates the deployment of
edge devices. The codes and models are publicly available at
https://github.com/D\-Robotics\-AI\-Lab/DOSOD.


**代码链接**：https://github.com/D-Robotics-AI-Lab/DOSOD.

**论文链接**：[阅读更多](http://arxiv.org/abs/2412.14680v2)

---


## Physics\-Based Adversarial Attack on Near\-Infrared Human Detector for Nighttime Surveillance Camera Systems

**发布日期**：2024-12-18

**作者**：Muyao Niu

**摘要**：Many surveillance cameras switch between daytime and nighttime modes based on
illuminance levels. During the day, the camera records ordinary RGB images
through an enabled IR\-cut filter. At night, the filter is disabled to capture
near\-infrared \(NIR\) light emitted from NIR LEDs typically mounted around the
lens. While RGB\-based AI algorithm vulnerabilities have been widely reported,
the vulnerabilities of NIR\-based AI have rarely been investigated. In this
paper, we identify fundamental vulnerabilities in NIR\-based image understanding
caused by color and texture loss due to the intrinsic characteristics of
clothes' reflectance and cameras' spectral sensitivity in the NIR range. We
further show that the nearly co\-located configuration of illuminants and
cameras in existing surveillance systems facilitates concealing and fully
passive attacks in the physical world. Specifically, we demonstrate how
retro\-reflective and insulation plastic tapes can manipulate the intensity
distribution of NIR images. We showcase an attack on the YOLO\-based human
detector using binary patterns designed in the digital space \(via black\-box
query and searching\) and then physically realized using tapes pasted onto
clothes. Our attack highlights significant reliability concerns for nighttime
surveillance systems, which are intended to enhance security. Codes Available:
https://github.com/MyNiuuu/AdvNIR


**代码链接**：https://github.com/MyNiuuu/AdvNIR

**论文链接**：[阅读更多](http://arxiv.org/abs/2412.13709v1)

---


## Training a Distributed Acoustic Sensing Traffic Monitoring Network With Video Inputs

**发布日期**：2024-12-17

**作者**：Khen Cohen

**摘要**：Distributed Acoustic Sensing \(DAS\) has emerged as a promising tool for
real\-time traffic monitoring in densely populated areas. In this paper, we
present a novel concept that integrates DAS data with co\-located visual
information. We use YOLO\-derived vehicle location and classification from
camera inputs as labeled data to train a detection and classification neural
network utilizing DAS data only. Our model achieves a performance exceeding 94%
for detection and classification, and about 1.2% false alarm rate. We
illustrate the model's application in monitoring traffic over a week, yielding
statistical insights that could benefit future smart city developments. Our
approach highlights the potential of combining fiber\-optic sensors with visual
information, focusing on practicality and scalability, protecting privacy, and
minimizing infrastructure costs. To encourage future research, we share our
dataset.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2412.12743v1)

---


## Domain Generalization in Autonomous Driving: Evaluating YOLOv8s, RT\-DETR, and YOLO\-NAS with the ROAD\-Almaty Dataset

**发布日期**：2024-12-16

**作者**：Madiyar Alimov

**摘要**：This study investigates the domain generalization capabilities of three
state\-of\-the\-art object detection models \- YOLOv8s, RT\-DETR, and YOLO\-NAS \-
within the unique driving environment of Kazakhstan. Utilizing the newly
constructed ROAD\-Almaty dataset, which encompasses diverse weather, lighting,
and traffic conditions, we evaluated the models' performance without any
retraining. Quantitative analysis revealed that RT\-DETR achieved an average
F1\-score of 0.672 at IoU=0.5, outperforming YOLOv8s \(0.458\) and YOLO\-NAS
\(0.526\) by approximately 46% and 27%, respectively. Additionally, all models
exhibited significant performance declines at higher IoU thresholds \(e.g., a
drop of approximately 20% when increasing IoU from 0.5 to 0.75\) and under
challenging environmental conditions, such as heavy snowfall and low\-light
scenarios. These findings underscore the necessity for geographically diverse
training datasets and the implementation of specialized domain adaptation
techniques to enhance the reliability of autonomous vehicle detection systems
globally. This research contributes to the understanding of domain
generalization challenges in autonomous driving, particularly in
underrepresented regions.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2412.12349v1)

---


## Coconut Palm Tree Counting on Drone Images with Deep Object Detection and Synthetic Training Data

**发布日期**：2024-12-16

**作者**：Tobias Rohe

**摘要**：Drones have revolutionized various domains, including agriculture. Recent
advances in deep learning have propelled among other things object detection in
computer vision. This study utilized YOLO, a real\-time object detector, to
identify and count coconut palm trees in Ghanaian farm drone footage. The farm
presented has lost track of its trees due to different planting phases. While
manual counting would be very tedious and error\-prone, accurately determining
the number of trees is crucial for efficient planning and management of
agricultural processes, especially for optimizing yields and predicting
production. We assessed YOLO for palm detection within a semi\-automated
framework, evaluated accuracy augmentations, and pondered its potential for
farmers. Data was captured in September 2022 via drones. To optimize YOLO with
scarce data, synthetic images were created for model training and validation.
The YOLOv7 model, pretrained on the COCO dataset \(excluding coconut palms\), was
adapted using tailored data. Trees from footage were repositioned on synthetic
images, with testing on distinct authentic images. In our experiments, we
adjusted hyperparameters, improving YOLO's mean average precision \(mAP\). We
also tested various altitudes to determine the best drone height. From an
initial mAP@.5 of $0.65$, we achieved 0.88, highlighting the value of synthetic
images in agricultural scenarios.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2412.11949v1)

---


## CLDA\-YOLO: Visual Contrastive Learning Based Domain Adaptive YOLO Detector

**发布日期**：2024-12-16

**作者**：Tianheng Qiu

**摘要**：Unsupervised domain adaptive \(UDA\) algorithms can markedly enhance the
performance of object detectors under conditions of domain shifts, thereby
reducing the necessity for extensive labeling and retraining. Current domain
adaptive object detection algorithms primarily cater to two\-stage detectors,
which tend to offer minimal improvements when directly applied to single\-stage
detectors such as YOLO. Intending to benefit the YOLO detector from UDA, we
build a comprehensive domain adaptive architecture using a teacher\-student
cooperative system for the YOLO detector. In this process, we propose
uncertainty learning to cope with pseudo\-labeling generated by the teacher
model with extreme uncertainty and leverage dynamic data augmentation to
asymptotically adapt the teacher\-student system to the environment. To address
the inability of single\-stage object detectors to align at multiple stages, we
utilize a unified visual contrastive learning paradigm that aligns instance at
backbone and head respectively, which steadily improves the robustness of the
detectors in cross\-domain tasks. In summary, we present an unsupervised domain
adaptive YOLO detector based on visual contrastive learning \(CLDA\-YOLO\), which
achieves highly competitive results across multiple domain adaptive datasets
without any reduction in inference speed.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2412.11812v1)

---

