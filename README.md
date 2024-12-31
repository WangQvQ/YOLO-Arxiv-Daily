# 每日从arXiv中获取最新YOLO相关论文


## YOLO\-UniOW: Efficient Universal Open\-World Object Detection

**发布日期**：2024-12-30

**作者**：Lihao Liu

**摘要**：Traditional object detection models are constrained by the limitations of
closed\-set datasets, detecting only categories encountered during training.
While multimodal models have extended category recognition by aligning text and
image modalities, they introduce significant inference overhead due to
cross\-modality fusion and still remain restricted by predefined vocabulary,
leaving them ineffective at handling unknown objects in open\-world scenarios.
In this work, we introduce Universal Open\-World Object Detection \(Uni\-OWD\), a
new paradigm that unifies open\-vocabulary and open\-world object detection
tasks. To address the challenges of this setting, we propose YOLO\-UniOW, a
novel model that advances the boundaries of efficiency, versatility, and
performance. YOLO\-UniOW incorporates Adaptive Decision Learning to replace
computationally expensive cross\-modality fusion with lightweight alignment in
the CLIP latent space, achieving efficient detection without compromising
generalization. Additionally, we design a Wildcard Learning strategy that
detects out\-of\-distribution objects as "unknown" while enabling dynamic
vocabulary expansion without the need for incremental learning. This design
empowers YOLO\-UniOW to seamlessly adapt to new categories in open\-world
environments. Extensive experiments validate the superiority of YOLO\-UniOW,
achieving achieving 34.6 AP and 30.0 APr on LVIS with an inference speed of
69.6 FPS. The model also sets benchmarks on M\-OWODB, S\-OWODB, and nuScenes
datasets, showcasing its unmatched performance in open\-world object detection.
Code and models are available at https://github.com/THU\-MIG/YOLO\-UniOW.


**代码链接**：https://github.com/THU-MIG/YOLO-UniOW.

**论文链接**：[阅读更多](http://arxiv.org/abs/2412.20645v1)

---


## Differential Evolution Integrated Hybrid Deep Learning Model for Object Detection in Pre\-made Dishes

**发布日期**：2024-12-29

**作者**：Lujia Lv

**摘要**：With the continuous improvement of people's living standards and fast\-paced
working conditions, pre\-made dishes are becoming increasingly popular among
families and restaurants due to their advantages of time\-saving, convenience,
variety, cost\-effectiveness, standard quality, etc. Object detection is a key
technology for selecting ingredients and evaluating the quality of dishes in
the pre\-made dishes industry. To date, many object detection approaches have
been proposed. However, accurate object detection of pre\-made dishes is
extremely difficult because of overlapping occlusion of ingredients, similarity
of ingredients, and insufficient light in the processing environment. As a
result, the recognition scene is relatively complex and thus leads to poor
object detection by a single model. To address this issue, this paper proposes
a Differential Evolution Integrated Hybrid Deep Learning \(DEIHDL\) model. The
main idea of DEIHDL is three\-fold: 1\) three YOLO\-based and transformer\-based
base models are developed respectively to increase diversity for detecting
objects of pre\-made dishes, 2\) the three base models are integrated by
differential evolution optimized self\-adjusting weights, and 3\) weighted boxes
fusion strategy is employed to score the confidence of the three base models
during the integration. As such, DEIHDL possesses the multi\-performance
originating from the three base models to achieve accurate object detection in
complex pre\-made dish scenes. Extensive experiments on real datasets
demonstrate that the proposed DEIHDL model significantly outperforms the base
models in detecting objects of pre\-made dishes.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2412.20370v1)

---


## Plastic Waste Classification Using Deep Learning: Insights from the WaDaBa Dataset

**发布日期**：2024-12-28

**作者**：Suman Kunwar

**摘要**：With the increasing use of plastic, the challenges associated with managing
plastic waste have become more challenging, emphasizing the need of effective
solutions for classification and recycling. This study explores the potential
of deep learning, focusing on convolutional neural networks \(CNNs\) and object
detection models like YOLO \(You Only Look Once\), to tackle this issue using the
WaDaBa dataset. The study shows that YOLO\- 11m achieved highest accuracy
\(98.03%\) and mAP50 \(0.990\), with YOLO\-11n performing similarly but highest
mAP50\(0.992\). Lightweight models like YOLO\-10n trained faster but with lower
accuracy, whereas MobileNet V2 showed impressive performance \(97.12% accuracy\)
but fell short in object detection. Our study highlights the potential of deep
learning models in transforming how we classify plastic waste, with YOLO models
proving to be the most effective. By balancing accuracy and computational
efficiency, these models can help to create scalable, impactful solutions in
waste management and recycling.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2412.20232v1)

---


## YOLO\-MST: Multiscale deep learning method for infrared small target detection based on super\-resolution and YOLO

**发布日期**：2024-12-27

**作者**：Taoran Yue

**摘要**：With the advancement of aerospace technology and the increasing demands of
military applications, the development of low false\-alarm and high\-precision
infrared small target detection algorithms has emerged as a key focus of
research globally. However, the traditional model\-driven method is not robust
enough when dealing with features such as noise, target size, and contrast. The
existing deep\-learning methods have limited ability to extract and fuse key
features, and it is difficult to achieve high\-precision detection in complex
backgrounds and when target features are not obvious. To solve these problems,
this paper proposes a deep\-learning infrared small target detection method that
combines image super\-resolution technology with multi\-scale observation. First,
the input infrared images are preprocessed with super\-resolution and multiple
data enhancements are performed. Secondly, based on the YOLOv5 model, we
proposed a new deep\-learning network named YOLO\-MST. This network includes
replacing the SPPF module with the self\-designed MSFA module in the backbone,
optimizing the neck, and finally adding a multi\-scale dynamic detection head to
the prediction head. By dynamically fusing features from different scales, the
detection head can better adapt to complex scenes. The mAP@0.5 detection rates
of this method on two public datasets, SIRST and IRIS, reached 96.4% and 99.5%
respectively, more effectively solving the problems of missed detection, false
alarms, and low precision.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2412.19878v1)

---


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

