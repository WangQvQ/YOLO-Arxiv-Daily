# 每日从arXiv中获取最新YOLO相关论文


## CIB\-SE\-YOLOv8: Optimized YOLOv8 for Real\-Time Safety Equipment Detection on Construction Sites

**发布日期**：2024-10-28

**作者**：Xiaoyi Liu

**摘要**：Ensuring safety on construction sites is critical, with helmets playing a key
role in reducing injuries. Traditional safety checks are labor\-intensive and
often insufficient. This study presents a computer vision\-based solution using
YOLO for real\-time helmet detection, leveraging the SHEL5K dataset. Our
proposed CIB\-SE\-YOLOv8 model incorporates SE attention mechanisms and modified
C2f blocks, enhancing detection accuracy and efficiency. This model offers a
more effective solution for promoting safety compliance on construction sites.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2410.20699v1)

---


## DECADE: Towards Designing Efficient\-yet\-Accurate Distance Estimation Modules for Collision Avoidance in Mobile Advanced Driver Assistance Systems

**发布日期**：2024-10-25

**作者**：Muhammad Zaeem Shahzad

**摘要**：The proliferation of smartphones and other mobile devices provides a unique
opportunity to make Advanced Driver Assistance Systems \(ADAS\) accessible to
everyone in the form of an application empowered by low\-cost Machine/Deep
Learning \(ML/DL\) models to enhance road safety. For the critical feature of
Collision Avoidance in Mobile ADAS, lightweight Deep Neural Networks \(DNN\) for
object detection exist, but conventional pixel\-wise depth/distance estimation
DNNs are vastly more computationally expensive making them unsuitable for a
real\-time application on resource\-constrained devices. In this paper, we
present a distance estimation model, DECADE, that processes each detector
output instead of constructing pixel\-wise depth/disparity maps. In it, we
propose a pose estimation DNN to estimate allocentric orientation of detections
to supplement the distance estimation DNN in its prediction of distance using
bounding box features. We demonstrate that these modules can be attached to any
detector to extend object detection with fast distance estimation. Evaluation
of the proposed modules with attachment to and fine\-tuning on the outputs of
the YOLO object detector on the KITTI 3D Object Detection dataset achieves
state\-of\-the\-art performance with 1.38 meters in Mean Absolute Error and 7.3%
in Mean Relative Error in the distance range of 0\-150 meters. Our extensive
evaluation scheme not only evaluates class\-wise performance, but also evaluates
range\-wise accuracy especially in the critical range of 0\-70m.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2410.19336v1)

---


## Complexity Matters: Effective Dimensionality as a Measure for Adversarial Robustness

**发布日期**：2024-10-24

**作者**：David Khachaturov

**摘要**：Quantifying robustness in a single measure for the purposes of model
selection, development of adversarial training methods, and anticipating trends
has so far been elusive. The simplest metric to consider is the number of
trainable parameters in a model but this has previously been shown to be
insufficient at explaining robustness properties. A variety of other metrics,
such as ones based on boundary thickness and gradient flatness have been
proposed but have been shown to be inadequate proxies for robustness.
  In this work, we investigate the relationship between a model's effective
dimensionality, which can be thought of as model complexity, and its robustness
properties. We run experiments on commercial\-scale models that are often used
in real\-world environments such as YOLO and ResNet. We reveal a near\-linear
inverse relationship between effective dimensionality and adversarial
robustness, that is models with a lower dimensionality exhibit better
robustness. We investigate the effect of a variety of adversarial training
methods on effective dimensionality and find the same inverse linear
relationship present, suggesting that effective dimensionality can serve as a
useful criterion for model selection and robustness evaluation, providing a
more nuanced and effective metric than parameter count or previously\-tested
measures.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2410.18556v1)

---


## Comparing YOLO11 and YOLOv8 for instance segmentation of occluded and non\-occluded immature green fruits in complex orchard environment

**发布日期**：2024-10-24

**作者**：Ranjan Sapkota

**摘要**：This study conducted a comprehensive performance evaluation on YOLO11 and
YOLOv8, the latest in the "You Only Look Once" \(YOLO\) series, focusing on their
instance segmentation capabilities for immature green apples in orchard
environments. YOLO11n\-seg achieved the highest mask precision across all
categories with a notable score of 0.831, highlighting its effectiveness in
fruit detection. YOLO11m\-seg and YOLO11l\-seg excelled in non\-occluded and
occluded fruitlet segmentation with scores of 0.851 and 0.829, respectively.
Additionally, YOLO11x\-seg led in mask recall for all categories, achieving a
score of 0.815, with YOLO11m\-seg performing best for non\-occluded immature
green fruitlets at 0.858 and YOLOv8x\-seg leading the occluded category with
0.800. In terms of mean average precision at a 50\\% intersection over union
\(mAP@50\), YOLO11m\-seg consistently outperformed, registering the highest scores
for both box and mask segmentation, at 0.876 and 0.860 for the "All" class and
0.908 and 0.909 for non\-occluded immature fruitlets, respectively. YOLO11l\-seg
and YOLOv8l\-seg shared the top box mAP@50 for occluded immature fruitlets at
0.847, while YOLO11m\-seg achieved the highest mask mAP@50 of 0.810. Despite the
advancements in YOLO11, YOLOv8n surpassed its counterparts in image processing
speed, with an impressive inference speed of 3.3 milliseconds, compared to the
fastest YOLO11 series model at 4.8 milliseconds, underscoring its suitability
for real\-time agricultural applications related to complex green fruit
environments.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2410.19869v1)

---


## Breaking the Illusion: Real\-world Challenges for Adversarial Patches in Object Detection

**发布日期**：2024-10-23

**作者**：Jakob Shack

**摘要**：Adversarial attacks pose a significant threat to the robustness and
reliability of machine learning systems, particularly in computer vision
applications. This study investigates the performance of adversarial patches
for the YOLO object detection network in the physical world. Two attacks were
tested: a patch designed to be placed anywhere within the scene \- global patch,
and another patch intended to partially overlap with specific object targeted
for removal from detection \- local patch. Various factors such as patch size,
position, rotation, brightness, and hue were analyzed to understand their
impact on the effectiveness of the adversarial patches. The results reveal a
notable dependency on these parameters, highlighting the challenges in
maintaining attack efficacy in real\-world conditions. Learning to align
digitally applied transformation parameters with those measured in the real
world still results in up to a 64\\% discrepancy in patch performance. These
findings underscore the importance of understanding environmental influences on
adversarial attacks, which can inform the development of more robust defenses
for practical machine learning applications.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2410.19863v1)

---


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

