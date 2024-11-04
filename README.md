# 每日从arXiv中获取最新YOLO相关论文


## YOLOv11 for Vehicle Detection: Advancements, Performance, and Applications in Intelligent Transportation Systems

**发布日期**：2024-10-30

**作者**：Mujadded Al Rabbani Alif

**摘要**：Accurate vehicle detection is essential for the development of intelligent
transportation systems, autonomous driving, and traffic monitoring. This paper
presents a detailed analysis of YOLO11, the latest advancement in the YOLO
series of deep learning models, focusing exclusively on vehicle detection
tasks. Building upon the success of its predecessors, YOLO11 introduces
architectural improvements designed to enhance detection speed, accuracy, and
robustness in complex environments. Using a comprehensive dataset comprising
multiple vehicle types\-cars, trucks, buses, motorcycles, and bicycles we
evaluate YOLO11's performance using metrics such as precision, recall, F1
score, and mean average precision \(mAP\). Our findings demonstrate that YOLO11
surpasses previous versions \(YOLOv8 and YOLOv10\) in detecting smaller and more
occluded vehicles while maintaining a competitive inference time, making it
well\-suited for real\-time applications. Comparative analysis shows significant
improvements in the detection of complex vehicle geometries, further
contributing to the development of efficient and scalable vehicle detection
systems. This research highlights YOLO11's potential to enhance autonomous
vehicle performance and traffic monitoring systems, offering insights for
future developments in the field.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2410.22898v1)

---


## From Explicit Rules to Implicit Reasoning in an Interpretable Violence Monitoring System

**发布日期**：2024-10-29

**作者**：Wen\-Dong Jiang

**摘要**：Recently, research based on pre\-trained models has demonstrated outstanding
performance in violence surveillance tasks. However, these black\-box systems
face challenges regarding explainability during training and inference
processes. An important question is how to incorporate explicit knowledge into
these implicit models, thereby designing expert\-driven and interpretable
violence surveillance systems. This paper proposes a new paradigm for weakly
supervised violence monitoring \(WSVM\) called Rule base Violence monitoring
\(RuleVM\). The proposed RuleVM uses a dual\-branch structure for different
designs for images and text. One of the branches is called the implicit branch,
which uses only visual features for coarse\-grained binary classification. In
this branch, image feature extraction is divided into two channels: one
responsible for extracting scene frames and the other focusing on extracting
actions. The other branch is called the explicit branch, which utilizes
language\-image alignment to perform fine\-grained classification. For the
language channel design in the explicit branch, the proposed RuleCLIP uses the
state\-of\-the\-art YOLO\-World model to detect objects and actions in video
frames, and association rules are identified through data mining methods as
descriptions of the video. Leveraging the dual\-branch architecture, RuleVM
achieves interpretable coarse\-grained and fine\-grained violence surveillance.
Extensive experiments were conducted on two commonly used benchmarks, and the
results show that RuleCLIP achieved the best performance in both coarse\-grained
and fine\-grained detection, significantly outperforming existing
state\-of\-the\-art methods. Moreover, interpretability experiments uncovered some
interesting rules, such as the observation that as the number of people
increases, the risk level of violent behavior also rises.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2410.21991v2)

---


## PK\-YOLO: Pretrained Knowledge Guided YOLO for Brain Tumor Detection in Multiplanar MRI Slices

**发布日期**：2024-10-29

**作者**：Ming Kang

**摘要**：Brain tumor detection in multiplane Magnetic Resonance Imaging \(MRI\) slices
is a challenging task due to the various appearances and relationships in the
structure of the multiplane images. In this paper, we propose a new You Only
Look Once \(YOLO\)\-based detection model that incorporates Pretrained Knowledge
\(PK\), called PK\-YOLO, to improve the performance for brain tumor detection in
multiplane MRI slices. To our best knowledge, PK\-YOLO is the first pretrained
knowledge guided YOLO\-based object detector. The main components of the new
method are a pretrained pure lightweight convolutional neural network\-based
backbone via sparse masked modeling, a YOLO architecture with the pretrained
backbone, and a regression loss function for improving small object detection.
The pretrained backbone allows for feature transferability of object queries on
individual plane MRI slices into the model encoders, and the learned domain
knowledge base can improve in\-domain detection. The improved loss function can
further boost detection performance on small\-size brain tumors in multiplanar
two\-dimensional MRI slices. Experimental results show that the proposed PK\-YOLO
achieves competitive performance on the multiplanar MRI brain tumor detection
datasets compared to state\-of\-the\-art YOLO\-like and DETR\-like object detectors.
The code is available at https://github.com/mkang315/PK\-YOLO.


**代码链接**：https://github.com/mkang315/PK-YOLO.

**论文链接**：[阅读更多](http://arxiv.org/abs/2410.21822v1)

---


## TACO: Adversarial Camouflage Optimization on Trucks to Fool Object Detectors

**发布日期**：2024-10-28

**作者**：Adonisz Dimitriu

**摘要**：Adversarial attacks threaten the reliability of machine learning models in
critical applications like autonomous vehicles and defense systems. As object
detectors become more robust with models like YOLOv8, developing effective
adversarial methodologies is increasingly challenging. We present Truck
Adversarial Camouflage Optimization \(TACO\), a novel framework that generates
adversarial camouflage patterns on 3D vehicle models to deceive
state\-of\-the\-art object detectors. Adopting Unreal Engine 5, TACO integrates
differentiable rendering with a Photorealistic Rendering Network to optimize
adversarial textures targeted at YOLOv8. To ensure the generated textures are
both effective in deceiving detectors and visually plausible, we introduce the
Convolutional Smooth Loss function, a generalized smooth loss function.
Experimental evaluations demonstrate that TACO significantly degrades YOLOv8's
detection performance, achieving an AP@0.5 of 0.0099 on unseen test data.
Furthermore, these adversarial patterns exhibit strong transferability to other
object detection models such as Faster R\-CNN and earlier YOLO versions.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2410.21443v1)

---


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

**论文链接**：[阅读更多](http://arxiv.org/abs/2410.19869v2)

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

