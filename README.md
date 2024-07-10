# 每日从arXiv中获取最新YOLO相关论文


## Muzzle\-Based Cattle Identification System Using Artificial Intelligence \(AI\)

**发布日期**：2024-07-08

**作者**：Hasan Zohirul Islam

**摘要**：Absence of tamper\-proof cattle identification technology was a significant
problem preventing insurance companies from providing livestock insurance. This
lack of technology had devastating financial consequences for marginal farmers
as they did not have the opportunity to claim compensation for any unexpected
events such as the accidental death of cattle in Bangladesh. Using machine
learning and deep learning algorithms, we have solved the bottleneck of cattle
identification by developing and introducing a muzzle\-based cattle
identification system. The uniqueness of cattle muzzles has been scientifically
established, which resembles human fingerprints. This is the fundamental
premise that prompted us to develop a cattle identification system that
extracts the uniqueness of cattle muzzles. For this purpose, we collected
32,374 images from 826 cattle. Contrast\-limited adaptive histogram equalization
\(CLAHE\) with sharpening filters was applied in the preprocessing steps to
remove noise from images. We used the YOLO algorithm for cattle muzzle
detection in the image and the FaceNet architecture to learn unified embeddings
from muzzle images using squared $L\_2$ distances. Our system performs with an
accuracy of $96.489\\%$, $F\_1$ score of $97.334\\%$, and a true positive rate
\(tpr\) of $87.993\\%$ at a remarkably low false positive rate \(fpr\) of $0.098\\%$.
This reliable and efficient system for identifying cattle can significantly
advance livestock insurance and precision farming.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2407.06096v1)

---


## Quantizing YOLOv7: A Comprehensive Study

**发布日期**：2024-07-06

**作者**：Mohammadamin Baghbanbashi

**摘要**：YOLO is a deep neural network \(DNN\) model presented for robust real\-time
object detection following the one\-stage inference approach. It outperforms
other real\-time object detectors in terms of speed and accuracy by a wide
margin. Nevertheless, since YOLO is developed upon a DNN backbone with numerous
parameters, it will cause excessive memory load, thereby deploying it on
memory\-constrained devices is a severe challenge in practice. To overcome this
limitation, model compression techniques, such as quantizing parameters to
lower\-precision values, can be adopted. As the most recent version of YOLO,
YOLOv7 achieves such state\-of\-the\-art performance in speed and accuracy in the
range of 5 FPS to 160 FPS that it surpasses all former versions of YOLO and
other existing models in this regard. So far, the robustness of several
quantization schemes has been evaluated on older versions of YOLO. These
methods may not necessarily yield similar results for YOLOv7 as it utilizes a
different architecture. In this paper, we conduct in\-depth research on the
effectiveness of a variety of quantization schemes on the pre\-trained weights
of the state\-of\-the\-art YOLOv7 model. Experimental results demonstrate that
using 4\-bit quantization coupled with the combination of different
granularities results in ~3.92x and ~3.86x memory\-saving for uniform and
non\-uniform quantization, respectively, with only 2.5% and 1% accuracy loss
compared to the full\-precision baseline model.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2407.04943v1)

---


## SH17: A Dataset for Human Safety and Personal Protective Equipment Detection in Manufacturing Industry

**发布日期**：2024-07-05

**作者**：Hafiz Mughees Ahmad

**摘要**：Workplace accidents continue to pose significant risks for human safety,
particularly in industries such as construction and manufacturing, and the
necessity for effective Personal Protective Equipment \(PPE\) compliance has
become increasingly paramount. Our research focuses on the development of
non\-invasive techniques based on the Object Detection \(OD\) and Convolutional
Neural Network \(CNN\) to detect and verify the proper use of various types of
PPE such as helmets, safety glasses, masks, and protective clothing. This study
proposes the SH17 Dataset, consisting of 8,099 annotated images containing
75,994 instances of 17 classes collected from diverse industrial environments,
to train and validate the OD models. We have trained state\-of\-the\-art OD models
for benchmarking, and initial results demonstrate promising accuracy levels
with You Only Look Once \(YOLO\)v9\-e model variant exceeding 70.9% in PPE
detection. The performance of the model validation on cross\-domain datasets
suggests that integrating these technologies can significantly improve safety
management systems, providing a scalable and efficient solution for industries
striving to meet human safety regulations and protect their workforce. The
dataset is available at https://github.com/ahmadmughees/sh17dataset.


**代码链接**：https://github.com/ahmadmughees/sh17dataset.

**论文链接**：[阅读更多](http://arxiv.org/abs/2407.04590v1)

---


## Multi\-Branch Auxiliary Fusion YOLO with Re\-parameterization Heterogeneous Convolutional for accurate object detection

**发布日期**：2024-07-05

**作者**：Zhiqiang Yang

**摘要**：Due to the effective performance of multi\-scale feature fusion, Path
Aggregation FPN \(PAFPN\) is widely employed in YOLO detectors. However, it
cannot efficiently and adaptively integrate high\-level semantic information
with low\-level spatial information simultaneously. We propose a new model named
MAF\-YOLO in this paper, which is a novel object detection framework with a
versatile neck named Multi\-Branch Auxiliary FPN \(MAFPN\). Within MAFPN, the
Superficial Assisted Fusion \(SAF\) module is designed to combine the output of
the backbone with the neck, preserving an optimal level of shallow information
to facilitate subsequent learning. Meanwhile, the Advanced Assisted Fusion
\(AAF\) module deeply embedded within the neck conveys a more diverse range of
gradient information to the output layer.
  Furthermore, our proposed Re\-parameterized Heterogeneous Efficient Layer
Aggregation Network \(RepHELAN\) module ensures that both the overall model
architecture and convolutional design embrace the utilization of heterogeneous
large convolution kernels. Therefore, this guarantees the preservation of
information related to small targets while simultaneously achieving the
multi\-scale receptive field. Finally, taking the nano version of MAF\-YOLO for
example, it can achieve 42.4% AP on COCO with only 3.76M learnable parameters
and 10.51G FLOPs, and approximately outperforms YOLOv8n by about 5.1%. The
source code of this work is available at:
https://github.com/yang\-0201/MAF\-YOLO.


**代码链接**：https://github.com/yang-0201/MAF-YOLO.

**论文链接**：[阅读更多](http://arxiv.org/abs/2407.04381v1)

---


## YOLOv5, YOLOv8 and YOLOv10: The Go\-To Detectors for Real\-time Vision

**发布日期**：2024-07-03

**作者**：Muhammad Hussain

**摘要**：This paper presents a comprehensive review of the evolution of the YOLO \(You
Only Look Once\) object detection algorithm, focusing on YOLOv5, YOLOv8, and
YOLOv10. We analyze the architectural advancements, performance improvements,
and suitability for edge deployment across these versions. YOLOv5 introduced
significant innovations such as the CSPDarknet backbone and Mosaic
Augmentation, balancing speed and accuracy. YOLOv8 built upon this foundation
with enhanced feature extraction and anchor\-free detection, improving
versatility and performance. YOLOv10 represents a leap forward with NMS\-free
training, spatial\-channel decoupled downsampling, and large\-kernel
convolutions, achieving state\-of\-the\-art performance with reduced computational
overhead. Our findings highlight the progressive enhancements in accuracy,
efficiency, and real\-time performance, particularly emphasizing their
applicability in resource\-constrained environments. This review provides
insights into the trade\-offs between model complexity and detection accuracy,
offering guidance for selecting the most appropriate YOLO version for specific
edge computing applications.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2407.02988v1)

---


## GSO\-YOLO: Global Stability Optimization YOLO for Construction Site Detection

**发布日期**：2024-07-01

**作者**：Yuming Zhang

**摘要**：Safety issues at construction sites have long plagued the industry, posing
risks to worker safety and causing economic damage due to potential hazards.
With the advancement of artificial intelligence, particularly in the field of
computer vision, the automation of safety monitoring on construction sites has
emerged as a solution to this longstanding issue. Despite achieving impressive
performance, advanced object detection methods like YOLOv8 still face
challenges in handling the complex conditions found at construction sites. To
solve these problems, this study presents the Global Stability Optimization
YOLO \(GSO\-YOLO\) model to address challenges in complex construction sites. The
model integrates the Global Optimization Module \(GOM\) and Steady Capture Module
\(SCM\) to enhance global contextual information capture and detection stability.
The innovative AIoU loss function, which combines CIoU and EIoU, improves
detection accuracy and efficiency. Experiments on datasets like SODA, MOCS, and
CIS show that GSO\-YOLO outperforms existing methods, achieving SOTA
performance.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2407.00906v1)

---


## A Universal Railway Obstacle Detection System based on Semi\-supervised Segmentation And Optical Flow

**发布日期**：2024-06-27

**作者**：Qiushi Guo

**摘要**：Detecting obstacles in railway scenarios is both crucial and challenging due
to the wide range of obstacle categories and varying ambient conditions such as
weather and light. Given the impossibility of encompassing all obstacle
categories during the training stage, we address this out\-of\-distribution \(OOD\)
issue with a semi\-supervised segmentation approach guided by optical flow
clues. We reformulate the task as a binary segmentation problem instead of the
traditional object detection approach. To mitigate data shortages, we generate
highly realistic synthetic images using Segment Anything \(SAM\) and YOLO,
eliminating the need for manual annotation to produce abundant pixel\-level
annotations. Additionally, we leverage optical flow as prior knowledge to train
the model effectively. Several experiments are conducted, demonstrating the
feasibility and effectiveness of our approach.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2406.18908v1)

---


## Detection of Synthetic Face Images: Accuracy, Robustness, Generalization

**发布日期**：2024-06-25

**作者**：Nela Petrzelkova

**摘要**：An experimental study on detecting synthetic face images is presented. We
collected a dataset, called FF5, of five fake face image generators, including
recent diffusion models. We find that a simple model trained on a specific
image generator can achieve near\-perfect accuracy in separating synthetic and
real images. The model handles common image distortions \(reduced resolution,
compression\) by using data augmentation. Moreover, partial manipulations, where
synthetic images are blended into real ones by inpainting, are identified and
the area of the manipulation is localized by a simple model of YOLO
architecture. However, the model turned out to be vulnerable to adversarial
attacks and does not generalize to unseen generators. Failure to generalize to
detect images produced by a newer generator also occurs for recent
state\-of\-the\-art methods, which we tested on Realistic Vision, a fine\-tuned
version of StabilityAI's Stable Diffusion image generator.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2406.17547v1)

---


## POPCat: Propagation of particles for complex annotation tasks

**发布日期**：2024-06-24

**作者**：Adam Srebrnjak Yang

**摘要**：Novel dataset creation for all multi\-object tracking, crowd\-counting, and
industrial\-based videos is arduous and time\-consuming when faced with a unique
class that densely populates a video sequence. We propose a time efficient
method called POPCat that exploits the multi\-target and temporal features of
video data to produce a semi\-supervised pipeline for segmentation or box\-based
video annotation. The method retains the accuracy level associated with human
level annotation while generating a large volume of semi\-supervised annotations
for greater generalization. The method capitalizes on temporal features through
the use of a particle tracker to expand the domain of human\-provided target
points. This is done through the use of a particle tracker to reassociate the
initial points to a set of images that follow the labeled frame. A YOLO model
is then trained with this generated data, and then rapidly infers on the target
video. Evaluations are conducted on GMOT\-40, AnimalTrack, and Visdrone\-2019
benchmarks. These multi\-target video tracking/detection sets contain multiple
similar\-looking targets, camera movements, and other features that would
commonly be seen in "wild" situations. We specifically choose these difficult
datasets to demonstrate the efficacy of the pipeline and for comparison
purposes. The method applied on GMOT\-40, AnimalTrack, and Visdrone shows a
margin of improvement on recall/mAP50/mAP over the best results by a value of
24.5%/9.6%/4.8%, \-/43.1%/27.8%, and 7.5%/9.4%/7.5% where metrics were
collected.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2406.17183v1)

---


## Review of Zero\-Shot and Few\-Shot AI Algorithms in The Medical Domain

**发布日期**：2024-06-23

**作者**：Maged Badawi

**摘要**：In this paper, different techniques of few\-shot, zero\-shot, and regular
object detection have been investigated. The need for few\-shot learning and
zero\-shot learning techniques is crucial and arises from the limitations and
challenges in traditional machine learning, deep learning, and computer vision
methods where they require large amounts of data, plus the poor generalization
of those traditional methods.
  Those techniques can give us prominent results by using only a few training
sets reducing the required amounts of data and improving the generalization.
  This survey will highlight the recent papers of the last three years that
introduce the usage of few\-shot learning and zero\-shot learning techniques in
addressing the challenges mentioned earlier. In this paper we reviewed the
Zero\-shot, few\-shot and regular object detection methods and categorized them
in an understandable manner. Based on the comparison made within each category.
It been found that the approaches are quite impressive.
  This integrated review of diverse papers on few\-shot, zero\-shot, and regular
object detection reveals a shared focus on advancing the field through novel
frameworks and techniques. A noteworthy observation is the scarcity of detailed
discussions regarding the difficulties encountered during the development
phase. Contributions include the introduction of innovative models, such as
ZSD\-YOLO and GTNet, often showcasing improvements with various metrics such as
mean average precision \(mAP\),Recall@100 \(RE@100\), the area under the receiver
operating characteristic curve \(AUROC\) and precision. These findings underscore
a collective move towards leveraging vision\-language models for versatile
applications, with potential areas for future research including a more
thorough exploration of limitations and domain\-specific adaptations.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2406.16143v1)

---

