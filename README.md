# 每日从arXiv中获取最新YOLO相关论文


## microYOLO: Towards Single\-Shot Object Detection on Microcontrollers

**发布日期**：2024-08-28

**作者**：Mark Deutel

**摘要**：This work\-in\-progress paper presents results on the feasibility of
single\-shot object detection on microcontrollers using YOLO. Single\-shot object
detectors like YOLO are widely used, however due to their complexity mainly on
larger GPU\-based platforms. We present microYOLO, which can be used on Cortex\-M
based microcontrollers, such as the OpenMV H7 R2, achieving about 3.5 FPS when
classifying 128x128 RGB images while using less than 800 KB Flash and less than
350 KB RAM. Furthermore, we share experimental results for three different
object detection tasks, analyzing the accuracy of microYOLO on them.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2408.15865v1)

---


## YOLO\-Stutter: End\-to\-end Region\-Wise Speech Dysfluency Detection

**发布日期**：2024-08-27

**作者**：Xuanru Zhou

**摘要**：Dysfluent speech detection is the bottleneck for disordered speech analysis
and spoken language learning. Current state\-of\-the\-art models are governed by
rule\-based systems which lack efficiency and robustness, and are sensitive to
template design. In this paper, we propose YOLO\-Stutter: a first end\-to\-end
method that detects dysfluencies in a time\-accurate manner. YOLO\-Stutter takes
imperfect speech\-text alignment as input, followed by a spatial feature
aggregator, and a temporal dependency extractor to perform region\-wise boundary
and class predictions. We also introduce two dysfluency corpus, VCTK\-Stutter
and VCTK\-TTS, that simulate natural spoken dysfluencies including repetition,
block, missing, replacement, and prolongation. Our end\-to\-end method achieves
state\-of\-the\-art performance with a minimum number of trainable parameters for
on both simulated data and real aphasia speech. Code and datasets are
open\-sourced at https://github.com/rorizzz/YOLO\-Stutter


**代码链接**：https://github.com/rorizzz/YOLO-Stutter

**论文链接**：[阅读更多](http://arxiv.org/abs/2408.15297v1)

---


## Intraoperative Glioma Segmentation with YOLO \+ SAM for Improved Accuracy in Tumor Resection

**发布日期**：2024-08-27

**作者**：Samir Kassam

**摘要**：Gliomas, a common type of malignant brain tumor, present significant surgical
challenges due to their similarity to healthy tissue. Preoperative Magnetic
Resonance Imaging \(MRI\) images are often ineffective during surgery due to
factors such as brain shift, which alters the position of brain structures and
tumors. This makes real\-time intraoperative MRI \(ioMRI\) crucial, as it provides
updated imaging that accounts for these shifts, ensuring more accurate tumor
localization and safer resections. This paper presents a deep learning pipeline
combining You Only Look Once Version 8 \(YOLOv8\) and Segment Anything Model
Vision Transformer\-base \(SAM ViT\-b\) to enhance glioma detection and
segmentation during ioMRI. Our model was trained using the Brain Tumor
Segmentation 2021 \(BraTS 2021\) dataset, which includes standard magnetic
resonance imaging \(MRI\) images, and noise\-augmented MRI images that simulate
ioMRI images. Noised MRI images are harder for a deep learning pipeline to
segment, but they are more representative of surgical conditions. Achieving a
Dice Similarity Coefficient \(DICE\) score of 0.79, our model performs comparably
to state\-of\-the\-art segmentation models tested on noiseless data. This
performance demonstrates the model's potential to assist surgeons in maximizing
tumor resection and improving surgical outcomes.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2408.14847v1)

---


## Multi\-faceted Sensory Substitution for Curb Alerting: A Pilot Investigation in Persons with Blindness and Low Vision

**发布日期**：2024-08-26

**作者**：Ligao Ruan

**摘要**：Curbs \-\- the edge of a raised sidewalk at the point where it meets a street
\-\- crucial in urban environments where they help delineate safe pedestrian
zones, from dangerous vehicular lanes. However, curbs themselves are
significant navigation hazards, particularly for people who are blind or have
low vision \(pBLV\). The challenges faced by pBLV in detecting and properly
orientating themselves for these abrupt elevation changes can lead to falls and
serious injuries. Despite recent advancements in assistive technologies, the
detection and early warning of curbs remains a largely unsolved challenge. This
paper aims to tackle this gap by introducing a novel, multi\-faceted sensory
substitution approach hosted on a smart wearable; the platform leverages an RGB
camera and an embedded system to capture and segment curbs in real time and
provide early warning and orientation information. The system utilizes YOLO
\(You Only Look Once\) v8 segmentation model, trained on our custom curb dataset
for the camera input. The output of the system consists of adaptive auditory
beeps, abstract sonification, and speech, conveying information about the
relative distance and orientation of curbs. Through human\-subjects
experimentation, we demonstrate the effectiveness of the system as compared to
the white cane. Results show that our system can provide advanced warning
through a larger safety window than the cane, while offering nearly identical
curb orientation information.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2408.14578v2)

---


## LSM\-YOLO: A Compact and Effective ROI Detector for Medical Detection

**发布日期**：2024-08-26

**作者**：Zhongwen Yu

**摘要**：In existing medical Region of Interest \(ROI\) detection, there lacks an
algorithm that can simultaneously satisfy both real\-time performance and
accuracy, not meeting the growing demand for automatic detection in medicine.
Although the basic YOLO framework ensures real\-time detection due to its fast
speed, it still faces challenges in maintaining precision concurrently. To
alleviate the above problems, we propose a novel model named Lightweight Shunt
Matching\-YOLO \(LSM\-YOLO\), with Lightweight Adaptive Extraction \(LAE\) and
Multipath Shunt Feature Matching \(MSFM\). Firstly, by using LAE to refine
feature extraction, the model can obtain more contextual information and
high\-resolution details from multiscale feature maps, thereby extracting
detailed features of ROI in medical images while reducing the influence of
noise. Secondly, MSFM is utilized to further refine the fusion of high\-level
semantic features and low\-level visual features, enabling better fusion between
ROI features and neighboring features, thereby improving the detection rate for
better diagnostic assistance. Experimental results demonstrate that LSM\-YOLO
achieves 48.6% AP on a private dataset of pancreatic tumors, 65.1% AP on the
BCCD blood cell detection public dataset, and 73.0% AP on the Br35h brain tumor
detection public dataset. Our model achieves state\-of\-the\-art performance with
minimal parameter cost on the above three datasets. The source codes are at:
https://github.com/VincentYuuuuuu/LSM\-YOLO.


**代码链接**：https://github.com/VincentYuuuuuu/LSM-YOLO.

**论文链接**：[阅读更多](http://arxiv.org/abs/2408.14087v1)

---


## Enhancing Robustness of Human Detection Algorithms in Maritime SAR through Augmented Aerial Images to Simulate Weather Conditions

**发布日期**：2024-08-25

**作者**：Miguel Tjia

**摘要**：7,651 cases of Search and Rescue Missions \(SAR\) were reported by the United
States Coast Guard in 2024, with over 1322 SAR helicopters deployed in the 6
first months alone. Through the utilizations of YOLO, we were able to run
different weather conditions and lighting from our augmented dataset for
training. YOLO then utilizes CNNs to apply a series of convolutions and pooling
layers to the input image, where the convolution layers are able to extract the
main features of the image. Through this, our YOLO model is able to learn to
differentiate different objects which may considerably improve its accuracy,
possibly enhancing the efficiency of SAR operations through enhanced detection
accuracy. This paper aims to improve the model's accuracy of human detection in
maritime SAR by evaluating a robust datasets containing various elevations and
geological locations, as well as through data augmentation which simulates
different weather and lighting. We observed that models trained on augmented
datasets outperformed their non\-augmented counterparts in which the human
recall scores ranged from 0.891 to 0.911 with an improvement rate of 3.4\\% on
the YOLOv5l model. Results showed that these models demonstrate greater
robustness to real\-world conditions in varying of weather, brightness, tint,
and contrast.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2408.13766v2)

---


## VFM\-Det: Towards High\-Performance Vehicle Detection via Large Foundation Models

**发布日期**：2024-08-23

**作者**：Wentao Wu

**摘要**：Existing vehicle detectors are usually obtained by training a typical
detector \(e.g., YOLO, RCNN, DETR series\) on vehicle images based on a
pre\-trained backbone \(e.g., ResNet, ViT\). Some researchers also exploit and
enhance the detection performance using pre\-trained large foundation models.
However, we think these detectors may only get sub\-optimal results because the
large models they use are not specifically designed for vehicles. In addition,
their results heavily rely on visual features, and seldom of they consider the
alignment between the vehicle's semantic information and visual
representations. In this work, we propose a new vehicle detection paradigm
based on a pre\-trained foundation vehicle model \(VehicleMAE\) and a large
language model \(T5\), termed VFM\-Det. It follows the region proposal\-based
detection framework and the features of each proposal can be enhanced using
VehicleMAE. More importantly, we propose a new VAtt2Vec module that predicts
the vehicle semantic attributes of these proposals and transforms them into
feature vectors to enhance the vision features via contrastive learning.
Extensive experiments on three vehicle detection benchmark datasets thoroughly
proved the effectiveness of our vehicle detector. Specifically, our model
improves the baseline approach by $\+5.1\\%$, $\+6.2\\%$ on the $AP\_\{0.5\}$,
$AP\_\{0.75\}$ metrics, respectively, on the Cityscapes dataset.The source code of
this work will be released at https://github.com/Event\-AHU/VFM\-Det.


**代码链接**：https://github.com/Event-AHU/VFM-Det.

**论文链接**：[阅读更多](http://arxiv.org/abs/2408.13031v1)

---


## Enhanced Parking Perception by Multi\-Task Fisheye Cross\-view Transformers

**发布日期**：2024-08-22

**作者**：Antonyo Musabini

**摘要**：Current parking area perception algorithms primarily focus on detecting
vacant slots within a limited range, relying on error\-prone homographic
projection for both labeling and inference. However, recent advancements in
Advanced Driver Assistance System \(ADAS\) require interaction with end\-users
through comprehensive and intelligent Human\-Machine Interfaces \(HMIs\). These
interfaces should present a complete perception of the parking area going from
distinguishing vacant slots' entry lines to the orientation of other parked
vehicles. This paper introduces Multi\-Task Fisheye Cross View Transformers \(MT
F\-CVT\), which leverages features from a four\-camera fisheye Surround\-view
Camera System \(SVCS\) with multihead attentions to create a detailed Bird\-Eye
View \(BEV\) grid feature map. Features are processed by both a segmentation
decoder and a Polygon\-Yolo based object detection decoder for parking slots and
vehicles. Trained on data labeled using LiDAR, MT F\-CVT positions objects
within a 25m x 25m real open\-road scenes with an average error of only 20 cm.
Our larger model achieves an F\-1 score of 0.89. Moreover the smaller model
operates at 16 fps on an Nvidia Jetson Orin embedded board, with similar
detection results to the larger one. MT F\-CVT demonstrates robust
generalization capability across different vehicles and camera rig
configurations. A demo video from an unseen vehicle and camera rig is available
at: https://streamable.com/jjw54x.


**代码链接**：https://streamable.com/jjw54x.

**论文链接**：[阅读更多](http://arxiv.org/abs/2408.12575v1)

---


## OVA\-DETR: Open Vocabulary Aerial Object Detection Using Image\-Text Alignment and Fusion

**发布日期**：2024-08-22

**作者**：Guoting Wei

**摘要**：Aerial object detection has been a hot topic for many years due to its wide
application requirements. However, most existing approaches can only handle
predefined categories, which limits their applicability for the open scenarios
in real\-world. In this paper, we extend aerial object detection to open
scenarios by exploiting the relationship between image and text, and propose
OVA\-DETR, a high\-efficiency open\-vocabulary detector for aerial images.
Specifically, based on the idea of image\-text alignment, we propose region\-text
contrastive loss to replace the category regression loss in the traditional
detection framework, which breaks the category limitation. Then, we propose
Bidirectional Vision\-Language Fusion \(Bi\-VLF\), which includes a dual\-attention
fusion encoder and a multi\-level text\-guided Fusion Decoder. The dual\-attention
fusion encoder enhances the feature extraction process in the encoder part. The
multi\-level text\-guided Fusion Decoder is designed to improve the detection
ability for small objects, which frequently appear in aerial object detection
scenarios. Experimental results on three widely used benchmark datasets show
that our proposed method significantly improves the mAP and recall, while
enjoying faster inference speed. For instance, in zero shot detection
experiments on DIOR, the proposed OVA\-DETR outperforms DescReg and YOLO\-World
by 37.4% and 33.1%, respectively, while achieving 87 FPS inference speed, which
is 7.9x faster than DescReg and 3x faster than YOLO\-world. The code is
available at https://github.com/GT\-Wei/OVA\-DETR.


**代码链接**：https://github.com/GT-Wei/OVA-DETR.

**论文链接**：[阅读更多](http://arxiv.org/abs/2408.12246v1)

---


## On the Potential of Open\-Vocabulary Models for Object Detection in Unusual Street Scenes

**发布日期**：2024-08-20

**作者**：Sadia Ilyas

**摘要**：Out\-of\-distribution \(OOD\) object detection is a critical task focused on
detecting objects that originate from a data distribution different from that
of the training data. In this study, we investigate to what extent
state\-of\-the\-art open\-vocabulary object detectors can detect unusual objects in
street scenes, which are considered as OOD or rare scenarios with respect to
common street scene datasets. Specifically, we evaluate their performance on
the OoDIS Benchmark, which extends RoadAnomaly21 and RoadObstacle21 from
SegmentMeIfYouCan, as well as LostAndFound, which was recently extended to
object level annotations. The objective of our study is to uncover
short\-comings of contemporary object detectors in challenging real\-world, and
particularly in open\-world scenarios. Our experiments reveal that open
vocabulary models are promising for OOD object detection scenarios, however far
from perfect. Substantial improvements are required before they can be reliably
deployed in real\-world applications. We benchmark four state\-of\-the\-art
open\-vocabulary object detection models on three different datasets.
Noteworthily, Grounding DINO achieves the best results on RoadObstacle21 and
LostAndFound in our study with an AP of 48.3% and 25.4% respectively.
YOLO\-World excels on RoadAnomaly21 with an AP of 21.2%.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2408.11221v1)

---

