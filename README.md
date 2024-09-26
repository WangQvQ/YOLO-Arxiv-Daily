# 每日从arXiv中获取最新YOLO相关论文


## Classification of Gleason Grading in Prostate Cancer Histopathology Images Using Deep Learning Techniques: YOLO, Vision Transformers, and Vision Mamba

**发布日期**：2024-09-25

**作者**：Amin Malekmohammadi

**摘要**：Prostate cancer ranks among the leading health issues impacting men, with the
Gleason scoring system serving as the primary method for diagnosis and
prognosis. This system relies on expert pathologists to evaluate samples of
prostate tissue and assign a Gleason grade, a task that requires significant
time and manual effort. To address this challenge, artificial intelligence \(AI\)
solutions have been explored to automate the grading process. In light of these
challenges, this study evaluates and compares the effectiveness of three deep
learning methodologies, YOLO, Vision Transformers, and Vision Mamba, in
accurately classifying Gleason grades from histopathology images. The goal is
to enhance diagnostic precision and efficiency in prostate cancer management.
This study utilized two publicly available datasets, Gleason2019 and SICAPv2,
to train and test the performance of YOLO, Vision Transformers, and Vision
Mamba models. Each model was assessed based on its ability to classify Gleason
grades accurately, considering metrics such as false positive rate, false
negative rate, precision, and recall. The study also examined the computational
efficiency and applicability of each method in a clinical setting. Vision Mamba
demonstrated superior performance across all metrics, achieving high precision
and recall rates while minimizing false positives and negatives. YOLO showed
promise in terms of speed and efficiency, particularly beneficial for real\-time
analysis. Vision Transformers excelled in capturing long\-range dependencies
within images, although they presented higher computational complexity compared
to the other models. Vision Mamba emerges as the most effective model for
Gleason grade classification in histopathology images, offering a balance
between accuracy and computational efficiency.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2409.17122v1)

---


## Deep Learning and Machine Learning, Advancing Big Data Analytics and Management: Handy Appetizer

**发布日期**：2024-09-25

**作者**：Benji Peng

**摘要**：This book explores the role of Artificial Intelligence \(AI\), Machine Learning
\(ML\), and Deep Learning \(DL\) in driving the progress of big data analytics and
management. The book focuses on simplifying the complex mathematical concepts
behind deep learning, offering intuitive visualizations and practical case
studies to help readers understand how neural networks and technologies like
Convolutional Neural Networks \(CNNs\) work. It introduces several classic models
and technologies such as Transformers, GPT, ResNet, BERT, and YOLO,
highlighting their applications in fields like natural language processing,
image recognition, and autonomous driving. The book also emphasizes the
importance of pre\-trained models and how they can enhance model performance and
accuracy, with instructions on how to apply these models in various real\-world
scenarios. Additionally, it provides an overview of key big data management
technologies like SQL and NoSQL databases, as well as distributed computing
frameworks such as Apache Hadoop and Spark, explaining their importance in
managing and processing vast amounts of data. Ultimately, the book underscores
the value of mastering deep learning and big data management skills as critical
tools for the future workforce, making it an essential resource for both
beginners and experienced professionals.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2409.17120v1)

---


## Source\-Free Domain Adaptation for YOLO Object Detection

**发布日期**：2024-09-25

**作者**：Simon Varailhon

**摘要**：Source\-free domain adaptation \(SFDA\) is a challenging problem in object
detection, where a pre\-trained source model is adapted to a new target domain
without using any source domain data for privacy and efficiency reasons. Most
state\-of\-the\-art SFDA methods for object detection have been proposed for
Faster\-RCNN, a detector that is known to have high computational complexity.
This paper focuses on domain adaptation techniques for real\-world vision
systems, particularly for the YOLO family of single\-shot detectors known for
their fast baselines and practical applications. Our proposed SFDA method \-
Source\-Free YOLO \(SF\-YOLO\) \- relies on a teacher\-student framework in which the
student receives images with a learned, target domain\-specific augmentation,
allowing the model to be trained with only unlabeled target data and without
requiring feature alignment. A challenge with self\-training using a
mean\-teacher architecture in the absence of labels is the rapid decline of
accuracy due to noisy or drifting pseudo\-labels. To address this issue, a
teacher\-to\-student communication mechanism is introduced to help stabilize the
training and reduce the reliance on annotated target data for model selection.
Despite its simplicity, our approach is competitive with state\-of\-the\-art
detectors on several challenging benchmark datasets, even sometimes
outperforming methods that use source data for adaptation.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2409.16538v1)

---


## Segmentation Strategies in Deep Learning for Prostate Cancer Diagnosis: A Comparative Study of Mamba, SAM, and YOLO

**发布日期**：2024-09-24

**作者**：Ali Badiezadeh

**摘要**：Accurate segmentation of prostate cancer histopathology images is crucial for
diagnosis and treatment planning. This study presents a comparative analysis of
three deep learning\-based methods, Mamba, SAM, and YOLO, for segmenting
prostate cancer histopathology images. We evaluated the performance of these
models on two comprehensive datasets, Gleason 2019 and SICAPv2, using Dice
score, precision, and recall metrics. Our results show that the High\-order
Vision Mamba UNet \(H\-vmunet\) model outperforms the other two models, achieving
the highest scores across all metrics on both datasets. The H\-vmunet model's
advanced architecture, which integrates high\-order visual state spaces and
2D\-selective\-scan operations, enables efficient and sensitive lesion detection
across different scales. Our study demonstrates the potential of the H\-vmunet
model for clinical applications and highlights the importance of robust
validation and comparison of deep learning\-based methods for medical image
analysis. The findings of this study contribute to the development of accurate
and reliable computer\-aided diagnosis systems for prostate cancer. The code is
available at http://github.com/alibdz/prostate\-segmentation.


**代码链接**：http://github.com/alibdz/prostate-segmentation.

**论文链接**：[阅读更多](http://arxiv.org/abs/2409.16205v1)

---


## A Computer Vision Approach for Autonomous Cars to Drive Safe at Construction Zone

**发布日期**：2024-09-24

**作者**：Abu Shad Ahammed

**摘要**：To build a smarter and safer city, a secure, efficient, and sustainable
transportation system is a key requirement. The autonomous driving system \(ADS\)
plays an important role in the development of smart transportation and is
considered one of the major challenges facing the automotive sector in recent
decades. A car equipped with an autonomous driving system \(ADS\) comes with
various cutting\-edge functionalities such as adaptive cruise control, collision
alerts, automated parking, and more. A primary area of research within ADAS
involves identifying road obstacles in construction zones regardless of the
driving environment. This paper presents an innovative and highly accurate road
obstacle detection model utilizing computer vision technology that can be
activated in construction zones and functions under diverse drift conditions,
ultimately contributing to build a safer road transportation system. The model
developed with the YOLO framework achieved a mean average precision exceeding
94\\% and demonstrated an inference time of 1.6 milliseconds on the validation
dataset, underscoring the robustness of the methodology applied to mitigate
hazards and risks for autonomous vehicles.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2409.15809v1)

---


## Real\-Time Pedestrian Detection on IoT Edge Devices: A Lightweight Deep Learning Approach

**发布日期**：2024-09-24

**作者**：Muhammad Dany Alfikri

**摘要**：Artificial intelligence \(AI\) has become integral to our everyday lives.
Computer vision has advanced to the point where it can play the safety critical
role of detecting pedestrians at road intersections in intelligent
transportation systems and alert vehicular traffic as to potential collisions.
Centralized computing analyzes camera feeds and generates alerts for nearby
vehicles. However, real\-time applications face challenges such as latency,
limited data transfer speeds, and the risk of life loss. Edge servers offer a
potential solution for real\-time applications, providing localized computing
and storage resources and lower response times. Unfortunately, edge servers
have limited processing power. Lightweight deep learning \(DL\) techniques enable
edge servers to utilize compressed deep neural network \(DNN\) models.
  The research explores implementing a lightweight DL model on Artificial
Intelligence of Things \(AIoT\) edge devices. An optimized You Only Look Once
\(YOLO\) based DL model is deployed for real\-time pedestrian detection, with
detection events transmitted to the edge server using the Message Queuing
Telemetry Transport \(MQTT\) protocol. The simulation results demonstrate that
the optimized YOLO model can achieve real\-time pedestrian detection, with a
fast inference speed of 147 milliseconds, a frame rate of 2.3 frames per
second, and an accuracy of 78%, representing significant improvements over
baseline models.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2409.15740v1)

---


## PDT: Uav Target Detection Dataset for Pests and Diseases Tree

**发布日期**：2024-09-24

**作者**：Mingle Zhou

**摘要**：UAVs emerge as the optimal carriers for visual weed iden?tification and
integrated pest and disease management in crops. How?ever, the absence of
specialized datasets impedes the advancement of model development in this
domain. To address this, we have developed the Pests and Diseases Tree dataset
\(PDT dataset\). PDT dataset repre?sents the first high\-precision UAV\-based
dataset for targeted detection of tree pests and diseases, which is collected
in real\-world operational environments and aims to fill the gap in available
datasets for this field. Moreover, by aggregating public datasets and network
data, we further introduced the Common Weed and Crop dataset \(CWC dataset\) to
ad?dress the challenge of inadequate classification capabilities of test models
within datasets for this field. Finally, we propose the YOLO\-Dense Pest
\(YOLO\-DP\) model for high\-precision object detection of weed, pest, and disease
crop images. We re\-evaluate the state\-of\-the\-art detection models with our
proposed PDT dataset and CWC dataset, showing the completeness of the dataset
and the effectiveness of the YOLO\-DP. The proposed PDT dataset, CWC dataset,
and YOLO\-DP model are pre?sented at
https://github.com/RuiXing123/PDT\_CWC\_YOLO\-DP.


**代码链接**：https://github.com/RuiXing123/PDT_CWC_YOLO-DP.

**论文链接**：[阅读更多](http://arxiv.org/abs/2409.15679v1)

---


## Two Deep Learning Solutions for Automatic Blurring of Faces in Videos

**发布日期**：2024-09-23

**作者**：Roman Plaud

**摘要**：The widespread use of cameras in everyday life situations generates a vast
amount of data that may contain sensitive information about the people and
vehicles moving in front of them \(location, license plates, physical
characteristics, etc\). In particular, people's faces are recorded by
surveillance cameras in public spaces. In order to ensure the privacy of
individuals, face blurring techniques can be applied to the collected videos.
In this paper we present two deep\-learning based options to tackle the problem.
First, a direct approach, consisting of a classical object detector \(based on
the YOLO architecture\) trained to detect faces, which are subsequently blurred.
Second, an indirect approach, in which a Unet\-like segmentation network is
trained to output a version of the input image in which all the faces have been
blurred.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2409.14828v1)

---


## Real\-time Detection and Auto focusing of Beam Profiles from Silicon Photonics Gratings using YOLO model

**发布日期**：2024-09-22

**作者**：Yu Dian Lim

**摘要**：When observing the chip\-to\-free\-space light beams from silicon photonics
\(SiPh\) to free\-space, manual adjustment of camera lens is often required to
obtain a focused image of the light beams. In this letter, we demonstrated an
auto\-focusing system based on you\-only\-look\-once \(YOLO\) model. The trained YOLO
model exhibits high classification accuracy of 99.7% and high confidence level
>0.95 when detecting light beams from SiPh gratings. A video demonstration of
real\-time light beam detection, real\-time computation of beam width, and auto
focusing of light beams are also included.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2409.14413v1)

---


## Enhancing Fruit and Vegetable Detection in Unconstrained Environment with a Novel Dataset

**发布日期**：2024-09-20

**作者**：Sandeep Khanna

**摘要**：Automating the detection of fruits and vegetables using computer vision is
essential for modernizing agriculture, improving efficiency, ensuring food
quality, and contributing to technologically advanced and sustainable farming
practices. This paper presents an end\-to\-end pipeline for detecting and
localizing fruits and vegetables in real\-world scenarios. To achieve this, we
have curated a dataset named FRUVEG67 that includes images of 67 classes of
fruits and vegetables captured in unconstrained scenarios, with only a few
manually annotated samples per class. We have developed a semi\-supervised data
annotation algorithm \(SSDA\) that generates bounding boxes for objects to label
the remaining non\-annotated images. For detection, we introduce the Fruit and
Vegetable Detection Network \(FVDNet\), an ensemble version of YOLOv7 featuring
three distinct grid configurations. We employ an averaging approach for
bounding\-box prediction and a voting mechanism for class prediction. We have
integrated Jensen\-Shannon divergence \(JSD\) in conjunction with focal loss to
better detect smaller objects. Our experimental results highlight the
superiority of FVDNet compared to previous versions of YOLO, showcasing
remarkable improvements in detection and localization performance. We achieved
an impressive mean average precision \(mAP\) score of 0.78 across all classes.
Furthermore, we evaluated the efficacy of FVDNet using open\-category
refrigerator images, where it demonstrates promising results.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2409.13330v1)

---

