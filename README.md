# 每日从arXiv中获取最新YOLO相关论文


## SurgeoNet: Realtime 3D Pose Estimation of Articulated Surgical Instruments from Stereo Images using a Synthetically\-trained Network

**发布日期**：2024-10-02

**作者**：Ahmed Tawfik Aboukhadra

**摘要**：Surgery monitoring in Mixed Reality \(MR\) environments has recently received
substantial focus due to its importance in image\-based decisions, skill
assessment, and robot\-assisted surgery. Tracking hands and articulated surgical
instruments is crucial for the success of these applications. Due to the lack
of annotated datasets and the complexity of the task, only a few works have
addressed this problem. In this work, we present SurgeoNet, a real\-time neural
network pipeline to accurately detect and track surgical instruments from a
stereo VR view. Our multi\-stage approach is inspired by state\-of\-the\-art
neural\-network architectural design, like YOLO and Transformers. We demonstrate
the generalization capabilities of SurgeoNet in challenging real\-world
scenarios, achieved solely through training on synthetic data. The approach can
be easily extended to any new set of articulated surgical instruments.
SurgeoNet's code and data are publicly available.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2410.01293v1)

---


## Optimizing Drug Delivery in Smart Pharmacies: A Novel Framework of Multi\-Stage Grasping Network Combined with Adaptive Robotics Mechanism

**发布日期**：2024-10-01

**作者**：Rui Tang

**摘要**：Robots\-based smart pharmacies are essential for modern healthcare systems,
enabling efficient drug delivery. However, a critical challenge exists in the
robotic handling of drugs with varying shapes and overlapping positions, which
previous studies have not adequately addressed. To enhance the robotic arm's
ability to grasp chaotic, overlapping, and variously shaped drugs, this paper
proposed a novel framework combining a multi\-stage grasping network with an
adaptive robotics mechanism. The framework first preprocessed images using an
improved Super\-Resolution Convolutional Neural Network \(SRCNN\) algorithm, and
then employed the proposed YOLOv5\+E\-A\-SPPFCSPC\+BIFPNC \(YOLO\-EASB\) instance
segmentation algorithm for precise drug segmentation. The most suitable drugs
for grasping can be determined by assessing the completeness of the
segmentation masks. Then, these segmented drugs were processed by our improved
Adaptive Feature Fusion and Grasp\-Aware Network \(IAFFGA\-Net\) with the optimized
loss function, which ensures accurate picking actions even in complex
environments. To control the robot grasping, a time\-optimal robotic arm
trajectory planning algorithm that combines an improved ant colony algorithm
with 3\-5\-3 interpolation was developed, further improving efficiency while
ensuring smooth trajectories. Finally, this system was implemented and
validated within an adaptive collaborative robot setup, which dynamically
adjusts to different production environments and task requirements.
Experimental results demonstrate the superiority of our multi\-stage grasping
network in optimizing smart pharmacy operations, while also showcasing its
remarkable adaptability and effectiveness in practical applications.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2410.00753v1)

---


## Drone Stereo Vision for Radiata Pine Branch Detection and Distance Measurement: Utilizing Deep Learning and YOLO Integration

**发布日期**：2024-10-01

**作者**：Yida Lin

**摘要**：This research focuses on the development of a drone equipped with pruning
tools and a stereo vision camera to accurately detect and measure the spatial
positions of tree branches. YOLO is employed for branch segmentation, while two
depth estimation approaches, monocular and stereo, are investigated. In
comparison to SGBM, deep learning techniques produce more refined and accurate
depth maps. In the absence of ground\-truth data, a fine\-tuning process using
deep neural networks is applied to approximate optimal depth values. This
methodology facilitates precise branch detection and distance measurement,
addressing critical challenges in the automation of pruning operations. The
results demonstrate notable advancements in both accuracy and efficiency,
underscoring the potential of deep learning to drive innovation and enhance
automation in the agricultural sector.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2410.00503v1)

---


## MCUBench: A Benchmark of Tiny Object Detectors on MCUs

**发布日期**：2024-09-27

**作者**：Sudhakar Sah

**摘要**：We introduce MCUBench, a benchmark featuring over 100 YOLO\-based object
detection models evaluated on the VOC dataset across seven different MCUs. This
benchmark provides detailed data on average precision, latency, RAM, and Flash
usage for various input resolutions and YOLO\-based one\-stage detectors. By
conducting a controlled comparison with a fixed training pipeline, we collect
comprehensive performance metrics. Our Pareto\-optimal analysis shows that
integrating modern detection heads and training techniques allows various YOLO
architectures, including legacy models like YOLOv3, to achieve a highly
efficient tradeoff between mean Average Precision \(mAP\) and latency. MCUBench
serves as a valuable tool for benchmarking the MCU performance of contemporary
object detectors and aids in model selection based on specific constraints.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2409.18866v1)

---


## YOLOv8\-ResCBAM: YOLOv8 Based on An Effective Attention Module for Pediatric Wrist Fracture Detection

**发布日期**：2024-09-27

**作者**：Rui\-Yang Ju

**摘要**：Wrist trauma and even fractures occur frequently in daily life, particularly
among children who account for a significant proportion of fracture cases.
Before performing surgery, surgeons often request patients to undergo X\-ray
imaging first, and prepare for the surgery based on the analysis of the X\-ray
images. With the development of neural networks, You Only Look Once \(YOLO\)
series models have been widely used in fracture detection for Computer\-Assisted
Diagnosis, where the YOLOv8 model has obtained the satisfactory results.
Applying the attention modules to neural networks is one of the effective
methods to improve the model performance. This paper proposes YOLOv8\-ResCBAM,
which incorporates Convolutional Block Attention Module integrated with
resblock \(ResCBAM\) into the original YOLOv8 network architecture. The
experimental results on the GRAZPEDWRI\-DX dataset demonstrate that the mean
Average Precision calculated at Intersection over Union threshold of 0.5 \(mAP
50\) of the proposed model increased from 63.6% of the original YOLOv8 model to
65.8%, which achieves the state\-of\-the\-art performance. The implementation code
is available at
https://github.com/RuiyangJu/Fracture\_Detection\_Improved\_YOLOv8.


**代码链接**：https://github.com/RuiyangJu/Fracture_Detection_Improved_YOLOv8.

**论文链接**：[阅读更多](http://arxiv.org/abs/2409.18826v1)

---


## Drone Stereo Vision for Radiata Pine Branch Detection and Distance Measurement: Integrating SGBM and Segmentation Models

**发布日期**：2024-09-26

**作者**：Yida Lin

**摘要**：Manual pruning of radiata pine trees presents significant safety risks due to
their substantial height and the challenging terrains in which they thrive. To
address these risks, this research proposes the development of a drone\-based
pruning system equipped with specialized pruning tools and a stereo vision
camera, enabling precise detection and trimming of branches. Deep learning
algorithms, including YOLO and Mask R\-CNN, are employed to ensure accurate
branch detection, while the Semi\-Global Matching algorithm is integrated to
provide reliable distance estimation. The synergy between these techniques
facilitates the precise identification of branch locations and enables
efficient, targeted pruning. Experimental results demonstrate that the combined
implementation of YOLO and SGBM enables the drone to accurately detect branches
and measure their distances from the drone. This research not only improves the
safety and efficiency of pruning operations but also makes a significant
contribution to the advancement of drone technology in the automation of
agricultural and forestry practices, laying a foundational framework for
further innovations in environmental management.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2409.17526v1)

---


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

**论文链接**：[阅读更多](http://arxiv.org/abs/2409.17122v2)

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

**论文链接**：[阅读更多](http://arxiv.org/abs/2409.16205v2)

---

