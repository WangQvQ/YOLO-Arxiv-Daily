# 每日从arXiv中获取最新YOLO相关论文


## Robust infrared small target detection using self\-supervised and a contrario paradigms

**发布日期**：2024-10-09

**作者**：Alina Ciocarlan

**摘要**：Detecting small targets in infrared images poses significant challenges in
defense applications due to the presence of complex backgrounds and the small
size of the targets. Traditional object detection methods often struggle to
balance high detection rates with low false alarm rates, especially when
dealing with small objects. In this paper, we introduce a novel approach that
combines a contrario paradigm with Self\-Supervised Learning \(SSL\) to improve
Infrared Small Target Detection \(IRSTD\). On the one hand, the integration of an
a contrario criterion into a YOLO detection head enhances feature map responses
for small and unexpected objects while effectively controlling false alarms. On
the other hand, we explore SSL techniques to overcome the challenges of limited
annotated data, common in IRSTD tasks. Specifically, we benchmark several
representative SSL strategies for their effectiveness in improving small object
detection performance. Our findings show that instance discrimination methods
outperform masked image modeling strategies when applied to YOLO\-based small
object detection. Moreover, the combination of the a contrario and SSL
paradigms leads to significant performance improvements, narrowing the gap with
state\-of\-the\-art segmentation methods and even outperforming them in frugal
settings. This two\-pronged approach offers a robust solution for improving
IRSTD performance, particularly under challenging conditions.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2410.07437v1)

---


## Iterative Optimization Annotation Pipeline and ALSS\-YOLO\-Seg for Efficient Banana Plantation Segmentation in UAV Imagery

**发布日期**：2024-10-09

**作者**：Ang He

**摘要**：Precise segmentation of Unmanned Aerial Vehicle \(UAV\)\-captured images plays a
vital role in tasks such as crop yield estimation and plant health assessment
in banana plantations. By identifying and classifying planted areas, crop area
can be calculated, which is indispensable for accurate yield predictions.
However, segmenting banana plantation scenes requires a substantial amount of
annotated data, and manual labeling of these images is both time\-consuming and
labor\-intensive, limiting the development of large\-scale datasets. Furthermore,
challenges such as changing target sizes, complex ground backgrounds, limited
computational resources, and correct identification of crop categories make
segmentation even more difficult. To address these issues, we proposed a
comprehensive solution. Firstly, we designed an iterative optimization
annotation pipeline leveraging SAM2's zero\-shot capabilities to generate
high\-quality segmentation annotations, thereby reducing the cost and time
associated with data annotation significantly. Secondly, we developed
ALSS\-YOLO\-Seg, an efficient lightweight segmentation model optimized for UAV
imagery. The model's backbone includes an Adaptive Lightweight Channel
Splitting and Shuffling \(ALSS\) module to improve information exchange between
channels and optimize feature extraction, aiding accurate crop identification.
Additionally, a Multi\-Scale Channel Attention \(MSCA\) module combines
multi\-scale feature extraction with channel attention to tackle challenges of
varying target sizes and complex ground backgrounds.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2410.07955v1)

---


## Human\-in\-the\-loop Reasoning For Traffic Sign Detection: Collaborative Approach Yolo With Video\-llava

**发布日期**：2024-10-07

**作者**：Mehdi Azarafza

**摘要**：Traffic Sign Recognition \(TSR\) detection is a crucial component of autonomous
vehicles. While You Only Look Once \(YOLO\) is a popular real\-time object
detection algorithm, factors like training data quality and adverse weather
conditions \(e.g., heavy rain\) can lead to detection failures. These failures
can be particularly dangerous when visual similarities between objects exist,
such as mistaking a 30 km/h sign for a higher speed limit sign. This paper
proposes a method that combines video analysis and reasoning, prompting with a
human\-in\-the\-loop guide large vision model to improve YOLOs accuracy in
detecting road speed limit signs, especially in semi\-real\-world conditions. It
is hypothesized that the guided prompting and reasoning abilities of
Video\-LLava can enhance YOLOs traffic sign detection capabilities. This
hypothesis is supported by an evaluation based on human\-annotated accuracy
metrics within a dataset of recorded videos from the CARLA car simulator. The
results demonstrate that a collaborative approach combining YOLO with
Video\-LLava and reasoning can effectively address challenging situations such
as heavy rain and overcast conditions that hinder YOLOs detection capabilities.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2410.05096v1)

---


## YOLO\-MARL: You Only LLM Once for Multi\-agent Reinforcement Learning

**发布日期**：2024-10-05

**作者**：Yuan Zhuang

**摘要**：Advancements in deep multi\-agent reinforcement learning \(MARL\) have
positioned it as a promising approach for decision\-making in cooperative games.
However, it still remains challenging for MARL agents to learn cooperative
strategies for some game environments. Recently, large language models \(LLMs\)
have demonstrated emergent reasoning capabilities, making them promising
candidates for enhancing coordination among the agents. However, due to the
model size of LLMs, it can be expensive to frequently infer LLMs for actions
that agents can take. In this work, we propose You Only LLM Once for MARL
\(YOLO\-MARL\), a novel framework that leverages the high\-level task planning
capabilities of LLMs to improve the policy learning process of multi\-agents in
cooperative games. Notably, for each game environment, YOLO\-MARL only requires
one time interaction with LLMs in the proposed strategy generation, state
interpretation and planning function generation modules, before the MARL policy
training process. This avoids the ongoing costs and computational time
associated with frequent LLMs API calls during training. Moreover, the trained
decentralized normal\-sized neural network\-based policies operate independently
of the LLM. We evaluate our method across three different environments and
demonstrate that YOLO\-MARL outperforms traditional MARL algorithms.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2410.03997v1)

---


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

**论文链接**：[阅读更多](http://arxiv.org/abs/2410.00503v2)

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

