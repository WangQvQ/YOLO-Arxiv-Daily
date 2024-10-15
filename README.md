# 每日从arXiv中获取最新YOLO相关论文


## Early Diagnoses of Acute Lymphoblastic Leukemia Using YOLOv8 and YOLOv11 Deep Learning Models

**发布日期**：2024-10-14

**作者**：Alaa Awad

**摘要**：Thousands of individuals succumb annually to leukemia alone. This study
explores the application of image processing and deep learning techniques for
detecting Acute Lymphoblastic Leukemia \(ALL\), a severe form of blood cancer
responsible for numerous annual fatalities. As artificial intelligence
technologies advance, the research investigates the reliability of these
methods in real\-world scenarios. The study focuses on recent developments in
ALL detection, particularly using the latest YOLO series models, to distinguish
between malignant and benign white blood cells and to identify different stages
of ALL, including early stages. Additionally, the models are capable of
detecting hematogones, which are often misclassified as ALL. By utilizing
advanced deep learning models like YOLOv8 and YOLOv11, the study achieves high
accuracy rates reaching 98.8%, demonstrating the effectiveness of these
algorithms across multiple datasets and various real\-world situations.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2410.10701v1)

---


## Words to Wheels: Vision\-Based Autonomous Driving Understanding Human Language Instructions Using Foundation Models

**发布日期**：2024-10-14

**作者**：Chanhoe Ryu

**摘要**：This paper introduces an innovative application of foundation models,
enabling Unmanned Ground Vehicles \(UGVs\) equipped with an RGB\-D camera to
navigate to designated destinations based on human language instructions.
Unlike learning\-based methods, this approach does not require prior training
but instead leverages existing foundation models, thus facilitating
generalization to novel environments. Upon receiving human language
instructions, these are transformed into a 'cognitive route description' using
a large language model \(LLM\)\-a detailed navigation route expressed in human
language. The vehicle then decomposes this description into landmarks and
navigation maneuvers. The vehicle also determines elevation costs and
identifies navigability levels of different regions through a terrain
segmentation model, GANav, trained on open datasets. Semantic elevation costs,
which take both elevation and navigability levels into account, are estimated
and provided to the Model Predictive Path Integral \(MPPI\) planner, responsible
for local path planning. Concurrently, the vehicle searches for target
landmarks using foundation models, including YOLO\-World and EfficientViT\-SAM.
Ultimately, the vehicle executes the navigation commands to reach the
designated destination, the final landmark. Our experiments demonstrate that
this application successfully guides UGVs to their destinations following human
language instructions in novel environments, such as unfamiliar terrain or
urban settings.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2410.10577v1)

---


## Innovative Deep Learning Techniques for Obstacle Recognition: A Comparative Study of Modern Detection Algorithms

**发布日期**：2024-10-14

**作者**：Santiago Pérez

**摘要**：This study explores a comprehensive approach to obstacle detection using
advanced YOLO models, specifically YOLOv8, YOLOv7, YOLOv6, and YOLOv5.
Leveraging deep learning techniques, the research focuses on the performance
comparison of these models in real\-time detection scenarios. The findings
demonstrate that YOLOv8 achieves the highest accuracy with improved
precision\-recall metrics. Detailed training processes, algorithmic principles,
and a range of experimental results are presented to validate the model's
effectiveness.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2410.10096v1)

---


## Optimizing Waste Management with Advanced Object Detection for Garbage Classification

**发布日期**：2024-10-13

**作者**：Everest Z. Kuang

**摘要**：Garbage production and littering are persistent global issues that pose
significant environmental challenges. Despite large\-scale efforts to manage
waste through collection and sorting, existing approaches remain inefficient,
leading to inadequate recycling and disposal. Therefore, developing advanced
AI\-based systems is less labor intensive approach for addressing the growing
waste problem more effectively. These models can be applied to sorting systems
or possibly waste collection robots that may produced in the future. AI models
have grown significantly at identifying objects through object detection.This
paper reviews the implementation of AI models for classifying trash through
object detection, specifically focusing on the use of YOLO V5 for training and
testing. The study demonstrates how YOLO V5 can effectively identify various
types of waste, including \\textit\{plastic\}, \\textit\{paper\}, \\textit\{glass\},
\\textit\{metal\}, \\textit\{cardboard\}, and \\textit\{biodegradables\}\}.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2410.09975v1)

---


## Token Pruning using a Lightweight Background Aware Vision Transformer

**发布日期**：2024-10-12

**作者**：Sudhakar Sah

**摘要**：High runtime memory and high latency puts significant constraint on Vision
Transformer training and inference, especially on edge devices. Token pruning
reduces the number of input tokens to the ViT based on importance criteria of
each token. We present a Background Aware Vision Transformer \(BAViT\) model, a
pre\-processing block to object detection models like DETR/YOLOS aimed to reduce
runtime memory and increase throughput by using a novel approach to identify
background tokens in the image. The background tokens can be pruned completely
or partially before feeding to a ViT based object detector. We use the semantic
information provided by segmentation map and/or bounding box annotation to
train a few layers of ViT to classify tokens to either foreground or
background. Using 2 layers and 10 layers of BAViT, background and foreground
tokens can be separated with 75% and 88% accuracy on VOC dataset and 71% and
80% accuracy on COCO dataset respectively. We show a 2 layer BAViT\-small model
as pre\-processor to YOLOS can increase the throughput by 30% \- 40% with a mAP
drop of 3% without any sparse fine\-tuning and 2% with sparse fine\-tuning. Our
approach is specifically targeted for Edge AI use cases.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2410.09324v1)

---


## Optimizing YOLO Architectures for Optimal Road Damage Detection and Classification: A Comparative Study from YOLOv7 to YOLOv10

**发布日期**：2024-10-10

**作者**：Vung Pham

**摘要**：Maintaining roadway infrastructure is essential for ensuring a safe,
efficient, and sustainable transportation system. However, manual data
collection for detecting road damage is time\-consuming, labor\-intensive, and
poses safety risks. Recent advancements in artificial intelligence,
particularly deep learning, offer a promising solution for automating this
process using road images. This paper presents a comprehensive workflow for
road damage detection using deep learning models, focusing on optimizations for
inference speed while preserving detection accuracy. Specifically, to
accommodate hardware limitations, large images are cropped, and lightweight
models are utilized. Additionally, an external pothole dataset is incorporated
to enhance the detection of this underrepresented damage class. The proposed
approach employs multiple model architectures, including a custom YOLOv7 model
with Coordinate Attention layers and a Tiny YOLOv7 model, which are trained and
combined to maximize detection performance. The models are further
reparameterized to optimize inference efficiency. Experimental results
demonstrate that the ensemble of the custom YOLOv7 model with three Coordinate
Attention layers and the default Tiny YOLOv7 model achieves an F1 score of
0.7027 with an inference speed of 0.0547 seconds per image. The complete
pipeline, including data preprocessing, model training, and inference scripts,
is publicly available on the project's GitHub repository, enabling
reproducibility and facilitating further research.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2410.08409v1)

---


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

