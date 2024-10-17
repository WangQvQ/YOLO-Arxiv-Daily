# 每日从arXiv中获取最新YOLO相关论文


## DocLayout\-YOLO: Enhancing Document Layout Analysis through Diverse Synthetic Data and Global\-to\-Local Adaptive Perception

**发布日期**：2024-10-16

**作者**：Zhiyuan Zhao

**摘要**：Document Layout Analysis is crucial for real\-world document understanding
systems, but it encounters a challenging trade\-off between speed and accuracy:
multimodal methods leveraging both text and visual features achieve higher
accuracy but suffer from significant latency, whereas unimodal methods relying
solely on visual features offer faster processing speeds at the expense of
accuracy. To address this dilemma, we introduce DocLayout\-YOLO, a novel
approach that enhances accuracy while maintaining speed advantages through
document\-specific optimizations in both pre\-training and model design. For
robust document pre\-training, we introduce the Mesh\-candidate BestFit
algorithm, which frames document synthesis as a two\-dimensional bin packing
problem, generating the large\-scale, diverse DocSynth\-300K dataset.
Pre\-training on the resulting DocSynth\-300K dataset significantly improves
fine\-tuning performance across various document types. In terms of model
optimization, we propose a Global\-to\-Local Controllable Receptive Module that
is capable of better handling multi\-scale variations of document elements.
Furthermore, to validate performance across different document types, we
introduce a complex and challenging benchmark named DocStructBench. Extensive
experiments on downstream datasets demonstrate that DocLayout\-YOLO excels in
both speed and accuracy. Code, data, and models are available at
https://github.com/opendatalab/DocLayout\-YOLO.


**代码链接**：https://github.com/opendatalab/DocLayout-YOLO.

**论文链接**：[阅读更多](http://arxiv.org/abs/2410.12628v1)

---


## Development of Image Collection Method Using YOLO and Siamese Network

**发布日期**：2024-10-16

**作者**：Chan Young Shin

**摘要**：As we enter the era of big data, collecting high\-quality data is very
important. However, collecting data by humans is not only very time\-consuming
but also expensive. Therefore, many scientists have devised various methods to
collect data using computers. Among them, there is a method called web
crawling, but the authors found that the crawling method has a problem in that
unintended data is collected along with the user. The authors found that this
can be filtered using the object recognition model YOLOv10. However, there are
cases where data that is not properly filtered remains. Here, image
reclassification was performed by additionally utilizing the distance output
from the Siamese network, and higher performance was recorded than other
classification models. \(average \\\_f1 score YOLO\+MobileNet
0.678\->YOLO\+SiameseNet 0.772\)\) The user can specify a distance threshold to
adjust the balance between data deficiency and noise\-robustness. The authors
also found that the Siamese network can achieve higher performance with fewer
resources because the cropped images are used for object recognition when
processing images in the Siamese network. \(Class 20 mean\-based f1 score,
non\-crop\+Siamese\(MobileNetV3\-Small\) 80.94 \-> crop
preprocessing\+Siamese\(MobileNetV3\-Small\) 82.31\) In this way, the image
retrieval system that utilizes two consecutive models to reduce errors can save
users' time and effort, and build better quality data faster and with fewer
resources than before.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2410.12561v1)

---


## YOLO\-ELA: Efficient Local Attention Modeling for High\-Performance Real\-Time Insulator Defect Detection

**发布日期**：2024-10-15

**作者**：Olalekan Akindele

**摘要**：Existing detection methods for insulator defect identification from unmanned
aerial vehicles \(UAV\) struggle with complex background scenes and small
objects, leading to suboptimal accuracy and a high number of false positives
detection. Using the concept of local attention modeling, this paper proposes a
new attention\-based foundation architecture, YOLO\-ELA, to address this issue.
The Efficient Local Attention \(ELA\) blocks were added into the neck part of the
one\-stage YOLOv8 architecture to shift the model's attention from background
features towards features of insulators with defects. The SCYLLA
Intersection\-Over\-Union \(SIoU\) criterion function was used to reduce detection
loss, accelerate model convergence, and increase the model's sensitivity
towards small insulator defects, yielding higher true positive outcomes. Due to
a limited dataset, data augmentation techniques were utilized to increase the
diversity of the dataset. In addition, we leveraged the transfer learning
strategy to improve the model's performance. Experimental results on
high\-resolution UAV images show that our method achieved a state\-of\-the\-art
performance of 96.9% mAP0.5 and a real\-time detection speed of 74.63 frames per
second, outperforming the baseline model. This further demonstrates the
effectiveness of attention\-based convolutional neural networks \(CNN\) in object
detection tasks.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2410.11727v1)

---


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


## ASTM :Autonomous Smart Traffic Management System Using Artificial Intelligence CNN and LSTM

**发布日期**：2024-10-14

**作者**：Christofel Rio Goenawan

**摘要**：In the modern world, the development of Artificial Intelligence \(AI\) has
contributed to improvements in various areas, including automation, computer
vision, fraud detection, and more. AI can be leveraged to enhance the
efficiency of Autonomous Smart Traffic Management \(ASTM\) systems and reduce
traffic congestion rates. This paper presents an Autonomous Smart Traffic
Management \(STM\) system that uses AI to improve traffic flow rates. The system
employs the YOLO V5 Convolutional Neural Network to detect vehicles in traffic
management images. Additionally, it predicts the number of vehicles for the
next 12 hours using a Recurrent Neural Network with Long Short\-Term Memory
\(RNN\-LSTM\). The Smart Traffic Management Cycle Length Analysis manages the
traffic cycle length based on these vehicle predictions, aided by AI. From the
results of the RNN\-LSTM model for predicting vehicle numbers over the next 12
hours, we observe that the model predicts traffic with a Mean Squared Error
\(MSE\) of 4.521 vehicles and a Root Mean Squared Error \(RMSE\) of 2.232 vehicles.
After simulating the STM system in the CARLA simulation environment, we found
that the Traffic Management Congestion Flow Rate with ASTM \(21 vehicles per
minute\) is 50\\% higher than the rate without STM \(around 15 vehicles per
minute\). Additionally, the Traffic Management Vehicle Pass Delay with STM \(5
seconds per vehicle\) is 70\\% lower than without STM \(around 12 seconds per
vehicle\). These results demonstrate that the STM system using AI can increase
traffic flow by 50\\% and reduce vehicle pass delays by 70\\%.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2410.10929v1)

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
have grown significantly at identifying objects through object detection. This
paper reviews the implementation of AI models for classifying trash through
object detection, specifically focusing on using YOLO V5 for training and
testing. The study demonstrates how YOLO V5 can effectively identify various
types of waste, including plastic, paper, glass, metal, cardboard, and
biodegradables.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2410.09975v2)

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


## ActNAS : Generating Efficient YOLO Models using Activation NAS

**发布日期**：2024-10-11

**作者**：Sudhakar Sah

**摘要**：Activation functions introduce non\-linearity into Neural Networks, enabling
them to learn complex patterns. Different activation functions vary in speed
and accuracy, ranging from faster but less accurate options like ReLU to slower
but more accurate functions like SiLU or SELU. Typically, same activation
function is used throughout an entire model architecture. In this paper, we
conduct a comprehensive study on the effects of using mixed activation
functions in YOLO\-based models, evaluating their impact on latency, memory
usage, and accuracy across CPU, NPU, and GPU edge devices. We also propose a
novel approach that leverages Neural Architecture Search \(NAS\) to design YOLO
models with optimized mixed activation functions.The best model generated
through this method demonstrates a slight improvement in mean Average Precision
\(mAP\) compared to baseline model \(SiLU\), while it is 22.28% faster and consumes
64.15% less memory on the reference NPU device.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2410.10887v1)

---

