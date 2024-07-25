# 每日从arXiv中获取最新YOLO相关论文


## RT\-DETRv2: Improved Baseline with Bag\-of\-Freebies for Real\-Time Detection Transformer

**发布日期**：2024-07-24

**作者**：Wenyu Lv

**摘要**：In this report, we present RT\-DETRv2, an improved Real\-Time DEtection
TRansformer \(RT\-DETR\). RT\-DETRv2 builds upon the previous state\-of\-the\-art
real\-time detector, RT\-DETR, and opens up a set of bag\-of\-freebies for
flexibility and practicality, as well as optimizing the training strategy to
achieve enhanced performance. To improve the flexibility, we suggest setting a
distinct number of sampling points for features at different scales in the
deformable attention to achieve selective multi\-scale feature extraction by the
decoder. To enhance practicality, we propose an optional discrete sampling
operator to replace the grid\_sample operator that is specific to RT\-DETR
compared to YOLOs. This removes the deployment constraints typically associated
with DETRs. For the training strategy, we propose dynamic data augmentation and
scale\-adaptive hyperparameters customization to improve performance without
loss of speed. Source code and pre\-trained models will be available at
https://github.com/lyuwenyu/RT\-DETR.


**代码链接**：https://github.com/lyuwenyu/RT-DETR.

**论文链接**：[阅读更多](http://arxiv.org/abs/2407.17140v1)

---


## YOLOv10 for Automated Fracture Detection in Pediatric Wrist Trauma X\-rays

**发布日期**：2024-07-22

**作者**：Ammar Ahmed

**摘要**：Wrist fractures are highly prevalent among children and can significantly
impact their daily activities, such as attending school, participating in
sports, and performing basic self\-care tasks. If not treated properly, these
fractures can result in chronic pain, reduced wrist functionality, and other
long\-term complications. Recently, advancements in object detection have shown
promise in enhancing fracture detection, with systems achieving accuracy
comparable to, or even surpassing, that of human radiologists. The YOLO series,
in particular, has demonstrated notable success in this domain. This study is
the first to provide a thorough evaluation of various YOLOv10 variants to
assess their performance in detecting pediatric wrist fractures using the
GRAZPEDWRI\-DX dataset. It investigates how changes in model complexity, scaling
the architecture, and implementing a dual\-label assignment strategy can enhance
detection performance. Experimental results indicate that our trained model
achieved mean average precision \(mAP@50\-95\) of 51.9\\% surpassing the current
YOLOv9 benchmark of 43.3\\% on this dataset. This represents an improvement of
8.6\\%. The implementation code is publicly available at
https://github.com/ammarlodhi255/YOLOv10\-Fracture\-Detection


**代码链接**：https://github.com/ammarlodhi255/YOLOv10-Fracture-Detection

**论文链接**：[阅读更多](http://arxiv.org/abs/2407.15689v1)

---


## Multiple Object Detection and Tracking in Panoramic Videos for Cycling Safety Analysis

**发布日期**：2024-07-21

**作者**：Jingwei Guo

**摘要**：Panoramic cycling videos can record 360\{\\deg\} views around the cyclists.
Thus, it is essential to conduct automatic road user analysis on them using
computer vision models to provide data for studies on cycling safety. However,
the features of panoramic data such as severe distortions, large number of
small objects and boundary continuity have brought great challenges to the
existing CV models, including poor performance and evaluation methods that are
no longer applicable. In addition, due to the lack of data with annotations, it
is not easy to re\-train the models.
  In response to these problems, the project proposed and implemented a
three\-step methodology: \(1\) improve the prediction performance of the
pre\-trained object detection models on panoramic data by projecting the
original image into 4 perspective sub\-images; \(2\) introduce supports for
boundary continuity and category information into DeepSORT, a commonly used
multiple object tracking model, and set an improved detection model as its
detector; \(3\) using the tracking results, develop an application for detecting
the overtaking behaviour of the surrounding vehicles.
  Evaluated on the panoramic cycling dataset built by the project, the proposed
methodology improves the average precision of YOLO v5m6 and Faster RCNN\-FPN
under any input resolution setting. In addition, it raises MOTA and IDF1 of
DeepSORT by 7.6\\% and 9.7\\% respectively. When detecting the overtakes in the
test videos, it achieves the F\-score of 0.88.
  The code is available on GitHub at github.com/cuppp1998/360\_object\_tracking
to ensure the reproducibility and further improvements of results.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2407.15199v1)

---


## GreenStableYolo: Optimizing Inference Time and Image Quality of Text\-to\-Image Generation

**发布日期**：2024-07-20

**作者**：Jingzhi Gong

**摘要**：Tuning the parameters and prompts for improving AI\-based text\-to\-image
generation has remained a substantial yet unaddressed challenge. Hence we
introduce GreenStableYolo, which improves the parameters and prompts for Stable
Diffusion to both reduce GPU inference time and increase image generation
quality using NSGA\-II and Yolo.
  Our experiments show that despite a relatively slight trade\-off \(18%\) in
image quality compared to StableYolo \(which only considers image quality\),
GreenStableYolo achieves a substantial reduction in inference time \(266% less\)
and a 526% higher hypervolume, thereby advancing the state\-of\-the\-art for
text\-to\-image generation.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2407.14982v1)

---


## Enhancing Layout Hotspot Detection Efficiency with YOLOv8 and PCA\-Guided Augmentation

**发布日期**：2024-07-19

**作者**：Dongyang Wu

**摘要**：In this paper, we present a YOLO\-based framework for layout hotspot
detection, aiming to enhance the efficiency and performance of the design rule
checking \(DRC\) process. Our approach leverages the YOLOv8 vision model to
detect multiple hotspots within each layout image, even when dealing with large
layout image sizes. Additionally, to enhance pattern\-matching effectiveness, we
introduce a novel approach to augment the layout image using information
extracted through Principal Component Analysis \(PCA\). The core of our proposed
method is an algorithm that utilizes PCA to extract valuable auxiliary
information from the layout image. This extracted information is then
incorporated into the layout image as an additional color channel. This
augmentation significantly improves the accuracy of multi\-hotspot detection
while reducing the false alarm rate of the object detection algorithm. We
evaluate the effectiveness of our framework using four datasets generated from
layouts found in the ICCAD\-2019 benchmark dataset. The results demonstrate that
our framework achieves a precision \(recall\) of approximately 83% \(86%\) while
maintaining a false alarm rate of less than 7.4\\%. Also, the studies show that
the proposed augmentation approach could improve the detection ability of
never\-seen\-before \(NSB\) hotspots by about 10%.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2407.14498v1)

---


## CerberusDet: Unified Multi\-Task Object Detection

**发布日期**：2024-07-17

**作者**：Irina Tolstykh

**摘要**：Object detection is a core task in computer vision. Over the years, the
development of numerous models has significantly enhanced performance. However,
these conventional models are usually limited by the data on which they were
trained and by the category logic they define. With the recent rise of
Language\-Visual Models, new methods have emerged that are not restricted to
these fixed categories. Despite their flexibility, such Open Vocabulary
detection models still fall short in accuracy compared to traditional models
with fixed classes. At the same time, more accurate data\-specific models face
challenges when there is a need to extend classes or merge different datasets
for training. The latter often cannot be combined due to different logics or
conflicting class definitions, making it difficult to improve a model without
compromising its performance. In this paper, we introduce CerberusDet, a
framework with a multi\-headed model designed for handling multiple object
detection tasks. Proposed model is built on the YOLO architecture and
efficiently shares visual features from both backbone and neck components,
while maintaining separate task heads. This approach allows CerberusDet to
perform very efficiently while still delivering optimal results. We evaluated
the model on the PASCAL VOC dataset and additional categories from the
Objects365 dataset to demonstrate its abilities. CerberusDet achieved results
comparable to state\-of\-the\-art data\-specific models with 36% less inference
time. The more tasks are trained together, the more efficient the proposed
model becomes compared to running individual models sequentially. The training
and inference code, as well as the model, are available as open\-source
\(https://github.com/ai\-forever/CerberusDet\).


**代码链接**：https://github.com/ai-forever/CerberusDet).

**论文链接**：[阅读更多](http://arxiv.org/abs/2407.12632v1)

---


## Enhancing Wrist Abnormality Detection with YOLO: Analysis of State\-of\-the\-art Single\-stage Detection Models

**发布日期**：2024-07-17

**作者**：Ammar Ahmed

**摘要**：Diagnosing and treating abnormalities in the wrist, specifically distal
radius, and ulna fractures, is a crucial concern among children, adolescents,
and young adults, with a higher incidence rate during puberty. However, the
scarcity of radiologists and the lack of specialized training among medical
professionals pose a significant risk to patient care. This problem is further
exacerbated by the rising number of imaging studies and limited access to
specialist reporting in certain regions. This highlights the need for
innovative solutions to improve the diagnosis and treatment of wrist
abnormalities. Automated wrist fracture detection using object detection has
shown potential, but current studies mainly use two\-stage detection methods
with limited evidence for single\-stage effectiveness. This study employs
state\-of\-the\-art single\-stage deep neural network\-based detection models
YOLOv5, YOLOv6, YOLOv7, and YOLOv8 to detect wrist abnormalities. Through
extensive experimentation, we found that these YOLO models outperform the
commonly used two\-stage detection algorithm, Faster R\-CNN, in bone fracture
detection. Additionally, compound\-scaled variants of each YOLO model were
compared, with YOLOv8x demonstrating a fracture detection mean average
precision \(mAP\) of 0.95 and an overall mAP of 0.77 on the GRAZPEDWRI\-DX
pediatric wrist dataset, highlighting the potential of single\-stage models for
enhancing pediatric wrist imaging.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2407.12597v1)

---


## Melon Fruit Detection and Quality Assessment Using Generative AI\-Based Image Data Augmentation

**发布日期**：2024-07-15

**作者**：Seungri Yoon

**摘要**：Monitoring and managing the growth and quality of fruits are very important
tasks. To effectively train deep learning models like YOLO for real\-time fruit
detection, high\-quality image datasets are essential. However, such datasets
are often lacking in agriculture. Generative AI models can help create
high\-quality images. In this study, we used MidJourney and Firefly tools to
generate images of melon greenhouses and post\-harvest fruits through
text\-to\-image, pre\-harvest image\-to\-image, and post\-harvest image\-to\-image
methods. We evaluated these AIgenerated images using PSNR and SSIM metrics and
tested the detection performance of the YOLOv9 model. We also assessed the net
quality of real and generated fruits. Our results showed that generative AI
could produce images very similar to real ones, especially for post\-harvest
fruits. The YOLOv9 model detected the generated images well, and the net
quality was also measurable. This shows that generative AI can create realistic
images useful for fruit detection and quality assessment, indicating its great
potential in agriculture. This study highlights the potential of AI\-generated
images for data augmentation in melon fruit detection and quality assessment
and envisions a positive future for generative AI applications in agriculture.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2407.10413v1)

---


## DART: An Automated End\-to\-End Object Detection Pipeline with Data Diversification, Open\-Vocabulary Bounding Box Annotation, Pseudo\-Label Review, and Model Training

**发布日期**：2024-07-12

**作者**：Chen Xin

**摘要**：Swift and accurate detection of specified objects is crucial for many
industrial applications, such as safety monitoring on construction sites.
However, traditional approaches rely heavily on arduous manual annotation and
data collection, which struggle to adapt to ever\-changing environments and
novel target objects. To address these limitations, this paper presents DART,
an automated end\-to\-end pipeline designed to streamline the entire workflow of
an object detection application from data collection to model deployment. DART
eliminates the need for human labeling and extensive data collection while
excelling in diverse scenarios. It employs a subject\-driven image generation
module \(DreamBooth with SDXL\) for data diversification, followed by an
annotation stage where open\-vocabulary object detection \(Grounding DINO\)
generates bounding box annotations for both generated and original images.
These pseudo\-labels are then reviewed by a large multimodal model \(GPT\-4o\) to
guarantee credibility before serving as ground truth to train real\-time object
detectors \(YOLO\). We apply DART to a self\-collected dataset of construction
machines named Liebherr Product, which contains over 15K high\-quality images
across 23 categories. The current implementation of DART significantly
increases average precision \(AP\) from 0.064 to 0.832. Furthermore, we adopt a
modular design for DART to ensure easy exchangeability and extensibility. This
allows for a smooth transition to more advanced algorithms in the future,
seamless integration of new object categories without manual labeling, and
adaptability to customized environments without extra data collection. The code
and dataset are released at https://github.com/chen\-xin\-94/DART.


**代码链接**：https://github.com/chen-xin-94/DART.

**论文链接**：[阅读更多](http://arxiv.org/abs/2407.09174v1)

---


## PowerYOLO: Mixed Precision Model for Hardware Efficient Object Detection with Event Data

**发布日期**：2024-07-11

**作者**：Dominika Przewlocka\-Rus

**摘要**：The performance of object detection systems in automotive solutions must be
as high as possible, with minimal response time and, due to the often
battery\-powered operation, low energy consumption. When designing such
solutions, we therefore face challenges typical for embedded vision systems:
the problem of fitting algorithms of high memory and computational complexity
into small low\-power devices. In this paper we propose PowerYOLO \- a mixed
precision solution, which targets three essential elements of such application.
First, we propose a system based on a Dynamic Vision Sensor \(DVS\), a novel
sensor, that offers low power requirements and operates well in conditions with
variable illumination. It is these features that may make event cameras a
preferential choice over frame cameras in some applications. Second, to ensure
high accuracy and low memory and computational complexity, we propose to use
4\-bit width Powers\-of\-Two \(PoT\) quantisation for convolution weights of the
YOLO detector, with all other parameters quantised linearly. Finally, we
embrace from PoT scheme and replace multiplication with bit\-shifting to
increase the efficiency of hardware acceleration of such solution, with a
special convolution\-batch normalisation fusion scheme. The use of specific
sensor with PoT quantisation and special batch normalisation fusion leads to a
unique system with almost 8x reduction in memory complexity and vast
computational simplifications, with relation to a standard approach. This
efficient system achieves high accuracy of mAP 0.301 on the GEN1 DVS dataset,
marking the new state\-of\-the\-art for such compressed model.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2407.08272v1)

---

