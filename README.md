# 每日从arXiv中获取最新YOLO相关论文


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


## Enrich the content of the image Using Context\-Aware Copy Paste

**发布日期**：2024-07-11

**作者**：Qiushi Guo

**摘要**：Data augmentation remains a widely utilized technique in deep learning,
particularly in tasks such as image classification, semantic segmentation, and
object detection. Among them, Copy\-Paste is a simple yet effective method and
gain great attention recently. However, existing Copy\-Paste often overlook
contextual relevance between source and target images, resulting in
inconsistencies in generated outputs. To address this challenge, we propose a
context\-aware approach that integrates Bidirectional Latent Information
Propagation \(BLIP\) for content extraction from source images. By matching
extracted content information with category information, our method ensures
cohesive integration of target objects using Segment Anything Model \(SAM\) and
You Only Look Once \(YOLO\). This approach eliminates the need for manual
annotation, offering an automated and user\-friendly solution. Experimental
evaluations across diverse datasets demonstrate the effectiveness of our method
in enhancing data diversity and generating high\-quality pseudo\-images across
various computer vision tasks.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2407.08151v1)

---


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

