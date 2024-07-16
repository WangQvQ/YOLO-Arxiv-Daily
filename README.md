# 每日从arXiv中获取最新YOLO相关论文


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

