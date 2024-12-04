# 每日从arXiv中获取最新YOLO相关论文


## Smart Parking with Pixel\-Wise ROI Selection for Vehicle Detection Using YOLOv8, YOLOv9, YOLOv10, and YOLOv11

**发布日期**：2024-12-02

**作者**：Gustavo P. C. P. da Luz

**摘要**：The increasing urbanization and the growing number of vehicles in cities have
underscored the need for efficient parking management systems. Traditional
smart parking solutions often rely on sensors or cameras for occupancy
detection, each with its limitations. Recent advancements in deep learning have
introduced new YOLO models \(YOLOv8, YOLOv9, YOLOv10, and YOLOv11\), but these
models have not been extensively evaluated in the context of smart parking
systems, particularly when combined with Region of Interest \(ROI\) selection for
object detection. Existing methods still rely on fixed polygonal ROI selections
or simple pixel\-based modifications, which limit flexibility and precision.
This work introduces a novel approach that integrates Internet of Things, Edge
Computing, and Deep Learning concepts, by using the latest YOLO models for
vehicle detection. By exploring both edge and cloud computing, it was found
that inference times on edge devices ranged from 1 to 92 seconds, depending on
the hardware and model version. Additionally, a new pixel\-wise post\-processing
ROI selection method is proposed for accurately identifying regions of interest
to count vehicles in parking lot images. The proposed system achieved 99.68%
balanced accuracy on a custom dataset of 3,484 images, offering a
cost\-effective smart parking solution that ensures precise vehicle detection
while preserving data privacy


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2412.01983v1)

---


## Research on Cervical Cancer p16/Ki\-67 Immunohistochemical Dual\-Staining Image Recognition Algorithm Based on YOLO

**发布日期**：2024-12-02

**作者**：Xiao\-Jun Wu

**摘要**：The p16/Ki\-67 dual staining method is a new approach for cervical cancer
screening with high sensitivity and specificity. However, there are issues of
mis\-detection and inaccurate recognition when the YOLOv5s algorithm is directly
applied to dual\-stained cell images. This paper Proposes a novel cervical
cancer dual\-stained image recognition \(DSIR\-YOLO\) model based on an YOLOv5. By
fusing the Swin\-Transformer module, GAM attention mechanism, multi\-scale
feature fusion, and EIoU loss function, the detection performance is
significantly improved, with mAP@0.5 and mAP@0.5:0.95 reaching 92.6% and 70.5%,
respectively. Compared with YOLOv5s in five\-fold cross\-validation, the
accuracy, recall, mAP@0.5, and mAP@0.5:0.95 of the improved algorithm are
increased by 2.3%, 4.1%, 4.3%, and 8.0%, respectively, with smaller variances
and higher stability. Compared with other detection algorithms, DSIR\-YOLO in
this paper sacrifices some performance requirements to improve the network
recognition effect. In addition, the influence of dataset quality on the
detection results is studied. By controlling the sealing property of pixels,
scale difference, unlabelled cells, and diagonal annotation, the model
detection accuracy, recall, mAP@0.5, and mAP@0.5:0.95 are improved by 13.3%,
15.3%, 18.3%, and 30.5%, respectively.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2412.01372v1)

---


## Thermal Vision: Pioneering Non\-Invasive Temperature Tracking in Congested Spaces

**发布日期**：2024-12-01

**作者**：Arijit Samal

**摘要**：Non\-invasive temperature monitoring of individuals plays a crucial role in
identifying and isolating symptomatic individuals. Temperature monitoring
becomes particularly vital in settings characterized by close human proximity,
often referred to as dense settings. However, existing research on non\-invasive
temperature estimation using thermal cameras has predominantly focused on
sparse settings. Unfortunately, the risk of disease transmission is
significantly higher in dense settings like movie theaters or classrooms.
Consequently, there is an urgent need to develop robust temperature estimation
methods tailored explicitly for dense settings.
  Our study proposes a non\-invasive temperature estimation system that combines
a thermal camera with an edge device. Our system employs YOLO models for face
detection and utilizes a regression framework for temperature estimation. We
evaluated the system on a diverse dataset collected in dense and sparse
settings. Our proposed face detection model achieves an impressive mAP score of
over 84 in both in\-dataset and cross\-dataset evaluations. Furthermore, the
regression framework demonstrates remarkable performance with a mean square
error of 0.18$^\{\\circ\}$C and an impressive $R^2$ score of 0.96. Our
experiments' results highlight the developed system's effectiveness,
positioning it as a promising solution for continuous temperature monitoring in
real\-world applications. With this paper, we release our dataset and
programming code publicly.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2412.00863v1)

---


## Visual Modality Prompt for Adapting Vision\-Language Object Detectors

**发布日期**：2024-12-01

**作者**：Heitor R. Medeiros

**摘要**：The zero\-shot performance of object detectors degrades when tested on
different modalities, such as infrared and depth. While recent work has
explored image translation techniques to adapt detectors to new modalities,
these methods are limited to a single modality and apply only to traditional
detectors. Recently, vision\-language detectors, such as YOLO\-World and
Grounding DINO, have shown promising zero\-shot capabilities, however, they have
not yet been adapted for other visual modalities. Traditional fine\-tuning
approaches tend to compromise the zero\-shot capabilities of the detectors. The
visual prompt strategies commonly used for classification with vision\-language
models apply the same linear prompt translation to each image making them less
effective. To address these limitations, we propose ModPrompt, a visual prompt
strategy to adapt vision\-language detectors to new modalities without degrading
zero\-shot performance. In particular, an encoder\-decoder visual prompt strategy
is proposed, further enhanced by the integration of inference\-friendly task
residuals, facilitating more robust adaptation. Empirically, we benchmark our
method for modality adaptation on two vision\-language detectors, YOLO\-World and
Grounding DINO, and on challenging infrared \(LLVIP, FLIR\) and depth \(NYUv2\)
data, achieving performance comparable to full fine\-tuning while preserving the
model's zero\-shot capability. Our code is available at:
https://github.com/heitorrapela/ModPrompt


**代码链接**：https://github.com/heitorrapela/ModPrompt

**论文链接**：[阅读更多](http://arxiv.org/abs/2412.00622v1)

---


## Real\-Time Anomaly Detection in Video Streams

**发布日期**：2024-11-29

**作者**：Fabien Poirier

**摘要**：This thesis is part of a CIFRE agreement between the company Othello and the
LIASD laboratory. The objective is to develop an artificial intelligence system
that can detect real\-time dangers in a video stream. To achieve this, a novel
approach combining temporal and spatial analysis has been proposed. Several
avenues have been explored to improve anomaly detection by integrating object
detection, human pose detection, and motion analysis. For result
interpretability, techniques commonly used for image analysis, such as
activation and saliency maps, have been extended to videos, and an original
method has been proposed. The proposed architecture performs binary or
multiclass classification depending on whether an alert or the cause needs to
be identified. Numerous neural networkmodels have been tested, and three of
them have been selected. You Only Looks Once \(YOLO\) has been used for spatial
analysis, a Convolutional Recurrent Neuronal Network \(CRNN\) composed of VGG19
and a Gated Recurrent Unit \(GRU\) for temporal analysis, and a multi\-layer
perceptron for classification. These models handle different types of data and
can be combined in parallel or in series. Although the parallel mode is faster,
the serial mode is generally more reliable. For training these models,
supervised learning was chosen, and two proprietary datasets were created. The
first dataset focuses on objects that may play a potential role in anomalies,
while the second consists of videos containing anomalies or non\-anomalies. This
approach allows for the processing of both continuous video streams and finite
videos, providing greater flexibility in detection.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2411.19731v1)

---


## Data Augmentation with Diffusion Models for Colon Polyp Localization on the Low Data Regime: How much real data is enough?

**发布日期**：2024-11-28

**作者**：Adrian Tormos

**摘要**：The scarcity of data in medical domains hinders the performance of Deep
Learning models. Data augmentation techniques can alleviate that problem, but
they usually rely on functional transformations of the data that do not
guarantee to preserve the original tasks. To approximate the distribution of
the data using generative models is a way of reducing that problem and also to
obtain new samples that resemble the original data. Denoising Diffusion models
is a promising Deep Learning technique that can learn good approximations of
different kinds of data like images, time series or tabular data.
  Automatic colonoscopy analysis and specifically Polyp localization in
colonoscopy videos is a task that can assist clinical diagnosis and treatment.
The annotation of video frames for training a deep learning model is a time
consuming task and usually only small datasets can be obtained. The fine tuning
of application models using a large dataset of generated data could be an
alternative to improve their performance. We conduct a set of experiments
training different diffusion models that can generate jointly colonoscopy
images with localization annotations using a combination of existing open
datasets. The generated data is used on various transfer learning experiments
in the task of polyp localization with a model based on YOLO v9 on the low data
regime.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2411.18926v1)

---


## Enhancing weed detection performance by means of GenAI\-based image augmentation

**发布日期**：2024-11-27

**作者**：Sourav Modak

**摘要**：Precise weed management is essential for sustaining crop productivity and
ecological balance. Traditional herbicide applications face economic and
environmental challenges, emphasizing the need for intelligent weed control
systems powered by deep learning. These systems require vast amounts of
high\-quality training data. The reality of scarcity of well\-annotated training
data, however, is often addressed through generating more data using data
augmentation. Nevertheless, conventional augmentation techniques such as random
flipping, color changes, and blurring lack sufficient fidelity and diversity.
This paper investigates a generative AI\-based augmentation technique that uses
the Stable Diffusion model to produce diverse synthetic images that improve the
quantity and quality of training datasets for weed detection models. Moreover,
this paper explores the impact of these synthetic images on the performance of
real\-time detection systems, thus focusing on compact CNN\-based models such as
YOLO nano for edge devices. The experimental results show substantial
improvements in mean Average Precision \(mAP50 and mAP50\-95\) scores for YOLO
models trained with generative AI\-augmented datasets, demonstrating the
promising potential of synthetic data to enhance model robustness and accuracy.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2411.18513v2)

---


## AI\-Driven Smartphone Solution for Digitizing Rapid Diagnostic Test Kits and Enhancing Accessibility for the Visually Impaired

**发布日期**：2024-11-27

**作者**：R. B. Dastagir

**摘要**：Rapid diagnostic tests are crucial for timely disease detection and
management, yet accurate interpretation of test results remains challenging. In
this study, we propose a novel approach to enhance the accuracy and reliability
of rapid diagnostic test result interpretation by integrating artificial
intelligence \(AI\) algorithms, including convolutional neural networks \(CNN\),
within a smartphone\-based application. The app enables users to take pictures
of their test kits, which YOLOv8 then processes to precisely crop and extract
the membrane region, even if the test kit is not centered in the frame or is
positioned at the very edge of the image. This capability offers greater
accessibility, allowing even visually impaired individuals to capture test
images without needing perfect alignment, thus promoting user independence and
inclusivity. The extracted image is analyzed by an additional CNN classifier
that determines if the results are positive, negative, or invalid, providing
users with the results and a confidence level. Through validation experiments
with commonly used rapid test kits across various diagnostic applications, our
results demonstrate that the synergistic integration of AI significantly
improves sensitivity and specificity in test result interpretation. This
improvement can be attributed to the extraction of the membrane zones from the
test kit images using the state\-of\-the\-art YOLO algorithm. Additionally, we
performed SHapley Additive exPlanations \(SHAP\) analysis to investigate the
factors influencing the model's decisions, identifying reasons behind both
correct and incorrect classifications. By facilitating the differentiation of
genuine test lines from background noise and providing valuable insights into
test line intensity and uniformity, our approach offers a robust solution to
challenges in rapid test interpretation.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2411.18007v1)

---


## DGNN\-YOLO: Dynamic Graph Neural Networks with YOLO11 for Small Object Detection and Tracking in Traffic Surveillance

**发布日期**：2024-11-26

**作者**：Shahriar Soudeep

**摘要**：Accurate detection and tracking of small objects such as pedestrians,
cyclists, and motorbikes are critical for traffic surveillance systems, which
are crucial in improving road safety and decision\-making in intelligent
transportation systems. However, traditional methods struggle with challenges
such as occlusion, low resolution, and dynamic traffic conditions,
necessitating innovative approaches to address these limitations. This paper
introduces DGNN\-YOLO, a novel framework integrating dynamic graph neural
networks \(DGNN\) with YOLO11 to enhance small object detection and tracking in
traffic surveillance systems. The framework leverages YOLO11's advanced spatial
feature extraction capabilities for precise object detection and incorporates
DGNN to model spatial\-temporal relationships for robust real\-time tracking
dynamically. By constructing and updating graph structures, DGNN\-YOLO
effectively represents objects as nodes and their interactions as edges,
ensuring adaptive and accurate tracking in complex and dynamic environments.
Extensive experiments demonstrate that DGNN\-YOLO consistently outperforms
state\-of\-the\-art methods in detecting and tracking small objects under diverse
traffic conditions, achieving the highest precision \(0.8382\), recall \(0.6875\),
and mAP@0.5:0.95 \(0.6476\), showcasing its robustness and scalability,
particularly in challenging scenarios involving small and occluded objects.
This work provides a scalable, real\-time traffic surveillance and analysis
solution, significantly contributing to intelligent transportation systems.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2411.17251v2)

---


## Learn from Foundation Model: Fruit Detection Model without Manual Annotation

**发布日期**：2024-11-25

**作者**：Yanan Wang

**摘要**：Recent breakthroughs in large foundation models have enabled the possibility
of transferring knowledge pre\-trained on vast datasets to domains with limited
data availability. Agriculture is one of the domains that lacks sufficient
data. This study proposes a framework to train effective, domain\-specific,
small models from foundation models without manual annotation. Our approach
begins with SDM \(Segmentation\-Description\-Matching\), a stage that leverages two
foundation models: SAM2 \(Segment Anything in Images and Videos\) for
segmentation and OpenCLIP \(Open Contrastive Language\-Image Pretraining\) for
zero\-shot open\-vocabulary classification. In the second stage, a novel
knowledge distillation mechanism is utilized to distill compact,
edge\-deployable models from SDM, enhancing both inference speed and perception
accuracy. The complete method, termed SDM\-D
\(Segmentation\-Description\-Matching\-Distilling\), demonstrates strong performance
across various fruit detection tasks object detection, semantic segmentation,
and instance segmentation\) without manual annotation. It nearly matches the
performance of models trained with abundant labels. Notably, SDM\-D outperforms
open\-set detection methods such as Grounding SAM and YOLO\-World on all tested
fruit detection datasets. Additionally, we introduce MegaFruits, a
comprehensive fruit segmentation dataset encompassing over 25,000 images, and
all code and datasets are made publicly available at
https://github.com/AgRoboticsResearch/SDM\-D.git.


**代码链接**：https://github.com/AgRoboticsResearch/SDM-D.git.

**论文链接**：[阅读更多](http://arxiv.org/abs/2411.16196v1)

---

