# 每日从arXiv中获取最新YOLO相关论文


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

**论文链接**：[阅读更多](http://arxiv.org/abs/2411.17251v1)

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


## You only thermoelastically deform once: Point Absorber Detection in LIGO Test Masses with YOLO

**发布日期**：2024-11-25

**作者**：Simon R. Goode

**摘要**：Current and future gravitational\-wave observatories rely on large\-scale,
precision interferometers to detect the gravitational\-wave signals. However,
microscopic imperfections on the test masses, known as point absorbers, cause
problematic heating of the optic via absorption of the high\-power laser beam,
which results in diminished sensitivity, lock loss, or even permanent damage.
Consistent monitoring of the test masses is crucial for detecting,
characterizing, and ultimately removing point absorbers. We present a
machine\-learning algorithm for detecting point absorbers based on the
object\-detection algorithm You Only Look Once \(YOLO\). The algorithm can perform
this task in situ while the detector is in operation. We validate our algorithm
by comparing it with past reports of point absorbers identified by humans at
LIGO. The algorithm confidently identifies the same point absorbers as humans
with minimal false positives. It also identifies some point absorbers
previously not identified by humans, which we confirm with human follow\-up. We
highlight the potential of machine learning in commissioning efforts.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2411.16104v1)

---


## Enhancing Object Detection Accuracy in Autonomous Vehicles Using Synthetic Data

**发布日期**：2024-11-23

**作者**：Sergei Voronin

**摘要**：The rapid progress in machine learning models has significantly boosted the
potential for real\-world applications such as autonomous vehicles, disease
diagnoses, and recognition of emergencies. The performance of many machine
learning models depends on the nature and size of the training data sets. These
models often face challenges due to the scarcity, noise, and imbalance in
real\-world data, limiting their performance. Nonetheless, high\-quality,
diverse, relevant and representative training data is essential to build
accurate and reliable machine learning models that adapt well to real\-world
scenarios.
  It is hypothesised that well\-designed synthetic data can improve the
performance of a machine learning algorithm. This work aims to create a
synthetic dataset and evaluate its effectiveness to improve the prediction
accuracy of object detection systems. This work considers autonomous vehicle
scenarios as an illustrative example to show the efficacy of synthetic data.
The effectiveness of these synthetic datasets in improving the performance of
state\-of\-the\-art object detection models is explored. The findings demonstrate
that incorporating synthetic data improves model performance across all
performance matrices.
  Two deep learning systems, System\-1 \(trained on real\-world data\) and System\-2
\(trained on a combination of real and synthetic data\), are evaluated using the
state\-of\-the\-art YOLO model across multiple metrics, including accuracy,
precision, recall, and mean average precision. Experimental results revealed
that System\-2 outperformed System\-1, showing a 3% improvement in accuracy,
along with superior performance in all other metrics.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2411.15602v1)

---


## Evaluating the Impact of Underwater Image Enhancement on Object Detection Performance: A Comprehensive Study

**发布日期**：2024-11-21

**作者**：Ali Awad

**摘要**：Underwater imagery often suffers from severe degradation that results in low
visual quality and object detection performance. This work aims to evaluate
state\-of\-the\-art image enhancement models, investigate their impact on
underwater object detection, and explore their potential to improve detection
performance. To this end, we selected representative underwater image
enhancement models covering major enhancement categories and applied them
separately to two recent datasets: 1\) the Real\-World Underwater Object
Detection Dataset \(RUOD\), and 2\) the Challenging Underwater Plant Detection
Dataset \(CUPDD\). Following this, we conducted qualitative and quantitative
analyses on the enhanced images and developed a quality index \(Q\-index\) to
compare the quality distribution of the original and enhanced images.
Subsequently, we compared the performance of several YOLO\-NAS detection models
that are separately trained and tested on the original and enhanced image sets.
Then, we performed a correlation study to examine the relationship between
enhancement metrics and detection performance. We also analyzed the inference
results from the trained detectors presenting cases where enhancement increased
the detection performance as well as cases where enhancement revealed missed
objects by human annotators. This study suggests that although enhancement
generally deteriorates the detection performance, it can still be harnessed in
some cases for increased detection performance and more accurate human
annotation.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2411.14626v2)

---


## Towards Context\-Rich Automated Biodiversity Assessments: Deriving AI\-Powered Insights from Camera Trap Data

**发布日期**：2024-11-21

**作者**：Paul Fergus

**摘要**：Camera traps offer enormous new opportunities in ecological studies, but
current automated image analysis methods often lack the contextual richness
needed to support impactful conservation outcomes. Here we present an
integrated approach that combines deep learning\-based vision and language
models to improve ecological reporting using data from camera traps. We
introduce a two\-stage system: YOLOv10\-X to localise and classify species
\(mammals and birds\) within images, and a Phi\-3.5\-vision\-instruct model to read
YOLOv10\-X binding box labels to identify species, overcoming its limitation
with hard to classify objects in images. Additionally, Phi\-3.5 detects broader
variables, such as vegetation type, and time of day, providing rich ecological
and environmental context to YOLO's species detection output. When combined,
this output is processed by the model's natural language system to answer
complex queries, and retrieval\-augmented generation \(RAG\) is employed to enrich
responses with external information, like species weight and IUCN status
\(information that cannot be obtained through direct visual analysis\). This
information is used to automatically generate structured reports, providing
biodiversity stakeholders with deeper insights into, for example, species
abundance, distribution, animal behaviour, and habitat selection. Our approach
delivers contextually rich narratives that aid in wildlife management
decisions. By providing contextually rich insights, our approach not only
reduces manual effort but also supports timely decision\-making in conservation,
potentially shifting efforts from reactive to proactive management.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2411.14219v1)

---

