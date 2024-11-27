# 每日从arXiv中获取最新YOLO相关论文


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


## WARLearn: Weather\-Adaptive Representation Learning

**发布日期**：2024-11-21

**作者**：Shubham Agarwal

**摘要**：This paper introduces WARLearn, a novel framework designed for adaptive
representation learning in challenging and adversarial weather conditions.
Leveraging the in\-variance principal used in Barlow Twins, we demonstrate the
capability to port the existing models initially trained on clear weather data
to effectively handle adverse weather conditions. With minimal additional
training, our method exhibits remarkable performance gains in scenarios
characterized by fog and low\-light conditions. This adaptive framework extends
its applicability beyond adverse weather settings, offering a versatile
solution for domains exhibiting variations in data distributions. Furthermore,
WARLearn is invaluable in scenarios where data distributions undergo
significant shifts over time, enabling models to remain updated and accurate.
Our experimental findings reveal a remarkable performance, with a mean average
precision \(mAP\) of 52.6% on unseen real\-world foggy dataset \(RTTS\). Similarly,
in low light conditions, our framework achieves a mAP of 55.7% on unseen
real\-world low light dataset \(ExDark\). Notably, WARLearn surpasses the
performance of state\-of\-the\-art frameworks including FeatEnHancer, Image
Adaptive YOLO, DENet, C2PNet, PairLIE and ZeroDCE, by a substantial margin in
adverse weather, improving the baseline performance in both foggy and low light
conditions. The WARLearn code is available at
https://github.com/ShubhamAgarwal12/WARLearn


**代码链接**：https://github.com/ShubhamAgarwal12/WARLearn

**论文链接**：[阅读更多](http://arxiv.org/abs/2411.14095v1)

---


## Mirror Target YOLO: An Improved YOLOv8 Method with Indirect Vision for Heritage Buildings Fire Detection

**发布日期**：2024-11-21

**作者**：Jian Liang

**摘要**：Fires can cause severe damage to heritage buildings, making timely fire
detection essential. Traditional dense cabling and drilling can harm these
structures, so reducing the number of cameras to minimize such impact is
challenging. Additionally, avoiding false alarms due to noise sensitivity and
preserving the expertise of managers in fire\-prone areas is crucial. To address
these needs, we propose a fire detection method based on indirect vision,
called Mirror Target YOLO \(MITA\-YOLO\). MITA\-YOLO integrates indirect vision
deployment and an enhanced detection module. It uses mirror angles to achieve
indirect views, solving issues with limited visibility in irregular spaces and
aligning each indirect view with the target monitoring area. The Target\-Mask
module is designed to automatically identify and isolate the indirect vision
areas in each image, filtering out non\-target areas. This enables the model to
inherit managers' expertise in assessing fire\-risk zones, improving focus and
resistance to interference in fire detection.In our experiments, we created an
800\-image fire dataset with indirect vision. Results show that MITA\-YOLO
significantly reduces camera requirements while achieving superior detection
performance compared to other mainstream models.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2411.13997v1)

---


## Enhancing Bidirectional Sign Language Communication: Integrating YOLOv8 and NLP for Real\-Time Gesture Recognition & Translation

**发布日期**：2024-11-18

**作者**：Hasnat Jamil Bhuiyan

**摘要**：The primary concern of this research is to take American Sign Language \(ASL\)
data through real time camera footage and be able to convert the data and
information into text. Adding to that, we are also putting focus on creating a
framework that can also convert text into sign language in real time which can
help us break the language barrier for the people who are in need. In this
work, for recognising American Sign Language \(ASL\), we have used the You Only
Look Once\(YOLO\) model and Convolutional Neural Network \(CNN\) model. YOLO model
is run in real time and automatically extracts discriminative spatial\-temporal
characteristics from the raw video stream without the need for any prior
knowledge, eliminating design flaws. The CNN model here is also run in real
time for sign language detection. We have introduced a novel method for
converting text based input to sign language by making a framework that will
take a sentence as input, identify keywords from that sentence and then show a
video where sign language is performed with respect to the sentence given as
input in real time. To the best of our knowledge, this is a rare study to
demonstrate bidirectional sign language communication in real time in the
American Sign Language \(ASL\).


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2411.13597v1)

---


## WoodYOLO: A Novel Object Detector for Wood Species Detection in Microscopic Images

**发布日期**：2024-11-18

**作者**：Lars Nieradzik

**摘要**：Wood species identification plays a crucial role in various industries, from
ensuring the legality of timber products to advancing ecological conservation
efforts. This paper introduces WoodYOLO, a novel object detection algorithm
specifically designed for microscopic wood fiber analysis. Our approach adapts
the YOLO architecture to address the challenges posed by large, high\-resolution
microscopy images and the need for high recall in localization of the cell type
of interest \(vessel elements\). Our results show that WoodYOLO significantly
outperforms state\-of\-the\-art models, achieving performance gains of 12.9% and
6.5% in F2 score over YOLOv10 and YOLOv7, respectively. This improvement in
automated wood cell type localization capabilities contributes to enhancing
regulatory compliance, supporting sustainable forestry practices, and promoting
biodiversity conservation efforts globally.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2411.11738v1)

---

