# 每日从arXiv中获取最新YOLO相关论文


## A Light\-Weight Framework for Open\-Set Object Detection with Decoupled Feature Alignment in Joint Space

**发布日期**：2024-12-19

**作者**：Yonghao He

**摘要**：Open\-set object detection \(OSOD\) is highly desirable for robotic manipulation
in unstructured environments. However, existing OSOD methods often fail to meet
the requirements of robotic applications due to their high computational burden
and complex deployment. To address this issue, this paper proposes a
light\-weight framework called Decoupled OSOD \(DOSOD\), which is a practical and
highly efficient solution to support real\-time OSOD tasks in robotic systems.
Specifically, DOSOD builds upon the YOLO\-World pipeline by integrating a
vision\-language model \(VLM\) with a detector. A Multilayer Perceptron \(MLP\)
adaptor is developed to transform text embeddings extracted by the VLM into a
joint space, within which the detector learns the region representations of
class\-agnostic proposals. Cross\-modality features are directly aligned in the
joint space, avoiding the complex feature interactions and thereby improving
computational efficiency. DOSOD operates like a traditional closed\-set detector
during the testing phase, effectively bridging the gap between closed\-set and
open\-set detection. Compared to the baseline YOLO\-World, the proposed DOSOD
significantly enhances real\-time performance while maintaining comparable
accuracy. The slight DOSOD\-S model achieves a Fixed AP of $26.7\\%$, compared to
$26.2\\%$ for YOLO\-World\-v1\-S and $22.7\\%$ for YOLO\-World\-v2\-S, using similar
backbones on the LVIS minival dataset. Meanwhile, the FPS of DOSOD\-S is
$57.1\\%$ higher than YOLO\-World\-v1\-S and $29.6\\%$ higher than YOLO\-World\-v2\-S.
Meanwhile, we demonstrate that the DOSOD model facilitates the deployment of
edge devices. The codes and models are publicly available at
https://github.com/D\-Robotics\-AI\-Lab/DOSOD.


**代码链接**：https://github.com/D-Robotics-AI-Lab/DOSOD.

**论文链接**：[阅读更多](http://arxiv.org/abs/2412.14680v1)

---


## Physics\-Based Adversarial Attack on Near\-Infrared Human Detector for Nighttime Surveillance Camera Systems

**发布日期**：2024-12-18

**作者**：Muyao Niu

**摘要**：Many surveillance cameras switch between daytime and nighttime modes based on
illuminance levels. During the day, the camera records ordinary RGB images
through an enabled IR\-cut filter. At night, the filter is disabled to capture
near\-infrared \(NIR\) light emitted from NIR LEDs typically mounted around the
lens. While RGB\-based AI algorithm vulnerabilities have been widely reported,
the vulnerabilities of NIR\-based AI have rarely been investigated. In this
paper, we identify fundamental vulnerabilities in NIR\-based image understanding
caused by color and texture loss due to the intrinsic characteristics of
clothes' reflectance and cameras' spectral sensitivity in the NIR range. We
further show that the nearly co\-located configuration of illuminants and
cameras in existing surveillance systems facilitates concealing and fully
passive attacks in the physical world. Specifically, we demonstrate how
retro\-reflective and insulation plastic tapes can manipulate the intensity
distribution of NIR images. We showcase an attack on the YOLO\-based human
detector using binary patterns designed in the digital space \(via black\-box
query and searching\) and then physically realized using tapes pasted onto
clothes. Our attack highlights significant reliability concerns for nighttime
surveillance systems, which are intended to enhance security. Codes Available:
https://github.com/MyNiuuu/AdvNIR


**代码链接**：https://github.com/MyNiuuu/AdvNIR

**论文链接**：[阅读更多](http://arxiv.org/abs/2412.13709v1)

---


## Training a Distributed Acoustic Sensing Traffic Monitoring Network With Video Inputs

**发布日期**：2024-12-17

**作者**：Khen Cohen

**摘要**：Distributed Acoustic Sensing \(DAS\) has emerged as a promising tool for
real\-time traffic monitoring in densely populated areas. In this paper, we
present a novel concept that integrates DAS data with co\-located visual
information. We use YOLO\-derived vehicle location and classification from
camera inputs as labeled data to train a detection and classification neural
network utilizing DAS data only. Our model achieves a performance exceeding 94%
for detection and classification, and about 1.2% false alarm rate. We
illustrate the model's application in monitoring traffic over a week, yielding
statistical insights that could benefit future smart city developments. Our
approach highlights the potential of combining fiber\-optic sensors with visual
information, focusing on practicality and scalability, protecting privacy, and
minimizing infrastructure costs. To encourage future research, we share our
dataset.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2412.12743v1)

---


## Domain Generalization in Autonomous Driving: Evaluating YOLOv8s, RT\-DETR, and YOLO\-NAS with the ROAD\-Almaty Dataset

**发布日期**：2024-12-16

**作者**：Madiyar Alimov

**摘要**：This study investigates the domain generalization capabilities of three
state\-of\-the\-art object detection models \- YOLOv8s, RT\-DETR, and YOLO\-NAS \-
within the unique driving environment of Kazakhstan. Utilizing the newly
constructed ROAD\-Almaty dataset, which encompasses diverse weather, lighting,
and traffic conditions, we evaluated the models' performance without any
retraining. Quantitative analysis revealed that RT\-DETR achieved an average
F1\-score of 0.672 at IoU=0.5, outperforming YOLOv8s \(0.458\) and YOLO\-NAS
\(0.526\) by approximately 46% and 27%, respectively. Additionally, all models
exhibited significant performance declines at higher IoU thresholds \(e.g., a
drop of approximately 20% when increasing IoU from 0.5 to 0.75\) and under
challenging environmental conditions, such as heavy snowfall and low\-light
scenarios. These findings underscore the necessity for geographically diverse
training datasets and the implementation of specialized domain adaptation
techniques to enhance the reliability of autonomous vehicle detection systems
globally. This research contributes to the understanding of domain
generalization challenges in autonomous driving, particularly in
underrepresented regions.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2412.12349v1)

---


## Coconut Palm Tree Counting on Drone Images with Deep Object Detection and Synthetic Training Data

**发布日期**：2024-12-16

**作者**：Tobias Rohe

**摘要**：Drones have revolutionized various domains, including agriculture. Recent
advances in deep learning have propelled among other things object detection in
computer vision. This study utilized YOLO, a real\-time object detector, to
identify and count coconut palm trees in Ghanaian farm drone footage. The farm
presented has lost track of its trees due to different planting phases. While
manual counting would be very tedious and error\-prone, accurately determining
the number of trees is crucial for efficient planning and management of
agricultural processes, especially for optimizing yields and predicting
production. We assessed YOLO for palm detection within a semi\-automated
framework, evaluated accuracy augmentations, and pondered its potential for
farmers. Data was captured in September 2022 via drones. To optimize YOLO with
scarce data, synthetic images were created for model training and validation.
The YOLOv7 model, pretrained on the COCO dataset \(excluding coconut palms\), was
adapted using tailored data. Trees from footage were repositioned on synthetic
images, with testing on distinct authentic images. In our experiments, we
adjusted hyperparameters, improving YOLO's mean average precision \(mAP\). We
also tested various altitudes to determine the best drone height. From an
initial mAP@.5 of $0.65$, we achieved 0.88, highlighting the value of synthetic
images in agricultural scenarios.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2412.11949v1)

---


## CLDA\-YOLO: Visual Contrastive Learning Based Domain Adaptive YOLO Detector

**发布日期**：2024-12-16

**作者**：Tianheng Qiu

**摘要**：Unsupervised domain adaptive \(UDA\) algorithms can markedly enhance the
performance of object detectors under conditions of domain shifts, thereby
reducing the necessity for extensive labeling and retraining. Current domain
adaptive object detection algorithms primarily cater to two\-stage detectors,
which tend to offer minimal improvements when directly applied to single\-stage
detectors such as YOLO. Intending to benefit the YOLO detector from UDA, we
build a comprehensive domain adaptive architecture using a teacher\-student
cooperative system for the YOLO detector. In this process, we propose
uncertainty learning to cope with pseudo\-labeling generated by the teacher
model with extreme uncertainty and leverage dynamic data augmentation to
asymptotically adapt the teacher\-student system to the environment. To address
the inability of single\-stage object detectors to align at multiple stages, we
utilize a unified visual contrastive learning paradigm that aligns instance at
backbone and head respectively, which steadily improves the robustness of the
detectors in cross\-domain tasks. In summary, we present an unsupervised domain
adaptive YOLO detector based on visual contrastive learning \(CLDA\-YOLO\), which
achieves highly competitive results across multiple domain adaptive datasets
without any reduction in inference speed.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2412.11812v1)

---


## PyPotteryLens: An Open\-Source Deep Learning Framework for Automated Digitisation of Archaeological Pottery Documentation

**发布日期**：2024-12-16

**作者**：Lorenzo Cardarelli

**摘要**：Archaeological pottery documentation and study represents a crucial but
time\-consuming aspect of archaeology. While recent years have seen advances in
digital documentation methods, vast amounts of legacy data remain locked in
traditional publications. This paper introduces PyPotteryLens, an open\-source
framework that leverages deep learning to automate the digitisation and
processing of archaeological pottery drawings from published sources. The
system combines state\-of\-the\-art computer vision models \(YOLO for instance
segmentation and EfficientNetV2 for classification\) with an intuitive user
interface, making advanced digital methods accessible to archaeologists
regardless of technical expertise. The framework achieves over 97\\% precision
and recall in pottery detection and classification tasks, while reducing
processing time by up to 5x to 20x compared to manual methods. Testing across
diverse archaeological contexts demonstrates robust generalisation
capabilities. Also, the system's modular architecture facilitates extension to
other archaeological materials, while its standardised output format ensures
long\-term preservation and reusability of digitised data as well as solid basis
for training machine learning algorithms. The software, documentation, and
examples are available on GitHub
\(https://github.com/lrncrd/PyPottery/tree/PyPotteryLens\).


**代码链接**：https://github.com/lrncrd/PyPottery/tree/PyPotteryLens).

**论文链接**：[阅读更多](http://arxiv.org/abs/2412.11574v1)

---


## Enhancing Road Crack Detection Accuracy with BsS\-YOLO: Optimizing Feature Fusion and Attention Mechanisms

**发布日期**：2024-12-14

**作者**：Jiaze Tang

**摘要**：Effective road crack detection is crucial for road safety, infrastructure
preservation, and extending road lifespan, offering significant economic
benefits. However, existing methods struggle with varied target scales, complex
backgrounds, and low adaptability to different environments. This paper
presents the BsS\-YOLO model, which optimizes multi\-scale feature fusion through
an enhanced Path Aggregation Network \(PAN\) and Bidirectional Feature Pyramid
Network \(BiFPN\). The incorporation of weighted feature fusion improves feature
representation, boosting detection accuracy and robustness. Furthermore, a
Simple and Effective Attention Mechanism \(SimAM\) within the backbone enhances
precision via spatial and channel\-wise attention. The detection layer
integrates a Shuffle Attention mechanism, which rearranges and mixes features
across channels, refining key representations and further improving accuracy.
Experimental results show that BsS\-YOLO achieves a 2.8% increase in mean
average precision \(mAP\) for road crack detection, supporting its applicability
in diverse scenarios, including urban road maintenance and highway inspections.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2412.10902v1)

---


## SPACE\-SUIT: An Artificial Intelligence based chromospheric feature extractor and classifier for SUIT

**发布日期**：2024-12-11

**作者**：Pranava Seth

**摘要**：The Solar Ultraviolet Imaging Telescope\(SUIT\) onboard Aditya\-L1 is an imager
that observes the solar photosphere and chromosphere through observations in
the wavelength range of 200\-400 nm. A comprehensive understanding of the plasma
and thermodynamic properties of chromospheric and photospheric morphological
structures requires a large sample statistical study, necessitating the
development of automatic feature detection methods. To this end, we develop the
feature detection algorithm SPACE\-SUIT: Solar Phenomena Analysis and
Classification using Enhanced vision techniques for SUIT, to detect and
classify the solar chromospheric features to be observed from SUIT's Mg II k
filter. Specifically, we target plage regions, sunspots, filaments, and
off\-limb structures. SPACE uses You Only Look Once\(YOLO\), a neural
network\-based model to identify regions of interest. We train and validate
SPACE using mock\-SUIT images developed from Interface Region Imaging
Spectrometer\(IRIS\) full\-disk mosaic images in Mg II k line, while we also
perform detection on Level\-1 SUIT data. SPACE achieves an approximate precision
of 0.788, recall 0.863 and MAP of 0.874 on the validation mock SUIT FITS
dataset. Given the manual labeling of our dataset, we perform "self\-validation"
by applying statistical measures and Tamura features on the ground truth and
predicted bounding boxes. We find the distributions of entropy, contrast,
dissimilarity, and energy to show differences in the features. These
differences are qualitatively captured by the detected regions predicted by
SPACE and validated with the observed SUIT images, even in the absence of
labeled ground truth. This work not only develops a chromospheric feature
extractor but also demonstrates the effectiveness of statistical metrics and
Tamura features for distinguishing chromospheric features, offering independent
validation for future detection schemes.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2412.08589v1)

---


## DynamicPAE: Generating Scene\-Aware Physical Adversarial Examples in Real\-Time

**发布日期**：2024-12-11

**作者**：Jin Hu

**摘要**：Physical adversarial examples \(PAEs\) are regarded as "whistle\-blowers" of
real\-world risks in deep\-learning applications. However, current PAE generation
studies show limited adaptive attacking ability to diverse and varying scenes.
The key challenges in generating dynamic PAEs are exploring their patterns
under noisy gradient feedback and adapting the attack to agnostic scenario
natures. To address the problems, we present DynamicPAE, the first generative
framework that enables scene\-aware real\-time physical attacks beyond static
attacks. Specifically, to train the dynamic PAE generator under noisy gradient
feedback, we introduce the residual\-driven sample trajectory guidance
technique, which redefines the training task to break the limited feedback
information restriction that leads to the degeneracy problem. Intuitively, it
allows the gradient feedback to be passed to the generator through a low\-noise
auxiliary task, thereby guiding the optimization away from degenerate solutions
and facilitating a more comprehensive and stable exploration of feasible PAEs.
To adapt the generator to agnostic scenario natures, we introduce the
context\-aligned scene expectation simulation process, consisting of the
conditional\-uncertainty\-aligned data module and the skewness\-aligned objective
re\-weighting module. The former enhances robustness in the context of
incomplete observation by employing a conditional probabilistic model for
domain randomization, while the latter facilitates consistent stealth control
across different attack targets by automatically reweighting losses based on
the skewness indicator. Extensive digital and physical evaluations demonstrate
the superior attack performance of DynamicPAE, attaining a 1.95 $\\times$ boost
\(65.55% average AP drop under attack\) on representative object detectors \(e.g.,
Yolo\-v8\) over state\-of\-the\-art static PAE generating methods.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2412.08053v1)

---

