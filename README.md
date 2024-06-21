# 每日从arXiv中获取最新YOLO相关论文


## LeYOLO, New Scalable and Efficient CNN Architecture for Object Detection

**发布日期**：2024-06-20

**作者**：Lilian Hollard

**摘要**：Computational efficiency in deep neural networks is critical for object
detection, especially as newer models prioritize speed over efficient
computation \(FLOP\). This evolution has somewhat left behind embedded and
mobile\-oriented AI object detection applications. In this paper, we focus on
design choices of neural network architectures for efficient object detection
computation based on FLOP and propose several optimizations to enhance the
efficiency of YOLO\-based models.
  Firstly, we introduce an efficient backbone scaling inspired by inverted
bottlenecks and theoretical insights from the Information Bottleneck principle.
Secondly, we present the Fast Pyramidal Architecture Network \(FPAN\), designed
to facilitate fast multiscale feature sharing while reducing computational
resources. Lastly, we propose a Decoupled Network\-in\-Network \(DNiN\) detection
head engineered to deliver rapid yet lightweight computations for
classification and regression tasks.
  Building upon these optimizations and leveraging more efficient backbones,
this paper contributes to a new scaling paradigm for object detection and
YOLO\-centric models called LeYOLO. Our contribution consistently outperforms
existing models in various resource constraints, achieving unprecedented
accuracy and flop ratio. Notably, LeYOLO\-Small achieves a competitive mAP score
of 38.2% on the COCOval with just 4.5 FLOP\(G\), representing a 42% reduction in
computational load compared to the latest state\-of\-the\-art YOLOv9\-Tiny model
while achieving similar accuracy. Our novel model family achieves a
FLOP\-to\-accuracy ratio previously unattained, offering scalability that spans
from ultra\-low neural network configurations \(< 1 GFLOP\) to efficient yet
demanding object detection setups \(> 4 GFLOPs\) with 25.2, 31.3, 35.2, 38.2,
39.3 and 41 mAP for 0.66, 1.47, 2.53, 4.51, 5.8 and 8.4 FLOP\(G\).


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2406.14239v1)

---


## Enhancing the LLM\-Based Robot Manipulation Through Human\-Robot Collaboration

**发布日期**：2024-06-20

**作者**：Haokun Liu

**摘要**：Large Language Models \(LLMs\) are gaining popularity in the field of robotics.
However, LLM\-based robots are limited to simple, repetitive motions due to the
poor integration between language models, robots, and the environment. This
paper proposes a novel approach to enhance the performance of LLM\-based
autonomous manipulation through Human\-Robot Collaboration \(HRC\). The approach
involves using a prompted GPT\-4 language model to decompose high\-level language
commands into sequences of motions that can be executed by the robot. The
system also employs a YOLO\-based perception algorithm, providing visual cues to
the LLM, which aids in planning feasible motions within the specific
environment. Additionally, an HRC method is proposed by combining teleoperation
and Dynamic Movement Primitives \(DMP\), allowing the LLM\-based robot to learn
from human guidance. Real\-world experiments have been conducted using the
Toyota Human Support Robot for manipulation tasks. The outcomes indicate that
tasks requiring complex trajectory planning and reasoning over environments can
be efficiently accomplished through the incorporation of human demonstrations.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2406.14097v1)

---


## Towards the in\-situ Trunk Identification and Length Measurement of Sea Cucumbers via Bézier Curve Modelling

**发布日期**：2024-06-20

**作者**：Shuaixin Liu

**摘要**：We introduce a novel vision\-based framework for in\-situ trunk identification
and length measurement of sea cucumbers, which plays a crucial role in the
monitoring of marine ranching resources and mechanized harvesting. To model sea
cucumber trunk curves with varying degrees of bending, we utilize the
parametric B\\'\{e\}zier curve due to its computational simplicity, stability, and
extensive range of transformation possibilities. Then, we propose an end\-to\-end
unified framework that combines parametric B\\'\{e\}zier curve modeling with the
widely used You\-Only\-Look\-Once \(YOLO\) pipeline, abbreviated as TISC\-Net, and
incorporates effective funnel activation and efficient multi\-scale attention
modules to enhance curve feature perception and learning. Furthermore, we
propose incorporating trunk endpoint loss as an additional constraint to
effectively mitigate the impact of endpoint deviations on the overall curve.
Finally, by utilizing the depth information of pixels located along the trunk
curve captured by a binocular camera, we propose accurately estimating the
in\-situ length of sea cucumbers through space curve integration. We established
two challenging benchmark datasets for curve\-based in\-situ sea cucumber trunk
identification. These datasets consist of over 1,000 real\-world marine
environment images of sea cucumbers, accompanied by B\\'\{e\}zier format
annotations. We conduct evaluation on SC\-ISTI, for which our method achieves
mAP50 above 0.9 on both object detection and trunk identification tasks.
Extensive length measurement experiments demonstrate that the average absolute
relative error is around 0.15.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2406.13951v1)

---


## SDNIA\-YOLO: A Robust Object Detection Model for Extreme Weather Conditions

**发布日期**：2024-06-18

**作者**：Yuexiong Ding

**摘要**：Though current object detection models based on deep learning have achieved
excellent results on many conventional benchmark datasets, their performance
will dramatically decline on real\-world images taken under extreme conditions.
Existing methods either used image augmentation based on traditional image
processing algorithms or applied customized and scene\-limited image adaptation
technologies for robust modeling. This study thus proposes a stylization
data\-driven neural\-image\-adaptive YOLO \(SDNIA\-YOLO\), which improves the model's
robustness by enhancing image quality adaptively and learning valuable
information related to extreme weather conditions from images synthesized by
neural style transfer \(NST\). Experiments show that the developed SDNIA\-YOLOv3
achieves significant mAP@.5 improvements of at least 15% on the real\-world
foggy \(RTTS\) and lowlight \(ExDark\) test sets compared with the baseline model.
Besides, the experiments also highlight the outstanding potential of
stylization data in simulating extreme weather conditions. The developed
SDNIA\-YOLO remains excellent characteristics of the native YOLO to a great
extent, such as end\-to\-end one\-stage, data\-driven, and fast.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2406.12395v1)

---


## DASSF: Dynamic\-Attention Scale\-Sequence Fusion for Aerial Object Detection

**发布日期**：2024-06-18

**作者**：Haodong Li

**摘要**：The detection of small objects in aerial images is a fundamental task in the
field of computer vision. Moving objects in aerial photography have problems
such as different shapes and sizes, dense overlap, occlusion by the background,
and object blur, however, the original YOLO algorithm has low overall detection
accuracy due to its weak ability to perceive targets of different scales. In
order to improve the detection accuracy of densely overlapping small targets
and fuzzy targets, this paper proposes a dynamic\-attention scale\-sequence
fusion algorithm \(DASSF\) for small target detection in aerial images. First, we
propose a dynamic scale sequence feature fusion \(DSSFF\) module that improves
the up\-sampling mechanism and reduces computational load. Secondly, a x\-small
object detection head is specially added to enhance the detection capability of
small targets. Finally, in order to improve the expressive ability of targets
of different types and sizes, we use the dynamic head \(DyHead\). The model we
proposed solves the problem of small target detection in aerial images and can
be applied to multiple different versions of the YOLO algorithm, which is
universal. Experimental results show that when the DASSF method is applied to
YOLOv8, compared to YOLOv8n, on the VisDrone\-2019 and DIOR datasets, the model
shows an increase of 9.2% and 2.4% in the mean average precision \(mAP\),
respectively, and outperforms the current mainstream methods.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2406.12285v1)

---


## YOLO\-FEDER FusionNet: A Novel Deep Learning Architecture for Drone Detection

**发布日期**：2024-06-17

**作者**：Tamara R. Lenhard

**摘要**：Predominant methods for image\-based drone detection frequently rely on
employing generic object detection algorithms like YOLOv5. While proficient in
identifying drones against homogeneous backgrounds, these algorithms often
struggle in complex, highly textured environments. In such scenarios, drones
seamlessly integrate into the background, creating camouflage effects that
adversely affect the detection quality. To address this issue, we introduce a
novel deep learning architecture called YOLO\-FEDER FusionNet. Unlike
conventional approaches, YOLO\-FEDER FusionNet combines generic object detection
methods with the specialized strength of camouflage object detection techniques
to enhance drone detection capabilities. Comprehensive evaluations of
YOLO\-FEDER FusionNet show the efficiency of the proposed model and demonstrate
substantial improvements in both reducing missed detections and false alarms.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2406.11641v1)

---


## ChildDiffusion: Unlocking the Potential of Generative AI and Controllable Augmentations for Child Facial Data using Stable Diffusion and Large Language Models

**发布日期**：2024-06-17

**作者**：Muhammad Ali Farooq

**摘要**：In this research work we have proposed high\-level ChildDiffusion framework
capable of generating photorealistic child facial samples and further embedding
several intelligent augmentations on child facial data using short text
prompts, detailed textual guidance from LLMs, and further image to image
transformation using text guidance control conditioning thus providing an
opportunity to curate fully synthetic large scale child datasets. The framework
is validated by rendering high\-quality child faces representing ethnicity data,
micro expressions, face pose variations, eye blinking effects, facial
accessories, different hair colours and styles, aging, multiple and different
child gender subjects in a single frame. Addressing privacy concerns regarding
child data acquisition requires a comprehensive approach that involves legal,
ethical, and technological considerations. Keeping this in view this framework
can be adapted to synthesise child facial data which can be effectively used
for numerous downstream machine learning tasks. The proposed method circumvents
common issues encountered in generative AI tools, such as temporal
inconsistency and limited control over the rendered outputs. As an exemplary
use case we have open\-sourced child ethnicity data consisting of 2.5k child
facial samples of five different classes which includes African, Asian, White,
South Asian/ Indian, and Hispanic races by deploying the model in production
inference phase. The rendered data undergoes rigorous qualitative as well as
quantitative tests to cross validate its efficacy and further fine\-tuning Yolo
architecture for detecting and classifying child ethnicity as an exemplary
downstream machine learning task.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2406.11592v1)

---


## YOLOv1 to YOLOv10: A comprehensive review of YOLO variants and their application in the agricultural domain

**发布日期**：2024-06-14

**作者**：Mujadded Al Rabbani Alif

**摘要**：This survey investigates the transformative potential of various YOLO
variants, from YOLOv1 to the state\-of\-the\-art YOLOv10, in the context of
agricultural advancements. The primary objective is to elucidate how these
cutting\-edge object detection models can re\-energise and optimize diverse
aspects of agriculture, ranging from crop monitoring to livestock management.
It aims to achieve key objectives, including the identification of contemporary
challenges in agriculture, a detailed assessment of YOLO's incremental
advancements, and an exploration of its specific applications in agriculture.
This is one of the first surveys to include the latest YOLOv10, offering a
fresh perspective on its implications for precision farming and sustainable
agricultural practices in the era of Artificial Intelligence and automation.
Further, the survey undertakes a critical analysis of YOLO's performance,
synthesizes existing research, and projects future trends. By scrutinizing the
unique capabilities packed in YOLO variants and their real\-world applications,
this survey provides valuable insights into the evolving relationship between
YOLO variants and agriculture. The findings contribute towards a nuanced
understanding of the potential for precision farming and sustainable
agricultural practices, marking a significant step forward in the integration
of advanced object detection technologies within the agricultural sector.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2406.10139v1)

---


## A Deep Learning Approach to Detect Complete Safety Equipment For Construction Workers Based On YOLOv7

**发布日期**：2024-06-11

**作者**：Md. Shariful Islam

**摘要**：In the construction sector, ensuring worker safety is of the utmost
significance. In this study, a deep learning\-based technique is presented for
identifying safety gear worn by construction workers, such as helmets, goggles,
jackets, gloves, and footwears. The recommended approach uses the YOLO v7 \(You
Only Look Once\) object detection algorithm to precisely locate these safety
items. The dataset utilized in this work consists of labeled images split into
training, testing and validation sets. Each image has bounding box labels that
indicate where the safety equipment is located within the image. The model is
trained to identify and categorize the safety equipment based on the labeled
dataset through an iterative training approach. We used custom dataset to train
this model. Our trained model performed admirably well, with good precision,
recall, and F1\-score for safety equipment recognition. Also, the model's
evaluation produced encouraging results, with a mAP@0.5 score of 87.7\\%. The
model performs effectively, making it possible to quickly identify safety
equipment violations on building sites. A thorough evaluation of the outcomes
reveals the model's advantages and points up potential areas for development.
By offering an automatic and trustworthy method for safety equipment detection,
this research makes a contribution to the fields of computer vision and
workplace safety. The proposed deep learning\-based approach will increase
safety compliance and reduce the risk of accidents in the construction industry


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2406.07707v2)

---


## Advancing Roadway Sign Detection with YOLO Models and Transfer Learning

**发布日期**：2024-06-11

**作者**：Selvia Nafaa

**摘要**：Roadway signs detection and recognition is an essential element in the
Advanced Driving Assistant Systems \(ADAS\). Several artificial intelligence
methods have been used widely among of them YOLOv5 and YOLOv8. In this paper,
we used a modified YOLOv5 and YOLOv8 to detect and classify different roadway
signs under different illumination conditions. Experimental results indicated
that for the YOLOv8 model, varying the number of epochs and batch size yields
consistent MAP50 scores, ranging from 94.6% to 97.1% on the testing set. The
YOLOv5 model demonstrates competitive performance, with MAP50 scores ranging
from 92.4% to 96.9%. These results suggest that both models perform well across
different training setups, with YOLOv8 generally achieving slightly higher
MAP50 scores. These findings suggest that both models can perform well under
different training setups, offering valuable insights for practitioners seeking
reliable and adaptable solutions in object detection applications.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2406.09437v1)

---

