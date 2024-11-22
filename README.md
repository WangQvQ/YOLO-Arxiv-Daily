# 每日从arXiv中获取最新YOLO相关论文


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


## SL\-YOLO: A Stronger and Lighter Drone Target Detection Model

**发布日期**：2024-11-18

**作者**：Defan Chen

**摘要**：Detecting small objects in complex scenes, such as those captured by drones,
is a daunting challenge due to the difficulty in capturing the complex features
of small targets. While the YOLO family has achieved great success in large
target detection, its performance is less than satisfactory when faced with
small targets. Because of this, this paper proposes a revolutionary model
SL\-YOLO \(Stronger and Lighter YOLO\) that aims to break the bottleneck of small
target detection. We propose the Hierarchical Extended Path Aggregation Network
\(HEPAN\), a pioneering cross\-scale feature fusion method that can ensure
unparalleled detection accuracy even in the most challenging environments. At
the same time, without sacrificing detection capabilities, we design the C2fDCB
lightweight module and add the SCDown downsampling module to greatly reduce the
model's parameters and computational complexity. Our experimental results on
the VisDrone2019 dataset reveal a significant improvement in performance, with
mAP@0.5 jumping from 43.0% to 46.9% and mAP@0.5:0.95 increasing from 26.0% to
28.9%. At the same time, the model parameters are reduced from 11.1M to 9.6M,
and the FPS can reach 132, making it an ideal solution for real\-time small
object detection in resource\-constrained environments.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2411.11477v1)

---


## Zero\-Shot Automatic Annotation and Instance Segmentation using LLM\-Generated Datasets: Eliminating Field Imaging and Manual Annotation for Deep Learning Model Development

**发布日期**：2024-11-18

**作者**：Ranjan Sapkota

**摘要**：Currently, deep learning\-based instance segmentation for various applications
\(e.g., Agriculture\) is predominantly performed using a labor\-intensive process
involving extensive field data collection using sophisticated sensors, followed
by careful manual annotation of images, presenting significant logistical and
financial challenges to researchers and organizations. The process also slows
down the model development and training process. In this study, we presented a
novel method for deep learning\-based instance segmentation of apples in
commercial orchards that eliminates the need for labor\-intensive field data
collection and manual annotation. Utilizing a Large Language Model \(LLM\), we
synthetically generated orchard images and automatically annotated them using
the Segment Anything Model \(SAM\) integrated with a YOLO11 base model. This
method significantly reduces reliance on physical sensors and manual data
processing, presenting a major advancement in "Agricultural AI". The synthetic,
auto\-annotated dataset was used to train the YOLO11 model for Apple instance
segmentation, which was then validated on real orchard images. The results
showed that the automatically generated annotations achieved a Dice Coefficient
of 0.9513 and an IoU of 0.9303, validating the accuracy and overlap of the mask
annotations. All YOLO11 configurations, trained solely on these synthetic
datasets with automated annotations, accurately recognized and delineated
apples, highlighting the method's efficacy. Specifically, the YOLO11m\-seg
configuration achieved a mask precision of 0.902 and a mask mAP@50 of 0.833 on
test images collected from a commercial orchard. Additionally, the YOLO11l\-seg
configuration outperformed other models in validation on 40 LLM\-generated
images, achieving the highest mask precision and mAP@50 metrics.
  Keywords: YOLO, SAM, SAMv2, YOLO11, YOLOv11, Segment Anything, YOLO\-SAM


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2411.11285v1)

---


## Diachronic Document Dataset for Semantic Layout Analysis

**发布日期**：2024-11-15

**作者**：Thibault Clérice

**摘要**：We present a novel, open\-access dataset designed for semantic layout
analysis, built to support document recreation workflows through mapping with
the Text Encoding Initiative \(TEI\) standard. This dataset includes 7,254
annotated pages spanning a large temporal range \(1600\-2024\) of digitised and
born\-digital materials across diverse document types \(magazines, papers from
sciences and humanities, PhD theses, monographs, plays, administrative reports,
etc.\) sorted into modular subsets. By incorporating content from different
periods and genres, it addresses varying layout complexities and historical
changes in document structure. The modular design allows domain\-specific
configurations. We evaluate object detection models on this dataset, examining
the impact of input size and subset\-based training. Results show that a
1280\-pixel input size for YOLO is optimal and that training on subsets
generally benefits from incorporating them into a generic model rather than
fine\-tuning pre\-trained weights.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2411.10068v1)

---


## Integrating Object Detection Modality into Visual Language Model for Enhanced Autonomous Driving Agent

**发布日期**：2024-11-08

**作者**：Linfeng He

**摘要**：In this paper, we propose a novel framework for enhancing visual
comprehension in autonomous driving systems by integrating visual language
models \(VLMs\) with additional visual perception module specialised in object
detection. We extend the Llama\-Adapter architecture by incorporating a
YOLOS\-based detection network alongside the CLIP perception network, addressing
limitations in object detection and localisation. Our approach introduces
camera ID\-separators to improve multi\-view processing, crucial for
comprehensive environmental awareness. Experiments on the DriveLM visual
question answering challenge demonstrate significant improvements over baseline
models, with enhanced performance in ChatGPT scores, BLEU scores, and CIDEr
metrics, indicating closeness of model answer to ground truth. Our method
represents a promising step towards more capable and interpretable autonomous
driving systems. Possible safety enhancement enabled by detection modality is
also discussed.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2411.05898v1)

---


## SynDroneVision: A Synthetic Dataset for Image\-Based Drone Detection

**发布日期**：2024-11-08

**作者**：Tamara R. Lenhard

**摘要**：Developing robust drone detection systems is often constrained by the limited
availability of large\-scale annotated training data and the high costs
associated with real\-world data collection. However, leveraging synthetic data
generated via game engine\-based simulations provides a promising and
cost\-effective solution to overcome this issue. Therefore, we present
SynDroneVision, a synthetic dataset specifically designed for RGB\-based drone
detection in surveillance applications. Featuring diverse backgrounds, lighting
conditions, and drone models, SynDroneVision offers a comprehensive training
foundation for deep learning algorithms. To evaluate the dataset's
effectiveness, we perform a comparative analysis across a selection of recent
YOLO detection models. Our findings demonstrate that SynDroneVision is a
valuable resource for real\-world data enrichment, achieving notable
enhancements in model performance and robustness, while significantly reducing
the time and costs of real\-world data acquisition. SynDroneVision will be
publicly released upon paper acceptance.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2411.05633v1)

---

