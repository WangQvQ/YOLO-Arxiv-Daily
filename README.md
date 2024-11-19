# 每日从arXiv中获取最新YOLO相关论文


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


## Deep Learning Models for UAV\-Assisted Bridge Inspection: A YOLO Benchmark Analysis

**发布日期**：2024-11-07

**作者**：Trong\-Nhan Phan

**摘要**：Visual inspections of bridges are critical to ensure their safety and
identify potential failures early. This inspection process can be rapidly and
accurately automated by using unmanned aerial vehicles \(UAVs\) integrated with
deep learning models. However, choosing an appropriate model that is
lightweight enough to integrate into the UAV and fulfills the strict
requirements for inference time and accuracy is challenging. Therefore, our
work contributes to the advancement of this model selection process by
conducting a benchmark of 23 models belonging to the four newest YOLO variants
\(YOLOv5, YOLOv6, YOLOv7, YOLOv8\) on COCO\-Bridge\-2021\+, a dataset for bridge
details detection. Through comprehensive benchmarking, we identify YOLOv8n,
YOLOv7tiny, YOLOv6m, and YOLOv6m6 as the models offering an optimal balance
between accuracy and processing speed, with mAP@50 scores of 0.803, 0.837,
0.853, and 0.872, and inference times of 5.3ms, 7.5ms, 14.06ms, and 39.33ms,
respectively. Our findings accelerate the model selection process for UAVs,
enabling more efficient and reliable bridge inspections.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2411.04475v1)

---


## Self\-supervised cross\-modality learning for uncertainty\-aware object detection and recognition in applications which lack pre\-labelled training data

**发布日期**：2024-11-05

**作者**：Irum Mehboob

**摘要**：This paper shows how an uncertainty\-aware, deep neural network can be trained
to detect, recognise and localise objects in 2D RGB images, in applications
lacking annotated train\-ng datasets. We propose a self\-supervising
teacher\-student pipeline, in which a relatively simple teacher classifier,
trained with only a few labelled 2D thumbnails, automatically processes a
larger body of unlabelled RGB\-D data to teach a student network based on a
modified YOLOv3 architecture. Firstly, 3D object detection with back projection
is used to automatically extract and teach 2D detection and localisation
information to the student network. Secondly, a weakly supervised 2D thumbnail
classifier, with minimal training on a small number of hand\-labelled images, is
used to teach object category recognition. Thirdly, we use a Gaussian Process
GP to encode and teach a robust uncertainty estimation functionality, so that
the student can output confidence scores with each categorization. The
resulting student significantly outperforms the same YOLO architecture trained
directly on the same amount of labelled data. Our GP\-based approach yields
robust and meaningful uncertainty estimations for complex industrial object
classifications. The end\-to\-end network is also capable of real\-time
processing, needed for robotics applications. Our method can be applied to many
important industrial tasks, where labelled datasets are typically unavailable.
In this paper, we demonstrate an example of detection, localisation, and object
category recognition of nuclear mixed\-waste materials in highly cluttered and
unstructured scenes. This is critical for robotic sorting and handling of
legacy nuclear waste, which poses complex environmental remediation challenges
in many nuclearised nations.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2411.03082v1)

---


## ERUP\-YOLO: Enhancing Object Detection Robustness for Adverse Weather Condition by Unified Image\-Adaptive Processing

**发布日期**：2024-11-05

**作者**：Yuka Ogino

**摘要**：We propose an image\-adaptive object detection method for adverse weather
conditions such as fog and low\-light. Our framework employs differentiable
preprocessing filters to perform image enhancement suitable for later\-stage
object detections. Our framework introduces two differentiable filters: a
B\\'ezier curve\-based pixel\-wise \(BPW\) filter and a kernel\-based local \(KBL\)
filter. These filters unify the functions of classical image processing filters
and improve performance of object detection. We also propose a domain\-agnostic
data augmentation strategy using the BPW filter. Our method does not require
data\-specific customization of the filter combinations, parameter ranges, and
data augmentation. We evaluate our proposed approach, called Enhanced
Robustness by Unified Image Processing \(ERUP\)\-YOLO, by applying it to the
YOLOv3 detector. Experiments on adverse weather datasets demonstrate that our
proposed filters match or exceed the expressiveness of conventional methods and
our ERUP\-YOLO achieved superior performance in a wide range of adverse weather
conditions, including fog and low\-light conditions.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2411.02799v1)

---


## One\-Stage\-TFS: Thai One\-Stage Fingerspelling Dataset for Fingerspelling Recognition Frameworks

**发布日期**：2024-11-05

**作者**：Siriwiwat Lata

**摘要**：The Thai One\-Stage Fingerspelling \(One\-Stage\-TFS\) dataset is a comprehensive
resource designed to advance research in hand gesture recognition, explicitly
focusing on the recognition of Thai sign language. This dataset comprises 7,200
images capturing 15 one\-stage consonant gestures performed by undergraduate
students from Rajabhat Maha Sarakham University, Thailand. The contributors
include both expert students from the Special Education Department with
proficiency in Thai sign language and students from other departments without
prior sign language experience. Images were collected between July and December
2021 using a DSLR camera, with contributors demonstrating hand gestures against
both simple and complex backgrounds. The One\-Stage\-TFS dataset presents
challenges in detecting and recognizing hand gestures, offering opportunities
to develop novel end\-to\-end recognition frameworks. Researchers can utilize
this dataset to explore deep learning methods, such as YOLO, EfficientDet,
RetinaNet, and Detectron, for hand detection, followed by feature extraction
and recognition using techniques like convolutional neural networks,
transformers, and adaptive feature fusion networks. The dataset is accessible
via the Mendeley Data repository and supports a wide range of applications in
computer science, including deep learning, computer vision, and pattern
recognition, thereby encouraging further innovation and exploration in these
fields.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2411.02768v1)

---

