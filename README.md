# 每日从arXiv中获取最新YOLO相关论文


## AssemAI: Interpretable Image\-Based Anomaly Detection for Manufacturing Pipelines

**发布日期**：2024-08-05

**作者**：Renjith Prasad

**摘要**：Anomaly detection in manufacturing pipelines remains a critical challenge,
intensified by the complexity and variability of industrial environments. This
paper introduces AssemAI, an interpretable image\-based anomaly detection system
tailored for smart manufacturing pipelines. Our primary contributions include
the creation of a tailored image dataset and the development of a custom object
detection model, YOLO\-FF, designed explicitly for anomaly detection in
manufacturing assembly environments. Utilizing the preprocessed image dataset
derived from an industry\-focused rocket assembly pipeline, we address the
challenge of imbalanced image data and demonstrate the importance of
image\-based methods in anomaly detection. The proposed approach leverages
domain knowledge in data preparation, model development and reasoning. We
compare our method against several baselines, including simple CNN and custom
Visual Transformer \(ViT\) models, showcasing the effectiveness of our custom
data preparation and pretrained CNN integration. Additionally, we incorporate
explainability techniques at both user and model levels, utilizing ontology for
user\-friendly explanations and SCORE\-CAM for in\-depth feature and model
analysis. Finally, the model was also deployed in a real\-time setting. Our
results include ablation studies on the baselines, providing a comprehensive
evaluation of the proposed system. This work highlights the broader impact of
advanced image\-based anomaly detection in enhancing the reliability and
efficiency of smart manufacturing processes.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2408.02181v1)

---


## CAF\-YOLO: A Robust Framework for Multi\-Scale Lesion Detection in Biomedical Imagery

**发布日期**：2024-08-04

**作者**：Zilin Chen

**摘要**：Object detection is of paramount importance in biomedical image analysis,
particularly for lesion identification. While current methodologies are
proficient in identifying and pinpointing lesions, they often lack the
precision needed to detect minute biomedical entities \(e.g., abnormal cells,
lung nodules smaller than 3 mm\), which are critical in blood and lung
pathology. To address this challenge, we propose CAF\-YOLO, based on the YOLOv8
architecture, a nimble yet robust method for medical object detection that
leverages the strengths of convolutional neural networks \(CNNs\) and
transformers. To overcome the limitation of convolutional kernels, which have a
constrained capacity to interact with distant information, we introduce an
attention and convolution fusion module \(ACFM\). This module enhances the
modeling of both global and local features, enabling the capture of long\-term
feature dependencies and spatial autocorrelation. Additionally, to improve the
restricted single\-scale feature aggregation inherent in feed\-forward networks
\(FFN\) within transformer architectures, we design a multi\-scale neural network
\(MSNN\). This network improves multi\-scale information aggregation by extracting
features across diverse scales. Experimental evaluations on widely used
datasets, such as BCCD and LUNA16, validate the rationale and efficacy of
CAF\-YOLO. This methodology excels in detecting and precisely locating diverse
and intricate micro\-lesions within biomedical imagery. Our codes are available
at https://github.com/xiaochen925/CAF\-YOLO.


**代码链接**：https://github.com/xiaochen925/CAF-YOLO.

**论文链接**：[阅读更多](http://arxiv.org/abs/2408.01897v1)

---


## Spatial Transformer Network YOLO Model for Agricultural Object Detection

**发布日期**：2024-07-31

**作者**：Yash Zambre

**摘要**：Object detection plays a crucial role in the field of computer vision by
autonomously identifying and locating objects of interest. The You Only Look
Once \(YOLO\) model is an effective single\-shot detector. However, YOLO faces
challenges in cluttered or partially occluded scenes and can struggle with
small, low\-contrast objects. We propose a new method that integrates spatial
transformer networks \(STNs\) into YOLO to improve performance. The proposed
STN\-YOLO aims to enhance the model's effectiveness by focusing on important
areas of the image and improving the spatial invariance of the model before the
detection process. Our proposed method improved object detection performance
both qualitatively and quantitatively. We explore the impact of different
localization networks within the STN module as well as the robustness of the
model across different spatial transformations. We apply the STN\-YOLO on
benchmark datasets for Agricultural object detection as well as a new dataset
from a state\-of\-the\-art plant phenotyping greenhouse facility. Our code and
dataset are publicly available.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2407.21652v1)

---


## A Comparative Analysis of YOLOv5, YOLOv8, and YOLOv10 in Kitchen Safety

**发布日期**：2024-07-30

**作者**：Athulya Sundaresan Geetha

**摘要**：Knife safety in the kitchen is essential for preventing accidents or injuries
with an emphasis on proper handling, maintenance, and storage methods. This
research presents a comparative analysis of three YOLO models, YOLOv5, YOLOv8,
and YOLOv10, to detect the hazards involved in handling knife, concentrating
mainly on ensuring fingers are curled while holding items to be cut and that
hands should only be in contact with knife handle avoiding the blade.
Precision, recall, F\-score, and normalized confusion matrix are used to
evaluate the performance of the models. The results indicate that YOLOv5
performed better than the other two models in identifying the hazard of
ensuring hands only touch the blade, while YOLOv8 excelled in detecting the
hazard of curled fingers while holding items. YOLOv5 and YOLOv8 performed
almost identically in recognizing classes such as hand, knife, and vegetable,
whereas YOLOv5, YOLOv8, and YOLOv10 accurately identified the cutting board.
This paper provides insights into the advantages and shortcomings of these
models in real\-world settings. Moreover, by detailing the optimization of YOLO
architectures for safe knife handling, this study promotes the development of
increased accuracy and efficiency in safety surveillance systems.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2407.20872v1)

---


## Integer\-Valued Training and Spike\-Driven Inference Spiking Neural Network for High\-performance and Energy\-efficient Object Detection

**发布日期**：2024-07-30

**作者**：Xinhao Luo

**摘要**：Brain\-inspired Spiking Neural Networks \(SNNs\) have bio\-plausibility and
low\-power advantages over Artificial Neural Networks \(ANNs\). Applications of
SNNs are currently limited to simple classification tasks because of their poor
performance. In this work, we focus on bridging the performance gap between
ANNs and SNNs on object detection. Our design revolves around network
architecture and spiking neuron. First, the overly complex module design causes
spike degradation when the YOLO series is converted to the corresponding
spiking version. We design a SpikeYOLO architecture to solve this problem by
simplifying the vanilla YOLO and incorporating meta SNN blocks. Second, object
detection is more sensitive to quantization errors in the conversion of
membrane potentials into binary spikes by spiking neurons. To address this
challenge, we design a new spiking neuron that activates Integer values during
training while maintaining spike\-driven by extending virtual timesteps during
inference. The proposed method is validated on both static and neuromorphic
object detection datasets. On the static COCO dataset, we obtain 66.2% mAP@50
and 48.9% mAP@50:95, which is \+15.0% and \+18.7% higher than the prior
state\-of\-the\-art SNN, respectively. On the neuromorphic Gen1 dataset, we
achieve 67.2% mAP@50, which is \+2.5% greater than the ANN with equivalent
architecture, and the energy efficiency is improved by 5.7\*. Code:
https://github.com/BICLab/SpikeYOLO


**代码链接**：https://github.com/BICLab/SpikeYOLO

**论文链接**：[阅读更多](http://arxiv.org/abs/2407.20708v3)

---


## Weakly Supervised Intracranial Hemorrhage Segmentation with YOLO and an Uncertainty Rectified Segment Anything Model

**发布日期**：2024-07-29

**作者**：Pascal Spiegler

**摘要**：Intracranial hemorrhage \(ICH\) is a life\-threatening condition that requires
rapid and accurate diagnosis to improve treatment outcomes and patient survival
rates. Recent advancements in supervised deep learning have greatly improved
the analysis of medical images, but often rely on extensive datasets with
high\-quality annotations, which are costly, time\-consuming, and require medical
expertise to prepare. To mitigate the need for large amounts of expert\-prepared
segmentation data, we have developed a novel weakly supervised ICH segmentation
method that utilizes the YOLO object detection model and an
uncertainty\-rectified Segment Anything Model \(SAM\). In addition, we have
proposed a novel point prompt generator for this model to further improve
segmentation results with YOLO\-predicted bounding box prompts. Our approach
achieved a high accuracy of 0.933 and an AUC of 0.796 in ICH detection, along
with a mean Dice score of 0.629 for ICH segmentation, outperforming existing
weakly supervised and popular supervised \(UNet and Swin\-UNETR\) approaches.
Overall, the proposed method provides a robust and accurate alternative to the
more commonly used supervised techniques for ICH quantification without
requiring refined segmentation ground truths during model training.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2407.20461v2)

---


## Octave\-YOLO: Cross frequency detection network with octave convolution

**发布日期**：2024-07-29

**作者**：Sangjune Shin

**摘要**：Despite the rapid advancement of object detection algorithms, processing
high\-resolution images on embedded devices remains a significant challenge.
Theoretically, the fully convolutional network architecture used in current
real\-time object detectors can handle all input resolutions. However, the
substantial computational demands required to process high\-resolution images
render them impractical for real\-time applications. To address this issue,
real\-time object detection models typically downsample the input image for
inference, leading to a loss of detail and decreased accuracy. In response, we
developed Octave\-YOLO, designed to process high\-resolution images in real\-time
within the constraints of embedded systems. We achieved this through the
introduction of the cross frequency partial network \(CFPNet\), which divides the
input feature map into low\-resolution, low\-frequency, and high\-resolution,
high\-frequency sections. This configuration enables complex operations such as
convolution bottlenecks and self\-attention to be conducted exclusively on
low\-resolution feature maps while simultaneously preserving the details in
high\-resolution maps. Notably, this approach not only dramatically reduces the
computational demands of convolution tasks but also allows for the integration
of attention modules, which are typically challenging to implement in real\-time
applications, with minimal additional cost. Additionally, we have incorporated
depthwise separable convolution into the core building blocks and downsampling
layers to further decrease latency. Experimental results have shown that
Octave\-YOLO matches the performance of YOLOv8 while significantly reducing
computational demands. For example, in 1080x1080 resolution, Octave\-YOLO\-N is
1.56 times faster than YOLOv8, achieving nearly the same accuracy on the COCO
dataset with approximately 40 percent fewer parameters and FLOPs.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2407.19746v1)

---


## Real Time American Sign Language Detection Using Yolo\-v9

**发布日期**：2024-07-25

**作者**：Amna Imran

**摘要**：This paper focuses on real\-time American Sign Language Detection. YOLO is a
convolutional neural network \(CNN\) based model, which was first released in
2015. In recent years, it gained popularity for its real\-time detection
capabilities. Our study specifically targets YOLO\-v9 model, released in 2024.
As the model is newly introduced, not much work has been done on it, especially
not in Sign Language Detection. Our paper provides deep insight on how YOLO\- v9
works and better than previous model.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2407.17950v1)

---


## Hierarchical Object Detection and Recognition Framework for Practical Plant Disease Diagnosis

**发布日期**：2024-07-25

**作者**：Kohei Iwano

**摘要**：Recently, object detection methods \(OD; e.g., YOLO\-based models\) have been
widely utilized in plant disease diagnosis. These methods demonstrate
robustness to distance variations and excel at detecting small lesions compared
to classification methods \(CL; e.g., CNN models\). However, there are issues
such as low diagnostic performance for hard\-to\-detect diseases and high
labeling costs. Additionally, since healthy cases cannot be explicitly trained,
there is a risk of false positives. We propose the Hierarchical object
detection and recognition framework \(HODRF\), a sophisticated and highly
integrated two\-stage system that combines the strengths of both OD and CL for
plant disease diagnosis. In the first stage, HODRF uses OD to identify regions
of interest \(ROIs\) without specifying the disease. In the second stage, CL
diagnoses diseases surrounding the ROIs. HODRF offers several advantages: \(1\)
Since OD detects only one type of ROI, HODRF can detect diseases with limited
training images by leveraging its ability to identify other lesions. \(2\) While
OD over\-detects healthy cases, HODRF significantly reduces these errors by
using CL in the second stage. \(3\) CL's accuracy improves in HODRF as it
identifies diagnostic targets given as ROIs, making it less vulnerable to size
changes. \(4\) HODRF benefits from CL's lower annotation costs, allowing it to
learn from a larger number of images. We implemented HODRF using YOLOv7 for OD
and EfficientNetV2 for CL and evaluated its performance on a large\-scale
dataset \(4 crops, 20 diseased and healthy classes, 281K images\). HODRF
outperformed YOLOv7 alone by 5.8 to 21.5 points on healthy data and 0.6 to 7.5
points on macro F1 scores, and it improved macro F1 by 1.1 to 7.2 points over
EfficientNetV2.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2407.17906v1)

---


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

