# 每日从arXiv中获取最新YOLO相关论文


## Impact Analysis of Data Drift Towards The Development of Safety\-Critical Automotive System

**发布日期**：2024-08-07

**作者**：Md Shahi Amran Hossain

**摘要**：A significant part of contemporary research in autonomous vehicles is
dedicated to the development of safety critical systems where state\-of\-the\-art
artificial intelligence \(AI\) algorithms, like computer vision \(CV\), can play a
major role. Vision models have great potential for the real\-time detection of
numerous traffic signs and obstacles, which is essential to avoid accidents and
protect human lives. Despite vast potential, computer vision\-based systems have
critical safety concerns too if the traffic condition drifts over time. This
paper represents an analysis of how data drift can affect the performance of
vision models in terms of traffic sign detection. The novelty in this research
is provided through a YOLO\-based fusion model that is trained with drifted data
from the CARLA simulator and delivers a robust and enhanced performance in
object detection. The enhanced model showed an average precision of 97.5\\%
compared to the 58.27\\% precision of the original model. A detailed performance
review of the original and fusion models is depicted in the paper, which
promises to have a significant impact on safety\-critical automotive systems.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2408.04476v1)

---


## Monitoring of Hermit Crabs Using drone\-captured imagery and Deep Learning based Super\-Resolution Reconstruction and Improved YOLOv8

**发布日期**：2024-08-07

**作者**：Fan Zhao

**摘要**：Hermit crabs play a crucial role in coastal ecosystems by dispersing seeds,
cleaning up debris, and disturbing soil. They serve as vital indicators of
marine environmental health, responding to climate change and pollution.
Traditional survey methods, like quadrat sampling, are labor\-intensive,
time\-consuming, and environmentally dependent. This study presents an
innovative approach combining UAV\-based remote sensing with Super\-Resolution
Reconstruction \(SRR\) and the CRAB\-YOLO detection network, a modification of
YOLOv8s, to monitor hermit crabs. SRR enhances image quality by addressing
issues such as motion blur and insufficient resolution, significantly improving
detection accuracy over conventional low\-resolution fuzzy images. The CRAB\-YOLO
network integrates three improvements for detection accuracy, hermit crab
characteristics, and computational efficiency, achieving state\-of\-the\-art
\(SOTA\) performance compared to other mainstream detection models. The RDN
networks demonstrated the best image reconstruction performance, and CRAB\-YOLO
achieved a mean average precision \(mAP\) of 69.5% on the SRR test set, a 40%
improvement over the conventional Bicubic method with a magnification factor of
4. These results indicate that the proposed method is effective in detecting
hermit crabs, offering a cost\-effective and automated solution for extensive
hermit crab monitoring, thereby aiding coastal benthos conservation.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2408.03559v1)

---


## GUI Element Detection Using SOTA YOLO Deep Learning Models

**发布日期**：2024-08-07

**作者**：Seyed Shayan Daneshvar

**摘要**：Detection of Graphical User Interface \(GUI\) elements is a crucial task for
automatic code generation from images and sketches, GUI testing, and GUI
search. Recent studies have leveraged both old\-fashioned and modern computer
vision \(CV\) techniques. Oldfashioned methods utilize classic image processing
algorithms \(e.g. edge detection and contour detection\) and modern methods use
mature deep learning solutions for general object detection tasks. GUI element
detection, however, is a domain\-specific case of object detection, in which
objects overlap more often, and are located very close to each other, plus the
number of object classes is considerably lower, yet there are more objects in
the images compared to natural images. Hence, the studies that have been
carried out on comparing various object detection models, might not apply to
GUI element detection. In this study, we evaluate the performance of the four
most recent successful YOLO models for general object detection tasks on GUI
element detection and investigate their accuracy performance in detecting
various GUI elements.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2408.03507v1)

---


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

