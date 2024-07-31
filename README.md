# 每日从arXiv中获取最新YOLO相关论文


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
architecture, and the energy efficiency is improved by 5.7. Code:
https://github.com/BICLab/SpikeYOLO


**代码链接**：https://github.com/BICLab/SpikeYOLO

**论文链接**：[阅读更多](http://arxiv.org/abs/2407.20708v1)

---


## Uncertainty\-Rectified YOLO\-SAM for Weakly Supervised ICH Segmentation

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

**论文链接**：[阅读更多](http://arxiv.org/abs/2407.20461v1)

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


## YOLOv10 for Automated Fracture Detection in Pediatric Wrist Trauma X\-rays

**发布日期**：2024-07-22

**作者**：Ammar Ahmed

**摘要**：Wrist fractures are highly prevalent among children and can significantly
impact their daily activities, such as attending school, participating in
sports, and performing basic self\-care tasks. If not treated properly, these
fractures can result in chronic pain, reduced wrist functionality, and other
long\-term complications. Recently, advancements in object detection have shown
promise in enhancing fracture detection, with systems achieving accuracy
comparable to, or even surpassing, that of human radiologists. The YOLO series,
in particular, has demonstrated notable success in this domain. This study is
the first to provide a thorough evaluation of various YOLOv10 variants to
assess their performance in detecting pediatric wrist fractures using the
GRAZPEDWRI\-DX dataset. It investigates how changes in model complexity, scaling
the architecture, and implementing a dual\-label assignment strategy can enhance
detection performance. Experimental results indicate that our trained model
achieved mean average precision \(mAP@50\-95\) of 51.9\\% surpassing the current
YOLOv9 benchmark of 43.3\\% on this dataset. This represents an improvement of
8.6\\%. The implementation code is publicly available at
https://github.com/ammarlodhi255/YOLOv10\-Fracture\-Detection


**代码链接**：https://github.com/ammarlodhi255/YOLOv10-Fracture-Detection

**论文链接**：[阅读更多](http://arxiv.org/abs/2407.15689v1)

---


## Multiple Object Detection and Tracking in Panoramic Videos for Cycling Safety Analysis

**发布日期**：2024-07-21

**作者**：Jingwei Guo

**摘要**：Panoramic cycling videos can record 360\{\\deg\} views around the cyclists.
Thus, it is essential to conduct automatic road user analysis on them using
computer vision models to provide data for studies on cycling safety. However,
the features of panoramic data such as severe distortions, large number of
small objects and boundary continuity have brought great challenges to the
existing CV models, including poor performance and evaluation methods that are
no longer applicable. In addition, due to the lack of data with annotations, it
is not easy to re\-train the models.
  In response to these problems, the project proposed and implemented a
three\-step methodology: \(1\) improve the prediction performance of the
pre\-trained object detection models on panoramic data by projecting the
original image into 4 perspective sub\-images; \(2\) introduce supports for
boundary continuity and category information into DeepSORT, a commonly used
multiple object tracking model, and set an improved detection model as its
detector; \(3\) using the tracking results, develop an application for detecting
the overtaking behaviour of the surrounding vehicles.
  Evaluated on the panoramic cycling dataset built by the project, the proposed
methodology improves the average precision of YOLO v5m6 and Faster RCNN\-FPN
under any input resolution setting. In addition, it raises MOTA and IDF1 of
DeepSORT by 7.6\\% and 9.7\\% respectively. When detecting the overtakes in the
test videos, it achieves the F\-score of 0.88.
  The code is available on GitHub at github.com/cuppp1998/360\_object\_tracking
to ensure the reproducibility and further improvements of results.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2407.15199v1)

---


## GreenStableYolo: Optimizing Inference Time and Image Quality of Text\-to\-Image Generation

**发布日期**：2024-07-20

**作者**：Jingzhi Gong

**摘要**：Tuning the parameters and prompts for improving AI\-based text\-to\-image
generation has remained a substantial yet unaddressed challenge. Hence we
introduce GreenStableYolo, which improves the parameters and prompts for Stable
Diffusion to both reduce GPU inference time and increase image generation
quality using NSGA\-II and Yolo.
  Our experiments show that despite a relatively slight trade\-off \(18%\) in
image quality compared to StableYolo \(which only considers image quality\),
GreenStableYolo achieves a substantial reduction in inference time \(266% less\)
and a 526% higher hypervolume, thereby advancing the state\-of\-the\-art for
text\-to\-image generation.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2407.14982v1)

---

