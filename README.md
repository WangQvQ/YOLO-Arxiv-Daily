# 每日从arXiv中获取最新YOLO相关论文


## Mamba YOLO: SSMs\-Based YOLO For Object Detection

**发布日期**：2024-06-09

**作者**：Zeyu Wang

**摘要**：Propelled by the rapid advancement of deep learning technologies, the YOLO
series has set a new benchmark for real\-time object detectors. Researchers have
continuously explored innovative applications of reparameterization, efficient
layer aggregation networks, and anchor\-free techniques on the foundation of
YOLO. To further enhance detection performance, Transformer\-based structures
have been introduced, significantly expanding the model's receptive field and
achieving notable performance gains. However, such improvements come at a cost,
as the quadratic complexity of the self\-attention mechanism increases the
computational burden of the model. Fortunately, the emergence of State Space
Models \(SSM\) as an innovative technology has effectively mitigated the issues
caused by quadratic complexity. In light of these advancements, we introduce
Mamba\-YOLO a novel object detection model based on SSM. Mamba\-YOLO not only
optimizes the SSM foundation but also adapts specifically for object detection
tasks. Given the potential limitations of SSM in sequence modeling, such as
insufficient receptive field and weak image locality, we have designed the
LSBlock and RGBlock. These modules enable more precise capture of local image
dependencies and significantly enhance the robustness of the model. Extensive
experimental results on the publicly available benchmark datasets COCO and VOC
demonstrate that Mamba\-YOLO surpasses the existing YOLO series models in both
performance and competitiveness, showcasing its substantial potential and
competitive edge.The PyTorch code is available
at:\\url\{https://github.com/HZAI\-ZJNU/Mamba\-YOLO\}


**代码链接**：https://github.com/HZAI-ZJNU/Mamba-YOLO}

**论文链接**：[阅读更多](http://arxiv.org/abs/2406.05835v1)

---


## LW\-DETR: A Transformer Replacement to YOLO for Real\-Time Detection

**发布日期**：2024-06-05

**作者**：Qiang Chen

**摘要**：In this paper, we present a light\-weight detection transformer, LW\-DETR,
which outperforms YOLOs for real\-time object detection. The architecture is a
simple stack of a ViT encoder, a projector, and a shallow DETR decoder. Our
approach leverages recent advanced techniques, such as training\-effective
techniques, e.g., improved loss and pretraining, and interleaved window and
global attentions for reducing the ViT encoder complexity. We improve the ViT
encoder by aggregating multi\-level feature maps, and the intermediate and final
feature maps in the ViT encoder, forming richer feature maps, and introduce
window\-major feature map organization for improving the efficiency of
interleaved attention computation. Experimental results demonstrate that the
proposed approach is superior over existing real\-time detectors, e.g., YOLO and
its variants, on COCO and other benchmark datasets. Code and models are
available at \(https://github.com/Atten4Vis/LW\-DETR\).


**代码链接**：https://github.com/Atten4Vis/LW-DETR).

**论文链接**：[阅读更多](http://arxiv.org/abs/2406.03459v1)

---


## Open\-YOLO 3D: Towards Fast and Accurate Open\-Vocabulary 3D Instance Segmentation

**发布日期**：2024-06-04

**作者**：Mohamed El Amine Boudjoghra

**摘要**：Recent works on open\-vocabulary 3D instance segmentation show strong promise,
but at the cost of slow inference speed and high computation requirements. This
high computation cost is typically due to their heavy reliance on 3D clip
features, which require computationally expensive 2D foundation models like
Segment Anything \(SAM\) and CLIP for multi\-view aggregation into 3D. As a
consequence, this hampers their applicability in many real\-world applications
that require both fast and accurate predictions. To this end, we propose a fast
yet accurate open\-vocabulary 3D instance segmentation approach, named Open\-YOLO
3D, that effectively leverages only 2D object detection from multi\-view RGB
images for open\-vocabulary 3D instance segmentation. We address this task by
generating class\-agnostic 3D masks for objects in the scene and associating
them with text prompts. We observe that the projection of class\-agnostic 3D
point cloud instances already holds instance information; thus, using SAM might
only result in redundancy that unnecessarily increases the inference time. We
empirically find that a better performance of matching text prompts to 3D masks
can be achieved in a faster fashion with a 2D object detector. We validate our
Open\-YOLO 3D on two benchmarks, ScanNet200 and Replica, under two scenarios:
\(i\) with ground truth masks, where labels are required for given object
proposals, and \(ii\) with class\-agnostic 3D proposals generated from a 3D
proposal network. Our Open\-YOLO 3D achieves state\-of\-the\-art performance on
both datasets while obtaining up to $\\sim$16$\\times$ speedup compared to the
best existing method in literature. On ScanNet200 val. set, our Open\-YOLO 3D
achieves mean average precision \(mAP\) of 24.7\\% while operating at 22 seconds
per scene. Code and model are available at github.com/aminebdj/OpenYOLO3D.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2406.02548v1)

---


## "Pass the butter": A study on desktop\-classic multitasking robotic arm based on advanced YOLOv7 and BERT

**发布日期**：2024-05-27

**作者**：Haohua Que

**摘要**：In recent years, various intelligent autonomous robots have begun to appear
in daily life and production. Desktop\-level robots are characterized by their
flexible deployment, rapid response, and suitability for light workload
environments. In order to meet the current societal demand for service robot
technology, this study proposes using a miniaturized desktop\-level robot \(by
ROS\) as a carrier, locally deploying a natural language model \(NLP\-BERT\), and
integrating visual recognition \(CV\-YOLO\) and speech recognition technology
\(ASR\-Whisper\) as inputs to achieve autonomous decision\-making and rational
action by the desktop robot. Three comprehensive experiments were designed to
validate the robotic arm, and the results demonstrate excellent performance
using this approach across all three experiments. In Task 1, the execution
rates for speech recognition and action performance were 92.6% and 84.3%,
respectively. In Task 2, the highest execution rates under the given conditions
reached 92.1% and 84.6%, while in Task 3, the highest execution rates were
95.2% and 80.8%, respectively. Therefore, it can be concluded that the proposed
solution integrating ASR, NLP, and other technologies on edge devices is
feasible and provides a technical and engineering foundation for realizing
multimodal desktop\-level robots.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2405.17250v1)

---


## Evaluation of Resource\-Efficient Crater Detectors on Embedded Systems

**发布日期**：2024-05-27

**作者**：Simon Vellas

**摘要**：Real\-time analysis of Martian craters is crucial for mission\-critical
operations, including safe landings and geological exploration. This work
leverages the latest breakthroughs for on\-the\-edge crater detection aboard
spacecraft. We rigorously benchmark several YOLO networks using a Mars craters
dataset, analyzing their performance on embedded systems with a focus on
optimization for low\-power devices. We optimize this process for a new wave of
cost\-effective, commercial\-off\-the\-shelf\-based smaller satellites.
Implementations on diverse platforms, including Google Coral Edge TPU, AMD
Versal SoC VCK190, Nvidia Jetson Nano and Jetson AGX Orin, undergo a detailed
trade\-off analysis. Our findings identify optimal network\-device pairings,
enhancing the feasibility of crater detection on resource\-constrained hardware
and setting a new precedent for efficient and resilient extraterrestrial
imaging. Code at: https://github.com/billpsomas/mars\_crater\_detection.


**代码链接**：https://github.com/billpsomas/mars_crater_detection.

**论文链接**：[阅读更多](http://arxiv.org/abs/2405.16953v1)

---


## Enhancing Pollinator Conservation towards Agriculture 4.0: Monitoring of Bees through Object Recognition

**发布日期**：2024-05-24

**作者**：Ajay John Alex

**摘要**：In an era of rapid climate change and its adverse effects on food production,
technological intervention to monitor pollinator conservation is of paramount
importance for environmental monitoring and conservation for global food
security. The survival of the human species depends on the conservation of
pollinators. This article explores the use of Computer Vision and Object
Recognition to autonomously track and report bee behaviour from images. A novel
dataset of 9664 images containing bees is extracted from video streams and
annotated with bounding boxes. With training, validation and testing sets
\(6722, 1915, and 997 images, respectively\), the results of the COCO\-based YOLO
model fine\-tuning approaches show that YOLOv5m is the most effective approach
in terms of recognition accuracy. However, YOLOv5s was shown to be the most
optimal for real\-time bee detection with an average processing and inference
time of 5.1ms per video frame at the cost of slightly lower ability. The
trained model is then packaged within an explainable AI interface, which
converts detection events into timestamped reports and charts, with the aim of
facilitating use by non\-technical users such as expert stakeholders from the
apiculture industry towards informing responsible consumption and production.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2405.15428v1)

---


## YOLOv10: Real\-Time End\-to\-End Object Detection

**发布日期**：2024-05-23

**作者**：Ao Wang

**摘要**：Over the past years, YOLOs have emerged as the predominant paradigm in the
field of real\-time object detection owing to their effective balance between
computational cost and detection performance. Researchers have explored the
architectural designs, optimization objectives, data augmentation strategies,
and others for YOLOs, achieving notable progress. However, the reliance on the
non\-maximum suppression \(NMS\) for post\-processing hampers the end\-to\-end
deployment of YOLOs and adversely impacts the inference latency. Besides, the
design of various components in YOLOs lacks the comprehensive and thorough
inspection, resulting in noticeable computational redundancy and limiting the
model's capability. It renders the suboptimal efficiency, along with
considerable potential for performance improvements. In this work, we aim to
further advance the performance\-efficiency boundary of YOLOs from both the
post\-processing and model architecture. To this end, we first present the
consistent dual assignments for NMS\-free training of YOLOs, which brings
competitive performance and low inference latency simultaneously. Moreover, we
introduce the holistic efficiency\-accuracy driven model design strategy for
YOLOs. We comprehensively optimize various components of YOLOs from both
efficiency and accuracy perspectives, which greatly reduces the computational
overhead and enhances the capability. The outcome of our effort is a new
generation of YOLO series for real\-time end\-to\-end object detection, dubbed
YOLOv10. Extensive experiments show that YOLOv10 achieves state\-of\-the\-art
performance and efficiency across various model scales. For example, our
YOLOv10\-S is 1.8$\\times$ faster than RT\-DETR\-R18 under the similar AP on COCO,
meanwhile enjoying 2.8$\\times$ smaller number of parameters and FLOPs. Compared
with YOLOv9\-C, YOLOv10\-B has 46\\% less latency and 25\\% fewer parameters for
the same performance.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2405.14458v1)

---


## Bangladeshi Native Vehicle Detection in Wild

**发布日期**：2024-05-20

**作者**：Bipin Saha

**摘要**：The success of autonomous navigation relies on robust and precise vehicle
recognition, hindered by the scarcity of region\-specific vehicle detection
datasets, impeding the development of context\-aware systems. To advance
terrestrial object detection research, this paper proposes a native vehicle
detection dataset for the most commonly appeared vehicle classes in Bangladesh.
17 distinct vehicle classes have been taken into account, with fully annotated
81542 instances of 17326 images. Each image width is set to at least 1280px.
The dataset's average vehicle bounding box\-to\-image ratio is 4.7036. This
Bangladesh Native Vehicle Dataset \(BNVD\) has accounted for several
geographical, illumination, variety of vehicle sizes, and orientations to be
more robust on surprised scenarios. In the context of examining the BNVD
dataset, this work provides a thorough assessment with four successive You Only
Look Once \(YOLO\) models, namely YOLO v5, v6, v7, and v8. These dataset's
effectiveness is methodically evaluated and contrasted with other vehicle
datasets already in use. The BNVD dataset exhibits mean average precision\(mAP\)
at 50% intersection over union \(IoU\) is 0.848 corresponding precision and
recall values of 0.841 and 0.774. The research findings indicate a mAP of 0.643
at an IoU range of 0.5 to 0.95. The experiments show that the BNVD dataset
serves as a reliable representation of vehicle distribution and presents
considerable complexities.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2405.12150v1)

---


## Drone\-type\-Set: Drone types detection benchmark for drone detection and tracking

**发布日期**：2024-05-16

**作者**：Kholoud AlDosari

**摘要**：The Unmanned Aerial Vehicles \(UAVs\) market has been significantly growing and
Considering the availability of drones at low\-cost prices the possibility of
misusing them, for illegal purposes such as drug trafficking, spying, and
terrorist attacks posing high risks to national security, is rising. Therefore,
detecting and tracking unauthorized drones to prevent future attacks that
threaten lives, facilities, and security, become a necessity. Drone detection
can be performed using different sensors, while image\-based detection is one of
them due to the development of artificial intelligence techniques. However,
knowing unauthorized drone types is one of the challenges due to the lack of
drone types datasets. For that, in this paper, we provide a dataset of various
drones as well as a comparison of recognized object detection models on the
proposed dataset including YOLO algorithms with their different versions, like,
v3, v4, and v5 along with the Detectronv2. The experimental results of
different models are provided along with a description of each method. The
collected dataset can be found in
https://drive.google.com/drive/folders/1EPOpqlF4vG7hp4MYnfAecVOsdQ2JwBEd?usp=share\_link


**代码链接**：https://drive.google.com/drive/folders/1EPOpqlF4vG7hp4MYnfAecVOsdQ2JwBEd?usp=share_link

**论文链接**：[阅读更多](http://arxiv.org/abs/2405.10398v1)

---


## WeedScout: Real\-Time Autonomous blackgrass Classification and Mapping using dedicated hardware

**发布日期**：2024-05-12

**作者**：Matthew Gazzard

**摘要**：Blackgrass \(Alopecurus myosuroides\) is a competitive weed that has
wide\-ranging impacts on food security by reducing crop yields and increasing
cultivation costs. In addition to the financial burden on agriculture, the
application of herbicides as a preventive to blackgrass can negatively affect
access to clean water and sanitation. The WeedScout project introduces a
Real\-Rime Autonomous Black\-Grass Classification and Mapping \(RT\-ABGCM\), a
cutting\-edge solution tailored for real\-time detection of blackgrass, for
precision weed management practices. Leveraging Artificial Intelligence \(AI\)
algorithms, the system processes live image feeds, infers blackgrass density,
and covers two stages of maturation. The research investigates the deployment
of You Only Look Once \(YOLO\) models, specifically the streamlined YOLOv8 and
YOLO\-NAS, accelerated at the edge with the NVIDIA Jetson Nano \(NJN\). By
optimising inference speed and model performance, the project advances the
integration of AI into agricultural practices, offering potential solutions to
challenges such as herbicide resistance and environmental impact. Additionally,
two datasets and model weights are made available to the research community,
facilitating further advancements in weed detection and precision farming
technologies.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2405.07349v1)

---

