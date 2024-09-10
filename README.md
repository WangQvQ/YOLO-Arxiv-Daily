# 每日从arXiv中获取最新YOLO相关论文


## BFA\-YOLO: Balanced multiscale object detection network for multi\-view building facade attachments detection

**发布日期**：2024-09-06

**作者**：Yangguang Chen

**摘要**：Detection of building facade attachments such as doors, windows, balconies,
air conditioner units, billboards, and glass curtain walls plays a pivotal role
in numerous applications. Building facade attachments detection aids in
vbuilding information modeling \(BIM\) construction and meeting Level of Detail 3
\(LOD3\) standards. Yet, it faces challenges like uneven object distribution,
small object detection difficulty, and background interference. To counter
these, we propose BFA\-YOLO, a model for detecting facade attachments in
multi\-view images. BFA\-YOLO incorporates three novel innovations: the Feature
Balanced Spindle Module \(FBSM\) for addressing uneven distribution, the Target
Dynamic Alignment Task Detection Head \(TDATH\) aimed at improving small object
detection, and the Position Memory Enhanced Self\-Attention Mechanism \(PMESA\) to
combat background interference, with each component specifically designed to
solve its corresponding challenge. Detection efficacy of deep network models
deeply depends on the dataset's characteristics. Existing open source datasets
related to building facades are limited by their single perspective, small
image pool, and incomplete category coverage. We propose a novel method for
building facade attachments detection dataset construction and construct the
BFA\-3D dataset for facade attachments detection. The BFA\-3D dataset features
multi\-view, accurate labels, diverse categories, and detailed classification.
BFA\-YOLO surpasses YOLOv8 by 1.8% and 2.9% in mAP@0.5 on the multi\-view BFA\-3D
and street\-view Facade\-WHU datasets, respectively. These results underscore
BFA\-YOLO's superior performance in detecting facade attachments.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2409.04025v1)

---


## YOLO\-CL cluster detection in the Rubin/LSST DC2 simulation

**发布日期**：2024-09-05

**作者**：Kirill Grishin

**摘要**：LSST will provide galaxy cluster catalogs up to z$\\sim$1 that can be used to
constrain cosmological models once their selection function is well\-understood.
We have applied the deep convolutional network YOLO for CLuster detection
\(YOLO\-CL\) to LSST simulations from the Dark Energy Science Collaboration Data
Challenge 2 \(DC2\), and characterized the LSST YOLO\-CL cluster selection
function. We have trained and validated the network on images from a hybrid
sample of \(1\) clusters observed in the Sloan Digital Sky Survey and detected
with the red\-sequence Matched\-filter Probabilistic Percolation, and \(2\)
simulated DC2 dark matter haloes with masses $M\_\{200c\} > 10^\{14\} M\_\{\\odot\}$. We
quantify the completeness and purity of the YOLO\-CL cluster catalog with
respect to DC2 haloes with $M\_\{200c\} > 10^\{14\} M\_\{\\odot\}$. The YOLO\-CL cluster
catalog is 100% and 94% complete for halo mass $M\_\{200c\} > 10^\{14.6\} M\_\{\\odot\}$
at $0.2<z<0.8$, and $M\_\{200c\} > 10^\{14\} M\_\{\\odot\}$ and redshift $z \\lesssim 1$,
respectively, with only 6% false positive detections. All the false positive
detections are dark matter haloes with $ 10^\{13.4\} M\_\{\\odot\} \\lesssim M\_\{200c\}
\\lesssim 10^\{14\} M\_\{\\odot\}$. The YOLO\-CL selection function is almost flat with
respect to the halo mass at $0.2 \\lesssim z \\lesssim 0.9$. The overall
performance of YOLO\-CL is comparable or better than other cluster detection
methods used for current and future optical and infrared surveys. YOLO\-CL shows
better completeness for low mass clusters when compared to current detections
in surveys using the Sunyaev Zel'dovich effect, and detects clusters at higher
redshifts than X\-ray\-based catalogs. The strong advantage of YOLO\-CL over
traditional galaxy cluster detection techniques is that it works directly on
images and does not require photometric and photometric redshift catalogs, nor
does it need to mask stellar sources and artifacts.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2409.03333v1)

---


## YOLO\-PPA based Efficient Traffic Sign Detection for Cruise Control in Autonomous Driving

**发布日期**：2024-09-05

**作者**：Jingyu Zhang

**摘要**：It is very important to detect traffic signs efficiently and accurately in
autonomous driving systems. However, the farther the distance, the smaller the
traffic signs. Existing object detection algorithms can hardly detect these
small scaled signs.In addition, the performance of embedded devices on vehicles
limits the scale of detection models.To address these challenges, a YOLO PPA
based traffic sign detection algorithm is proposed in this paper.The
experimental results on the GTSDB dataset show that compared to the original
YOLO, the proposed method improves inference efficiency by 11.2%. The mAP 50 is
also improved by 93.2%, which demonstrates the effectiveness of the proposed
YOLO PPA.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2409.03320v1)

---


## YoloTag: Vision\-based Robust UAV Navigation with Fiducial Markers

**发布日期**：2024-09-03

**作者**：Sourav Raxit

**摘要**：By harnessing fiducial markers as visual landmarks in the environment,
Unmanned Aerial Vehicles \(UAVs\) can rapidly build precise maps and navigate
spaces safely and efficiently, unlocking their potential for fluent
collaboration and coexistence with humans. Existing fiducial marker methods
rely on handcrafted feature extraction, which sacrifices accuracy. On the other
hand, deep learning pipelines for marker detection fail to meet real\-time
runtime constraints crucial for navigation applications. In this work, we
propose YoloTag \\textemdash a real\-time fiducial marker\-based localization
system. YoloTag uses a lightweight YOLO v8 object detector to accurately detect
fiducial markers in images while meeting the runtime constraints needed for
navigation. The detected markers are then used by an efficient
perspective\-n\-point algorithm to estimate UAV states. However, this
localization system introduces noise, causing instability in trajectory
tracking. To suppress noise, we design a higher\-order Butterworth filter that
effectively eliminates noise through frequency domain analysis. We evaluate our
algorithm through real\-robot experiments in an indoor environment, comparing
the trajectory tracking performance of our method against other approaches in
terms of several distance metrics.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2409.02334v1)

---


## DS MYOLO: A Reliable Object Detector Based on SSMs for Driving Scenarios

**发布日期**：2024-09-02

**作者**：Yang Li

**摘要**：Accurate real\-time object detection enhances the safety of advanced
driver\-assistance systems, making it an essential component in driving
scenarios. With the rapid development of deep learning technology, CNN\-based
YOLO real\-time object detectors have gained significant attention. However, the
local focus of CNNs results in performance bottlenecks. To further enhance
detector performance, researchers have introduced Transformer\-based
self\-attention mechanisms to leverage global receptive fields, but their
quadratic complexity incurs substantial computational costs. Recently, Mamba,
with its linear complexity, has made significant progress through global
selective scanning. Inspired by Mamba's outstanding performance, we propose a
novel object detector: DS MYOLO. This detector captures global feature
information through a simplified selective scanning fusion block \(SimVSS Block\)
and effectively integrates the network's deep features. Additionally, we
introduce an efficient channel attention convolution \(ECAConv\) that enhances
cross\-channel feature interaction while maintaining low computational
complexity. Extensive experiments on the CCTSDB 2021 and VLD\-45 driving
scenarios datasets demonstrate that DS MYOLO exhibits significant potential and
competitive advantage among similarly scaled YOLO series real\-time object
detectors.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2409.01093v1)

---


## A method for detecting dead fish on large water surfaces based on improved YOLOv10

**发布日期**：2024-08-31

**作者**：Qingbin Tian

**摘要**：Dead fish frequently appear on the water surface due to various factors. If
not promptly detected and removed, these dead fish can cause significant issues
such as water quality deterioration, ecosystem damage, and disease
transmission. Consequently, it is imperative to develop rapid and effective
detection methods to mitigate these challenges. Conventional methods for
detecting dead fish are often constrained by manpower and time limitations,
struggling to effectively manage the intricacies of aquatic environments. This
paper proposes an end\-to\-end detection model built upon an enhanced YOLOv10
framework, designed specifically to swiftly and precisely detect deceased fish
across extensive water surfaces.Key enhancements include: \(1\) Replacing
YOLOv10's backbone network with FasterNet to reduce model complexity while
maintaining high detection accuracy; \(2\) Improving feature fusion in the Neck
section through enhanced connectivity methods and replacing the original C2f
module with CSPStage modules; \(3\) Adding a compact target detection head to
enhance the detection performance of smaller objects. Experimental results
demonstrate significant improvements in P\(precision\), R\(recall\), and AP\(average
precision\) compared to the baseline model YOLOv10n. Furthermore, our model
outperforms other models in the YOLO series by significantly reducing model
size and parameter count, while sustaining high inference speed and achieving
optimal AP performance. The model facilitates rapid and accurate detection of
dead fish in large\-scale aquaculture systems. Finally, through ablation
experiments, we systematically analyze and assess the contribution of each
model component to the overall system performance.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2409.00388v1)

---


## FA\-YOLO: Research On Efficient Feature Selection YOLO Improved Algorithm Based On FMDS and AGMF Modules

**发布日期**：2024-08-29

**作者**：Yukang Huo

**摘要**：Over the past few years, the YOLO series of models has emerged as one of the
dominant methodologies in the realm of object detection. Many studies have
advanced these baseline models by modifying their architectures, enhancing data
quality, and developing new loss functions. However, current models still
exhibit deficiencies in processing feature maps, such as overlooking the fusion
of cross\-scale features and a static fusion approach that lacks the capability
for dynamic feature adjustment. To address these issues, this paper introduces
an efficient Fine\-grained Multi\-scale Dynamic Selection Module \(FMDS Module\),
which applies a more effective dynamic feature selection and fusion method on
fine\-grained multi\-scale feature maps, significantly enhancing the detection
accuracy of small, medium, and large\-sized targets in complex environments.
Furthermore, this paper proposes an Adaptive Gated Multi\-branch Focus Fusion
Module \(AGMF Module\), which utilizes multiple parallel branches to perform
complementary fusion of various features captured by the gated unit branch,
FMDS Module branch, and TripletAttention branch. This approach further enhances
the comprehensiveness, diversity, and integrity of feature fusion. This paper
has integrated the FMDS Module, AGMF Module, into Yolov9 to develop a novel
object detection model named FA\-YOLO. Extensive experimental results show that
under identical experimental conditions, FA\-YOLO achieves an outstanding 66.1%
mean Average Precision \(mAP\) on the PASCAL VOC 2007 dataset, representing 1.0%
improvement over YOLOv9's 65.1%. Additionally, the detection accuracies of
FA\-YOLO for small, medium, and large targets are 44.1%, 54.6%, and 70.8%,
respectively, showing improvements of 2.0%, 3.1%, and 0.9% compared to YOLOv9's
42.1%, 51.5%, and 69.9%.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2408.16313v1)

---


## microYOLO: Towards Single\-Shot Object Detection on Microcontrollers

**发布日期**：2024-08-28

**作者**：Mark Deutel

**摘要**：This work\-in\-progress paper presents results on the feasibility of
single\-shot object detection on microcontrollers using YOLO. Single\-shot object
detectors like YOLO are widely used, however due to their complexity mainly on
larger GPU\-based platforms. We present microYOLO, which can be used on Cortex\-M
based microcontrollers, such as the OpenMV H7 R2, achieving about 3.5 FPS when
classifying 128x128 RGB images while using less than 800 KB Flash and less than
350 KB RAM. Furthermore, we share experimental results for three different
object detection tasks, analyzing the accuracy of microYOLO on them.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2408.15865v1)

---


## YOLO\-Stutter: End\-to\-end Region\-Wise Speech Dysfluency Detection

**发布日期**：2024-08-27

**作者**：Xuanru Zhou

**摘要**：Dysfluent speech detection is the bottleneck for disordered speech analysis
and spoken language learning. Current state\-of\-the\-art models are governed by
rule\-based systems which lack efficiency and robustness, and are sensitive to
template design. In this paper, we propose YOLO\-Stutter: a first end\-to\-end
method that detects dysfluencies in a time\-accurate manner. YOLO\-Stutter takes
imperfect speech\-text alignment as input, followed by a spatial feature
aggregator, and a temporal dependency extractor to perform region\-wise boundary
and class predictions. We also introduce two dysfluency corpus, VCTK\-Stutter
and VCTK\-TTS, that simulate natural spoken dysfluencies including repetition,
block, missing, replacement, and prolongation. Our end\-to\-end method achieves
state\-of\-the\-art performance with a minimum number of trainable parameters for
on both simulated data and real aphasia speech. Code and datasets are
open\-sourced at https://github.com/rorizzz/YOLO\-Stutter


**代码链接**：https://github.com/rorizzz/YOLO-Stutter

**论文链接**：[阅读更多](http://arxiv.org/abs/2408.15297v2)

---


## Intraoperative Glioma Segmentation with YOLO \+ SAM for Improved Accuracy in Tumor Resection

**发布日期**：2024-08-27

**作者**：Samir Kassam

**摘要**：Gliomas, a common type of malignant brain tumor, present significant surgical
challenges due to their similarity to healthy tissue. Preoperative Magnetic
Resonance Imaging \(MRI\) images are often ineffective during surgery due to
factors such as brain shift, which alters the position of brain structures and
tumors. This makes real\-time intraoperative MRI \(ioMRI\) crucial, as it provides
updated imaging that accounts for these shifts, ensuring more accurate tumor
localization and safer resections. This paper presents a deep learning pipeline
combining You Only Look Once Version 8 \(YOLOv8\) and Segment Anything Model
Vision Transformer\-base \(SAM ViT\-b\) to enhance glioma detection and
segmentation during ioMRI. Our model was trained using the Brain Tumor
Segmentation 2021 \(BraTS 2021\) dataset, which includes standard magnetic
resonance imaging \(MRI\) images, and noise\-augmented MRI images that simulate
ioMRI images. Noised MRI images are harder for a deep learning pipeline to
segment, but they are more representative of surgical conditions. Achieving a
Dice Similarity Coefficient \(DICE\) score of 0.79, our model performs comparably
to state\-of\-the\-art segmentation models tested on noiseless data. This
performance demonstrates the model's potential to assist surgeons in maximizing
tumor resection and improving surgical outcomes.


**代码链接**：摘要中未找到代码链接。

**论文链接**：[阅读更多](http://arxiv.org/abs/2408.14847v1)

---

