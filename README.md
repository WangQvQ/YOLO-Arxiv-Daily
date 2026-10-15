<div align="center">

# YOLO ArXiv Daily

[![Daily Papers](https://img.shields.io/badge/📅-每日更新-blue)]()
[![arXiv](https://img.shields.io/badge/arXiv-最新论文-red)](https://arxiv.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)

*自动追踪 YOLO 相关最新论文，提供中英文双语摘要*

</div>

---

## 📑 论文列表

> ### 1. Identifying Planetary Nebulae Candidates from IPHAS and VPHAS\+ Surveys via YOLO Deep Learning Models
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-10-14 |
> | 👤 作者 | Titouan Sérandour |
>
> **📄 英文摘要：**
> We present a deep learning detection method capable of effectively identifying planetary nebulae \(PNe\) on IPHAS and VPHAS\+ surveys. Our detection pipeline is based on the YOLO architecture and is adaptable to other datasets. Using the HiPS system, we preprocess and augment the $r$, $i$, and $Hα$ bands to produce two red, green, and blue \(RGB\) image datasets: \(1\) $640 times 640$ px images at full resolution, and \(2\) $1 280 times 1 280$ px images designed to capture extended objects up to $0.2$ deg. Images of $sim 2600$ PNe are labelled using the HASH database. The loss function and evaluation metrics are adapted to the diffuse and low\-contrast nature of PNe; a very low intersection over union \(IoU\) threshold \($1%$\) is used. A $k$\-fold cross validation shows that the YOLOv12m model achieves an average precision of $0.979$ on the first dataset. At a confidence threshold of $0.15$, it achieves a precision rate of $98.7%$ and a recall rate of $97.4%$. We deployed our system \(on $1280 times 1280$ images\) on the full surveys' coverage \($sim 3755$ deg$^2$\), and present a preliminary analysis of the detection results. The system is robust against artifacts such as CCD corner defects and HiPS tile joints. Our convolutional\-neural\-network\-based model achieved excellent results on the test set, comparable to\-or exceeding\-that of a recent Swin Transformer approach. However, a challenge persists when applying deep learning models to extensive and densely populated Galactic fields: the development of an automated system that can effectively discriminate PNe from mimics that human experts are able to distinguish \(e.g., H II regions\). We outline possible improvements for future work.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2610.19760v1)

---

> ### 2. Augmenting Real\-time Visualization in Minimally Invasive Surgery with Particle Filtering and Depth\-Aware Occlusion Handling
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-10-13 |
> | 👤 作者 | Anuska Sarkar |
>
> **📄 英文摘要：**
> We present a real\-time, hybrid framework for occlusion handling and depth\-aware tracking in minimally invasive surgeries \(MIS\). Occlusions caused by surgical instruments often obscure the region of interest \(ROI\), affecting surgical precision. To address this, our approach integrates enhanced You Only Look Once \(YOLO\) based detection, Particle Filtering \(PF\), and monocular depth estimation using Multiple Depth Estimation Accuracy with Single Network \(MiDaS\). Occlusion is detected using a distance\-based threshold between instruments and ROI. Experimental results demonstrate robust instrument detection with a confidence score of 0.85, effective occlusion event handling, and accurate depth prediction with an MAE of 0.34 cm and RMSE of 0.53 cm. Independent phantom\-based validation against stereo depth measurements yielded an MAE of 0.65 cm and RMSE of 0.66 cm. The framework further achieved an occlusion detection accuracy of 81.95% while maintaining real\-time performance at 38.7 FPS. Preliminary surgeon feedback indicates that the proposed visualization framework is clinically relevant and holds promise for future intraoperative image\-guided navigation and decision support in MIS.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2610.18422v1)

---

> ### 3. Squeeze\-Aware\-YOLO: Early Channel Recalibration and Multi\-Scale Fusion for Real\-Time Traffic Sign Detection
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-10-11 |
> | 👤 作者 | Mizbaul Haque Maruf |
>
> **📄 英文摘要：**
> Traffic sign detection plays a vital role in intelligent transportation systems, supporting the safe operation of advanced driver assistance systems and autonomous vehicles. Despite significant progress in real\-time object detection, accurately identifying traffic signs under challenging conditions remains difficult. Variations in lighting, adverse weather, motion blur, and partial occlusion often reduce detection performance, especially for small or visually degraded signs where important spatial information can be lost. To address these challenges, we propose Squeeze\-Aware\-YOLO, a lightweight traffic sign detection framework that introduces a multi\-branch bottleneck module combining convolutional kernels of different sizes with a Squeeze\-and\-Excitation mechanism to improve feature extraction and adaptive channel weighting. In addition, we integrate the Multi\-Branch Auxiliary Feature Pyramid Network and Re\-parameterized Heterogeneous Multi\-Scale modules to strengthen multi\-scale feature representation and feature fusion while preserving real\-time inference efficiency for practical deployment. The proposed approach is evaluated on the Bangladeshi Traffic Sign Detection Dataset, the CSUST Chinese Traffic Sign Detection Benchmark 2021, and the German Traffic Sign Detection Benchmark, achieving mAP@0.5 scores of 74.7%, 79.4%, and 84.0%, respectively, while consistently outperforming recent state\-of\-the\-art YOLO\-based traffic sign detectors. These findings demonstrate that combining multi\-scale receptive fields, channel attention, and enhanced feature fusion enables more accurate and robust traffic sign detection while maintaining the computational efficiency required for real\-world deployment.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2610.15604v1)

---

> ### 4. JMT\-YOLO: Multi\-task Domain Adaptation via Segmentation\-Guided Consistency Learning
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-10-09 |
> | 👤 作者 | Zhenhua Yu |
>
> **📄 英文摘要：**
> Domain adaptive object detection \(DAOD\) aims to reduce performance degradation due to distribution shifts between training \(source\) and testing \(target\) datasets. However, owing to insufficient modeling of fine\-grained features and limited instance\-level alignment, existing approaches become less effective under complex environmental conditions, such as extreme weather conditions and object occlusion. We overcome these challenges by proposing JMT\-YOLO, a novel joint multi\-task learning framework for DAOD. JMT\-YOLO is based on the Mean Teacher architecture. It introduces an auxiliary semantic segmentation task and a masking mechanism to enhance the modeling of foreground\-background semantic boundaries in the target domain. Specifically, fine\-grained feature degradation is reduced by introducing a multiscale context\-aware segmentation head equipped with a dense connection structure to effectively capture local and global contextual features. A mask\-consistency loss is formulated by applying random masking operations to target images, thereby guiding the network to learn structural context and improving instance\-level feature alignment. Extensive experiments on three benchmark domain adaptation tasks \(Pascal VOC to Clipart1k, Cityscapes to Foggy Cityscapes, and KITTI to Cityscapes\) demonstrate that JMT\-YOLO achieves detection performances of 46.9%, 57.3%, and 61.0%, respectively. These results demonstrate superior performance over existing state\-of\-the\-art and baseline methods, thereby validating the effectiveness and generalization of JMT\-YOLO.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2610.12901v1)

---

> ### 5. Post\-Training Semantic Lifting for 3D Gaussian Splatting: Separating Detector, Lifting and Representation Error
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-10-06 |
> | 👤 作者 | Iván Verdugo Guerra |
>
> **📄 英文摘要：**
> The same Gaussian of a 3D Gaussian Splatting model is seen from many views, and these views do not always agree on the class it belongs to. The Gaussian may be occluded in some of them, and the confidence of the detector is not the same from one view to another. The ground truth, on the other hand, is given as an annotated mesh, because two training runs do not produce the same Gaussians. In this work, we propose a post\-training lifting method that works with one target class at a time and combines the information coming from all the views. Target and non\-target evidence are accumulated simultaneously, weighted by the visibility of each Gaussian in each view. After that, the Gaussians are filtered with two thresholds: a main threshold $β$ selects the high\-confidence seeds, and a lower one $γβ$ adds the connected components around them. For the evaluation, the labels are transferred from the Gaussians to the mesh vertices that are both visible and annotated. With this design, we can separate three sources of error: the 2D detector, the lifting and the transfer between representations. The thresholds and the transfer operator are chosen on seven Replica validation scenes, and the method is evaluated on ten held\-out ScanNet\+\+ scenes with the same values for every scene and class. The mean mIoU on the validation scenes was 0.93 with masks from the dataset annotations and 0.65 with YOLO masks, and on the ScanNet\+\+ test scenes it was 0.80 and 0.54. Compared with thresholding the evidence per view, as a previous version of the method did, the fraction improves the test mIoU by 0.24 and makes it possible to use a single threshold for all the classes and scenes of both datasets. Finally, the error analysis shows that most of the remaining error comes from the detector.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2610.08756v1)

---

<div align="center">

*由 [YOLO-Arxiv-Daily](https://github.com/WangQvQ/YOLO-Arxiv-Daily) 自动生成*

</div>