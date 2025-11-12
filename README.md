# 每日从arXiv中获取最新YOLO相关论文


## DAMO\-YOLO : A Report on Real\-Time Object Detection Design / 

发布日期：2022-11-23

作者：Xianzhe Xu

摘要：In this report, we present a fast and accurate object detection method dubbed DAMO\-YOLO, which achieves higher performance than the state\-of\-the\-art YOLO series. DAMO\-YOLO is extended from YOLO with some new technologies, including Neural Architecture Search \(NAS\), efficient Reparameterized Generalized\-FPN \(RepGFPN\), a lightweight head with AlignedOTA label assignment, and distillation enhancement. In particular, we use MAE\-NAS, a method guided by the principle of maximum entropy, to search our detection backbone under the constraints of low latency and high performance, producing ResNet/CSP\-like structures with spatial pyramid pooling and focus modules. In the design of necks and heads, we follow the rule of \`\`large neck, small head''.We import Generalized\-FPN with accelerated queen\-fusion to build the detector neck and upgrade its CSPNet with efficient layer aggregation networks \(ELAN\) and reparameterization. Then we investigate how detector head size affects detection performance and find that a heavy neck with only one task projection layer would yield better results.In addition, AlignedOTA is proposed to solve the misalignment problem in label assignment. And a distillation schema is introduced to improve performance to a higher level. Based on these new techs, we build a suite of models at various scales to meet the needs of different scenarios. For general industry requirements, we propose DAMO\-YOLO\-T/S/M/L. They can achieve 43.6/47.7/50.2/51.9 mAPs on COCO with the latency of 2.78/3.83/5.62/7.95 ms on T4 GPUs respectively. Additionally, for edge devices with limited computing power, we have also proposed DAMO\-YOLO\-Ns/Nm/Nl lightweight models. They can achieve 32.3/38.2/40.5 mAPs on COCO with the latency of 4.08/5.05/6.69 ms on X86\-CPU. Our proposed general and lightweight models have outperformed other YOLO series models in their respective application scenarios.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2211.15444v4)

---


## YOLO\-World: Real\-Time Open\-Vocabulary Object Detection / 

发布日期：2024-01-30

作者：Tianheng Cheng

摘要：The You Only Look Once \(YOLO\) series of detectors have established themselves as efficient and practical tools. However, their reliance on predefined and trained object categories limits their applicability in open scenarios. Addressing this limitation, we introduce YOLO\-World, an innovative approach that enhances YOLO with open\-vocabulary detection capabilities through vision\-language modeling and pre\-training on large\-scale datasets. Specifically, we propose a new Re\-parameterizable Vision\-Language Path Aggregation Network \(RepVL\-PAN\) and region\-text contrastive loss to facilitate the interaction between visual and linguistic information. Our method excels in detecting a wide range of objects in a zero\-shot manner with high efficiency. On the challenging LVIS dataset, YOLO\-World achieves 35.4 AP with 52.0 FPS on V100, which outperforms many state\-of\-the\-art methods in terms of both accuracy and speed. Furthermore, the fine\-tuned YOLO\-World achieves remarkable performance on several downstream tasks, including object detection and open\-vocabulary instance segmentation.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2401.17270v3)

---


## YOLO\-CL: Galaxy cluster detection in the SDSS with deep machine learning / 

发布日期：2023-01-23

作者：Kirill Grishin

摘要：\(Abridged\) Galaxy clusters are a powerful probe of cosmological models. Next generation large\-scale optical and infrared surveys will reach unprecedented depths over large areas and require highly complete and pure cluster catalogs, with a well defined selection function. We have developed a new cluster detection algorithm YOLO\-CL, which is a modified version of the state\-of\-the\-art object detection deep convolutional network YOLO, optimized for the detection of galaxy clusters. We trained YOLO\-CL on color images of the redMaPPer cluster detections in the SDSS. We find that YOLO\-CL detects $95\-98%$ of the redMaPPer clusters, with a purity of $95\-98%$ calculated by applying the network to SDSS blank fields. When compared to the MCXC2021 X\-ray catalog in the SDSS footprint,YOLO\-CL is more complete then redMaPPer, which means that the neural network improved the cluster detection efficiency of its training sample. The YOLO\-CL selection function is approximately constant with redshift, with respect to the MCXC2021 cluster mean X\-ray surface brightness. YOLO\-CL shows high performance when compared to traditional detection algorithms applied to SDSS. Deep learning networks benefit from a strong advantage over traditional galaxy cluster detection techniques because they do not need galaxy photometric and photometric redshift catalogs. This eliminates systematic uncertainties that can be introduced during source detection, and photometry and photometric redshift measurements. Our results show that YOLO\-CL is an efficient alternative to traditional cluster detection methods. In general, this work shows that it is worth exploring the performance of deep convolution networks for future cosmological cluster surveys, such as the Rubin/LSST, Euclid or the Roman Space Telescope surveys.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2301.09657v2)

---


## MS\-YOLO: Infrared Object Detection for Edge Deployment via MobileNetV4 and SlideLoss / 

发布日期：2025-09-25

作者：Jiali Zhang

摘要：Infrared imaging has emerged as a robust solution for urban object detection under low\-light and adverse weather conditions, offering significant advantages over traditional visible\-light cameras. However, challenges such as class imbalance, thermal noise, and computational constraints can significantly hinder model performance in practical settings. To address these issues, we evaluate multiple YOLO variants on the FLIR ADAS V2 dataset, ultimately selecting YOLOv8 as our baseline due to its balanced accuracy and efficiency. Building on this foundation, we present texttt\{MS\-YOLO\} \(textbf\{M\}obileNetv4 and textbf\{S\}lideLoss based on YOLO\), which replaces YOLOv8's CSPDarknet backbone with the more efficient MobileNetV4, reducing computational overhead by textbf\{1.5%\} while sustaining high accuracy. In addition, we introduce emph\{SlideLoss\}, a novel loss function that dynamically emphasizes under\-represented and occluded samples, boosting precision without sacrificing recall. Experiments on the FLIR ADAS V2 benchmark show that texttt\{MS\-YOLO\} attains competitive mAP and superior precision while operating at only textbf\{6.7 GFLOPs\}. These results demonstrate that texttt\{MS\-YOLO\} effectively addresses the dual challenge of maintaining high detection quality while minimizing computational costs, making it well\-suited for real\-time edge deployment in urban environments.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2509.21696v1)

---


## Poly\-YOLO: higher speed, more precise detection and instance segmentation for YOLOv3 / 

发布日期：2020-05-27

作者：Petr Hurtik

摘要：We present a new version of YOLO with better performance and extended with instance segmentation called Poly\-YOLO. Poly\-YOLO builds on the original ideas of YOLOv3 and removes two of its weaknesses: a large amount of rewritten labels and inefficient distribution of anchors. Poly\-YOLO reduces the issues by aggregating features from a light SE\-Darknet\-53 backbone with a hypercolumn technique, using stairstep upsampling, and produces a single scale output with high resolution. In comparison with YOLOv3, Poly\-YOLO has only 60% of its trainable parameters but improves mAP by a relative 40%. We also present Poly\-YOLO lite with fewer parameters and a lower output resolution. It has the same precision as YOLOv3, but it is three times smaller and twice as fast, thus suitable for embedded devices. Finally, Poly\-YOLO performs instance segmentation using bounding polygons. The network is trained to detect size\-independent polygons defined on a polar grid. Vertices of each polygon are being predicted with their confidence, and therefore Poly\-YOLO produces polygons with a varying number of vertices.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2005.13243v2)

---

