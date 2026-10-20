<div align="center">

# YOLO ArXiv Daily

[![Daily Papers](https://img.shields.io/badge/📅-每日更新-blue)]()
[![arXiv](https://img.shields.io/badge/arXiv-最新论文-red)](https://arxiv.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)

*自动追踪 YOLO 相关最新论文，提供中英文双语摘要*

</div>

---

## 📑 论文列表

> ### 1. Precision Cotton Disease Management: Leveraging UAV Imagery and Custom YOLO Deep Learning Framework for Enhanced Detection
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-10-16 |
> | 👤 作者 | 1. Anwar Iqbal\*1 aiqbal. rime23smme@student. nust. edu. pk 2. Kunwar Faraz Ahmad Khan1 kunwar. faraz@smme. nust. edu. pk 3. Shahbaz Khan1 shahbaz. khan@smme. nust. edu. pk 4. Khawaja Fahad Iqbal1 fahad. iqbal@smme. nust. edu. pk 5. Sara Ali1 |
>
> **📄 英文摘要：**
> Cotton sustains the textile economies of over 80 countries with an annual yield of 113 million bales, yet 30\-60% of yield losses are still attributed to viral, bacterial, and fungal diseases that remain undetected until manual field scouting. To address this critical issue, the integration of Unmanned Aerial Vehicle \(UAV\) imagery with deep learning offers a scalable early\-warning system; however, existing models suffer from low early\-stage accuracy, poor transferability, and high computational cost. In this paper, we propose Cotton\-Vision, a fully convolutional single\-stage detector that synergizes three lightweight architectural innovations: Ghost Bottleneck modules for a 20% parameter reduction, Efficient Channel Attention \(ECA\) for adaptive feature recalibration, and Cross\-Stage Partial \(CSP\) fusion for enriched gradient flow. The network is trained on a novel dataset collected across seven cotton fields in the Punjab province of Pakistan, comprising over 2,700 high\-resolution drone images. Extensive data augmentation \(rotation, shear, exposure, and saturation\) expands the dataset to 11,704 training samples. On a held\-out test set, Cotton\-Vision achieves a precision of 97.7%, a recall of 77.7%, an mAP@50 of 85.6%, and an mAP@50\-95 of 76.6%. These results represent a 10\-20% improvement in precision, recall, and mAP over state\-of\-the\-art baselines, including YOLOv5, YOLOv8, YOLOv11, YOLOv12, RT\-DETR, and RF\-DETR. Through a comprehensive ablation study, we demonstrate that each proposed architectural enhancement contributes to the model's superior performance, yielding a 12\-16% mAP improvement over the baseline architecture. Our findings confirm that this synergy of lightweight and attention\-based modules provides a robust, computationally viable solution for early cotton disease detection.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2610.24182v1)

---

> ### 2. Replay Pruning and Few\-Bit Integer Inference for Hardware\-Efficient Speed\-Sign Recognition
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-10-16 |
> | 👤 作者 | Se Un Park |
>
> **📄 英文摘要：**
> Speed\-limit sign recognition on an automotive camera system\-on\-chip \(SoC\) must fit a detector and a classifier \(the reader\) into the memory, arithmetic, and energy budgets of the device. In our two\-stage recognizer, the detector, a YOLO\-family network with ReLU and hard\-sigmoid operators, is compressed by channel replay: an iterative pruning cascade identifies the channels to remove, the recorded deletions are replayed on the original task\-trained network, the parent, and one continuous knowledge\-distillation recovery follows. On a YOLOv5n, replay removes 60.3% of the parameters at the accuracy of the parent and, in validation mAP50\-95 over five matched seeds, outperforms iterative pruning and training from scratch, and its per\-step variant \(per\-step replay\) is the most accurate pruning method. On a YOLOv8n, replay and per\-step replay outperform every other method and the unpruned parent. The reader, a five\-layer CNN with 18k parameters, and the post\-processing of the YOLOv5n and YOLOv8n heads use integer arithmetic only. With 4\-bit weights and 8\-bit activations \(W4A8\), the reader matches FP32 accuracy in 12.1 KB of on\-chip memory, and an executable streaming model reproduces the INT8 class predictions of its runtime on all 9,850 test crops. On a Jetson AGX Orin, INT8 is faster than FP16 only in the implicit formulation of TensorRT, and the replay detector runs fastest on the deep\-learning accelerator \(DLA\). Trained on public data only, the combined system of 0.72 M parameters recognizes 95.2% of the signs of a held\-out automotive camera at 90% precision and 94 frames per second.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2610.22811v1)

---

> ### 3. napari\-TurboBox: Interactive Multi\-Viewer Bounding Box Annotation for AI\-Guided Volumetric Bioimaging
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-10-14 |
> | 👤 作者 | Nils Friederich |
>
> **📄 英文摘要：**
> Promptable segmentation models turn one 3D bounding box into an instance mask. Box placement therefore limits how fast microscopy volumes with thousands of objects are annotated. Each box must be checked in several synchronized viewers, but napari\-bbox shares no state between viewers. An edit must update every viewer and slows down with more boxes. napari\-TurboBox instead holds one shared copy of the boxes for all viewers. In a four\-viewer layout, napari\-TurboBox needs a few milliseconds per mouse move to update the three slice viewers up to $10^4$ boxes. The same update in napari\-bbox leaves the 33 ms budget beyond about $10^3$ boxes and takes 159 ms at $10^4$. Boxes stay axis\-aligned and export to COCO and YOLO as per\-slice 2D boxes.   napari\-TurboBox is a pure\-Python plugin for napari 0.6.6 to 0.9. The code is available at https://github.com/N\-Friederich/napari\-turbobox under the Apache 2.0 licence, and version 0.1.0 is archived at https://doi.org/10.5281/zenodo.23090828 on Zenodo.
>
> **💻 代码链接：** https://github.com/N-Friederich/napari-turbobox，https://doi.org/10.5281/zenodo.23090828
>
> 🔗 [阅读论文](http://arxiv.org/abs/2610.22413v1)

---

> ### 4. Identifying Planetary Nebulae Candidates from IPHAS and VPHAS\+ Surveys via YOLO Deep Learning Models
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

> ### 5. Augmenting Real\-time Visualization in Minimally Invasive Surgery with Particle Filtering and Depth\-Aware Occlusion Handling
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

<div align="center">

*由 [YOLO-Arxiv-Daily](https://github.com/WangQvQ/YOLO-Arxiv-Daily) 自动生成*

</div>