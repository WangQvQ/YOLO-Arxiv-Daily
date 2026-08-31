<div align="center">

# YOLO ArXiv Daily

[![Daily Papers](https://img.shields.io/badge/📅-每日更新-blue)]()
[![arXiv](https://img.shields.io/badge/arXiv-最新论文-red)](https://arxiv.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)

*自动追踪 YOLO 相关最新论文，提供中英文双语摘要*

</div>

---

## 📑 论文列表

> ### 1. CF\-YOLO: Context\-Aware Feature Refinement for Camouflaged Industrial Micro\-Defect Detection
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-08-28 |
> | 👤 作者 | Xinda Yu |
>
> **📄 英文摘要：**
> Automated detection of surface micro\-defects on industrial components, such as copper tubes, is critically important for quality assurance but remains challenging due to the minute scale of anomalies and their visual camouflage against complex backgrounds. These factors lead to weak feature representations and high rates of false positives and missed detections. To address these issues, we propose a novel real\-time detection framework designed for efficient context perception and feature refinement. Our method integrates a Context\-Perception Aggregation Module \(CPAM\), which synergises large\-kernel perception for macro\-texture context and small\-kernel aggregation for sharp boundary delineation, effectively breaking the background camouflage. Furthermore, a Feature Additive Refinement Module \(FARM\) employs a linear\-complexity additive token mixer to globally verify and refine the representation of fine\-grained anomalies, suppressing noise\-induced errors. To support research in this domain, we introduce the Copper Tube Defect Dataset \(CTDD\), a manually annotated benchmark containing 1,847 images and 4,898 boundingbox defect instances from copper\-tube inspection scenarios. Extensive experiments demonstrate that our detector achieves strong and consistent performance on CTDD, outperforming representative baseline detectors, including YOLOv11, by 2.2% in mAP@50 and 3.9% in Precision while maintaining real\-time inference speed. This work provides a robust and efficient solution for high\-precision industrial inspection, bridging the gap between contextual understanding and detailed feature analysis. Our code and model are available at: https://github.com/Yu\-Xinda/CFYOLO\-Context\-Aware\-Feature\-Refinement\-for\-Camouflaged\-Industrial\-Micro\-Defect\-Detection
>
> **💻 代码链接：** https://github.com/Yu-Xinda/CFYOLO-Context-Aware-Feature-Refinement-for-Camouflaged-Industrial-Micro-Defect-Detection
>
> 🔗 [阅读论文](http://arxiv.org/abs/2608.28070v1)

---

> ### 2. Depth\-Aware Pothole Detection Using YOLO and RT\-DETR at the Edge
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-08-27 |
> | 👤 作者 | Md Monjurul Ahsan Prodhan |
>
> **📄 英文摘要：**
> Pothole detection and its severity measurement is still an important challenges in urban infrastructure management, where late maintenance directly contributes to vehicle damage, road accidents, and escalating repair costs. Existing automated approaches depend on 2D RGB images and cannot measure physical depth of potholes. In this paper, we present a depthaware pothole detection framework and then compare five architectures: YOLOv8n, YOLOv8nSeg, YOLOv9t, RTDETRL, and RTDETRX for RGB\-D sensor fusion\-based detection and automated depth measurement. A custom offline augmentation pipeline is used here to simulate adverse road monitoring conditions. All models are trained on the PothRGBD dataset with an 80% training and 20% validation split and evaluated using Precision, Recall, mAP@50, and mAP@50\_95. Before measuring the depth data, all depth maps are corrected for camera tilt using RANSAC ground\-plane orthorectification and all zero\-valued sensor pixels are cast to NaN before any statistic is computed. YOLOv8nSeg achieves the highest mAP@50 of 0.9556 and mAP@50\_95 of 0.6758 with the most accurate depth estimate of 2.96 cm with the pixel\-precise Dseg algorithm. YOLOv8n achieves the fastest inference at 3.6ms. RTDETRX achieves the highest detection confidence at 92.70%. An important finding is that even after full RANSAC orthorectification, bounding box models overestimate pothole depth by 0.16 to 0.21 cm compared to pixel precise segmentation masks. This confirms that the pavement inclusion bias is structural rather than a calibration artifact.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2608.27633v1)

---

> ### 3. Lowering the Barrier to AI\-Driven Inspection: A No\-Code Workflow for Automated Structural Defect Detection
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-08-25 |
> | 👤 作者 | Michael Holm |
>
> **📄 英文摘要：**
> Structural health monitoring \(SHM\) is essential in modern engineering, providing data for condition\-based maintenance, lifecycle assessment, and predictive decision\-making. Traditionally, SHM relied on visual inspection to detect defects such as cracks and deformations. Early computer vision \(CV\) methods, including thresholding, edge detection, and handcrafted features, aimed to automate this process but were highly sensitive to noise, imaging variations, and multiscale defects, limiting their reliability.   Recent advances in machine learning, particularly convolutional neural networks \(CNNs\) and You Only Look Once \(YOLO\), have improved defect detection accuracy and enabled real\-time analysis. However, adoption in SHM remains limited due to technical barriers such as data labeling, model training, and deployment, which typically require programming expertise.   To address this gap, we introduce YOLOEZ, an open\-source, GUI\-based tool for end\-to\-end YOLO model application. YOLOEZ integrates data labeling, training, and inference into a single interface, enabling high\-performance model development without code while supporting reproducible workflows.   Evaluation against existing software and classical image processing demonstrates that YOLOEZ not only outperforms traditional methods across most detection metrics, but also lowers adoption barriers present in other modern CV tools. By combining accuracy with accessibility, YOLOEZ facilitates wider use of AI\-driven monitoring for predictive maintenance, digital twins, and intelligent structural systems.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2608.25176v1)

---

> ### 4. Decoupling candidate dual AGN from chance superpositions in the GOTHIC survey via a deep\-learning framework
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-08-25 |
> | 👤 作者 | Bhavesh Mukheja |
>
> **📄 英文摘要：**
> Dual active galactic nuclei \(DAGN\) mark a critical phase in the evolution of merging galaxies and the pairing of supermassive black holes, yet they remain difficult to identify in large imaging surveys because of projection effects and limited spatial resolution. Compact foreground stars and unresolved substructure can mimic dual nuclei through chance superposition, complicating automated detection. We revisit the 46,061 galaxies flagged but rejected as DAGN candidates by the GOTHIC pipeline, primarily because the two nuclei fell within the SDSS fibre aperture or exceeded its separation threshold. We train a supervised deep\-learning framework based on the YOLOv11 oriented\-bounding\-box architecture on annotated SDSS imaging to separate genuine dual nuclei from foreground stellar contaminants and other spurious alignments. The final model attains a validation precision of 0.919, recall of 0.905, and $F\_1$ of 0.912 for the dual\-nuclei class, and yields 29,605 dual\-nucleus candidates after removing star\-dominated and blended detections. Structured visual inspection indicates that $54.5$\-\-$62%$ are consistent with genuine dual nuclei, implying $sim\(1.4$\-\-$1.8\)times10^\{4\}$ plausible systems. Cross\-calibrating the YOLO separation against the deterministic GOTHIC centroid measurement and restricting to the compact regime \($d le 6.87''$\) gives a conservative subset of $sim 13\{,\}672$ candidates, reaching calibrated separations of $sim 0.56''$. Spectroscopy of the most compact \($le 1$~kpc\) systems shows they are dominated by passive, absorption\-line galaxies with no resolved double\-peaked emission, so confirmation requires higher\-resolution follow\-up. The catalogue is a statistically refined list of candidates, not confirmed DAGN. Nonetheless, deep\-learning detection substantially reduces contamination and expands the plausible DAGN census.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2608.24164v1)

---

> ### 5. Cross\-Generation Optimization of YOLOv26, YOLOv11, and YOLOv8 for Fine\-Grained Small\-Object Detection and Instance Segmentation in Complex Orchards
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-08-23 |
> | 👤 作者 | Ranjan Sapkota |
>
> **📄 英文摘要：**
> Small\-object detection and instance segmentation remain challenging in orchard environments because of green\-on\-green similarity, occlusion, and limited pixel representation of fine fruit anatomy. This study presents a cross\-generation benchmark of Ultralytics YOLOv8, YOLOv11, and YOLOv26 for detecting and segmenting apple fruitlet, calyx, and peduncle structures for robotic orchard perception. Five model scales \(n, s, m, l, and x\) were evaluated under conventional 640 x 640 and small\-object focused 960 x 960 training configurations, yielding 30 experiments. Increasing model capacity did not consistently improve accuracy. YOLOv11s\-960 achieved the highest observed mask mAP@50:95 \(0.402\) and box mAP@50:95 \(0.426\), while YOLOv26s\-960 achieved comparable values of 0.397 and 0.425 with only 10.37 M parameters and 34.1 GFLOPs. Peduncle remained the most challenging class. Overall, compact\-to\-moderate YOLO models with small\-object\-focused training provided favorable accuracy efficiency trade\-offs, establishing a practical benchmark for fine\-grained agricultural robotics and orchard perception. Github Link: https://github.com/rnjnspkt/Optimizing\-and\-Comparing\-Ultralytics\-YOLOv26\-YOLOv11\-and\-YOLOv8\-for\-Small\-Object\-Detection\-and\-Seg
>
> **💻 代码链接：** https://github.com/rnjnspkt/Optimizing-and-Comparing-Ultralytics-YOLOv26-YOLOv11-and-YOLOv8-for-Small-Object-Detection-and-Seg
>
> 🔗 [阅读论文](http://arxiv.org/abs/2608.23636v1)

---

<div align="center">

*由 [YOLO-Arxiv-Daily](https://github.com/WangQvQ/YOLO-Arxiv-Daily) 自动生成*

</div>