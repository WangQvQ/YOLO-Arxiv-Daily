<div align="center">

# YOLO ArXiv Daily

[![Daily Papers](https://img.shields.io/badge/📅-每日更新-blue)]()
[![arXiv](https://img.shields.io/badge/arXiv-最新论文-red)](https://arxiv.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)

*自动追踪 YOLO 相关最新论文，提供中英文双语摘要*

</div>

---

## 📑 论文列表

> ### 1. Post\-Training Semantic Lifting for 3D Gaussian Splatting: Separating Detector, Lifting and Representation Error
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

> ### 2. Towards benchmarking Western Bluebird detection in the wild
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-10-06 |
> | 👤 作者 | Estela Monserrat Arriaga Santana |
>
> **📄 英文摘要：**
> Bird monitoring in natural environments is challenging due to the small size of some species of birds relative to the scene, background clutter, variability in illumination, and the observers' viewpoint. Progress is further limited by the scarcity of large\-scale, realistic datasets, which are essential for understanding behavioral patterns. To address this gap, we introduce a new benchmark dataset for the detection and segmentation of Western bluebirds \(Sialia Mexicana\), comprising over 6,000 labeled images from 41 recording sessions. The dataset features high\-resolution \(4K\) in\-the\-wild images in which birds occupy only a small fraction of the image. We evaluated supervised detectors, open\-vocabulary models under zero\-shot and fine\-tuned settings, and segmentation approaches. Supervised detectors remain the most reliable overall, with Faster R\-CNN achieving the highest detection mAP and RT\-DETR offering the best precision\-recall trade\-off. Open\-vocabulary models perform poorly in zero\-shot settings; however, fine\-tuning substantially improves their performance, with YOLO\-World becoming competitive with supervised methods and achieving the highest precision, F1\-score, and mAP@0.5. For segmentation, supervised methods significantly outperform Grounded\-SAM and SAM 3: Mask R\-CNN achieves the highest mask mAP, while YOLOv8\-Seg provides the best precision and fastest inference. A diagnostic analysis further shows that failures are not explained by object size alone, but by a combination of apparent scale, brightness, contrast, clutter, blur, crowding, and recording\-session variation. Overall, our findings highlight the difficulty of zero\-shot bird detection in cluttered ecological scenes and underscore the importance of domain adaptation in small\-object settings.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2610.07802v1)

---

> ### 3. AUTOPILOT An Advanced Perception, Localization and Path Planning Techniques for Autonomous Vehicles Using YOLOv7 and MiDaS
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-10-05 |
> | 👤 作者 | Harshkumar Devmurari |
>
> **📄 英文摘要：**
> Self driving vehicles have emerged as a reliable technology that has the capability to transform transportation and mobility. The development of self driving cars requires significant advances in a number of areas, including perception, localization, decision making, and control. This research paper is based on the project implementation of the combination of object detection using YOLO \(You Only Look Once\), depth sensing using MiDaS for the localization and perception of obstacles, perspective transform, and decision making for path planning in self driving cars. The contemporary state of the technology for object detection, depth sensing, localization, and path planning evaluates the performance of the combined system through simulations and experiments. The results show that the combination of YOLO and MiDaS provides a new robust system for object detection and depth sensing. This research paper contributes to the advancement of self driving car technology and provides new and innovative approaches to the perception and localization of obstacles in the environment. Keywords: YOLO, MiDaS, perception, localization, decision making
>
> 🔗 [阅读论文](http://arxiv.org/abs/2610.06232v1)

---

> ### 4. Vision\-enabled detection of safety helmet compliance in construction zones
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-10-05 |
> | 👤 作者 | Tri Nhut Do\* |
>
> **📄 英文摘要：**
> In the rapidly evolving field of construction management, worker safety remains a top priority. This paper introduces an innovative vision\-based system for real\-time detection of helmet compliance, specifically designed for construction sites, utilizing advanced computer vision techniques and machine learning algorithms within the YOLO \(you only look once\) framework. Our system leverages high\-resolution video feeds from strategically positioned cameras to monitor adherence to safety regulations regarding helmet usage. By employing deep learning methodologies, the system effectively identifies individuals not wearing helmets, thereby significantly mitigating the risk of head injuries among workers. Our training and validation results revealed an impressive precision exceeding 97% at mAP@0.5 for both helmeted and non\-helmeted individuals. Furthermore, our experiments demonstrate exceptional detection accuracy, demonstrating the system's resilience under varying lighting conditions and diverse worker movements. The consistent decrease in loss and improvement in metrics throughout training validates the effectiveness of the YOLOv8 model in enhancing recognition performance. The implications of this research extend beyond mere regulatory compliance, opening avenues for innovative applications in occupational safety management. This study highlights the critical role of technology in protecting lives and lays the groundwork for future advancements in smart construction environments.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2610.05756v1)

---

> ### 5. BeeWhere: Segmenting Bumble Bee Colonies to Quantify Behavioral Effects
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-10-02 |
> | 👤 作者 | Roberta Hunt |
>
> **📄 英文摘要：**
> Social bees are important pollinators that support biodiversity and crop pollination globally and serve as important model systems for collective behavior, but scalable measurement of individual\- and colony\-level behavior remains difficult in dense, occluded nest environments. Existing monitoring workflows use fiducial tags \(e.g., ArUco\) to preserve individual identity, yet tag\-based tracking can fail when markers are obscured and provide limited information about body extent, spatial context, and untagged individuals. We present BeeWhere, an AI\-assisted annotation and analysis workflow that combines ArUco detections with deep\-learnt instance segmentations to quantify bumble bee behavior from high\-resolution colony images and videos. Using bumble bee \(Bombus impatiens\) microcolonies as a test case, we annotate 483 frames containing 8,443 bee instances. We additionally annotate pollen balls, nest structures, and chamber boundaries, and train YOLO instance segmentation models for downstream behavioral analysis. Instance segmentations enable quantification of important behavioral metrics based on body contours, including nearest\-neighbor distance, proximity to nest structures, spatial occupancy within the nest, and detection counts over time. We apply the BeeWhere models to tag\-based tracking in an exploratory validation study assessing the behavioral impacts of neonicotinoid pesticide exposure. BeeWhere increased detection rates compared to tag\-based tracking, particularly when bees were partially obscured or under challenging imaging conditions, and also captured treatment\-associated changes in bee spatial organization not captured using tag\-based tracking alone. These results suggest that instance segmentation can complement fiducial\-marker tracking by recovering behaviorally meaningful signals under challenging colony conditions.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2610.03051v1)

---

<div align="center">

*由 [YOLO-Arxiv-Daily](https://github.com/WangQvQ/YOLO-Arxiv-Daily) 自动生成*

</div>