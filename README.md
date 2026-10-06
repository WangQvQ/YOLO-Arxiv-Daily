<div align="center">

# YOLO ArXiv Daily

[![Daily Papers](https://img.shields.io/badge/📅-每日更新-blue)]()
[![arXiv](https://img.shields.io/badge/arXiv-最新论文-red)](https://arxiv.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)

*自动追踪 YOLO 相关最新论文，提供中英文双语摘要*

</div>

---

## 📑 论文列表

> ### 1. Vision\-enabled detection of safety helmet compliance in construction zones
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

> ### 2. BeeWhere: Segmenting Bumble Bee Colonies to Quantify Behavioral Effects
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

> ### 3. HandAnthro: Automated Hand Anthropometry from a Single Image
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-09-29 |
> | 👤 作者 | Fan Zhou |
>
> **📄 英文摘要：**
> Hand anthropometry supports protective\-glove design, but existing measurement methods often require trained operators, specialized hardware, or manual landmarking. We present HandAnthro, which estimates 44 projected hand dimensions from a smartphone photograph of a palm\-up hand on US letter\-size paper. The pipeline reconstructs wrist\-occluded paper boundaries for rectification, whitens non\-hand pixels, and refines 41 anthropometry\-specific landmarks from a fine\-tuned You Only Look Once \(YOLO\) pose model using image\-specific geometry and contours. Controlled evaluation comprised 720 captures from 45 held\-out participants, each contributing 16 images across two smartphones, two backgrounds, two angles, and two nominal illumination settings. HandAnthro produced complete outputs for 704 captures \(97.8%\); among these, mean absolute error \(MAE\) was 3.80 mm per dimension against two trained operators' caliper measurements. Regional MAEs were 2.48 mm for non\-thumb fingers, 6.04 mm for thumbs, and 6.17 mm for palm and wrist. In a researcher\-assisted mobile\-app pilot, automated batch processing returned all 44 dimensions for 260 of 268 retained, researcher\-screened firefighter images \(97.0%\). A descriptive, unpaired comparison with an independent national firefighter reference yielded a mean absolute difference of 2.40 mm across 28 sex\-by\-dimension group\-mean contrasts. These results characterize controlled measurement performance and researcher\-assisted field feasibility for future distributed hand\-anthropometry studies.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2609.37855v1)

---

> ### 4. ByteTraX: Enhancing the ByteTrack Architecture with Optimised Thresholding
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-09-29 |
> | 👤 作者 | Thomas A. O'Shea\-Wheller |
>
> **📄 英文摘要：**
> The ByteTrack algorithm is a widely used and computationally efficient multi\-object tracking architecture. Its core innovation lies in the combination of lenient bounding box associations with tracklet similarity matching to robustly deal with object occlusions. However, this strategy is nevertheless vulnerable to erroneous track reclassification and identity switching, as detection confidence scores dictate association priority. To address this, I present a simple enhancement of the ByteTrack architecture\-\-named ByteTraX\-\-that optimises track continuity via a single unified matching threshold, while penalising identity switches through stringent track initiation criteria. This approach achieves consistently improved performance across a range of diverse benchmarks including GMOT\-40, LC\-MOT, SportsMOT, TeamTrack, DAMUNT, and DeepSea\-MOT, while simultaneously increasing processing speed by >10%. Specifically, results demonstrate a >40% reduction in identity switches, accompanied by mean increases in HOTA of 3.6, IDF1 of 5.6, and FPS of 6.3. As such, adoption of the ByteTraX algorithm has the potential to substantially enhance tracking performance over the ByteTrack baseline, while retaining the efficiency needed for real\-time deployment. To facilitate usage, I provide the source code, integration functionality for the YOLO family of object detection models, and deployment instructions via an open source repository.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2609.37801v2)

---

> ### 5. A Multi\-Dataset Benchmark of YOLO\-Based Weed Detection in Precision Agriculture
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-09-27 |
> | 👤 作者 | Hristina Zdraveska |
>
> **📄 英文摘要：**
> Weed detection is an important component of precision agriculture, enabling site\-specific weed management and reducing unnecessary herbicide use. Although deep learning methods have achieved strong results for crop and weed detection, many studies rely on single\-dataset evaluation, making it difficult to assess robustness across different agricultural domains. This paper presents a multi\-dataset benchmark of deep object detectors for weed detection in precision agriculture, with a focused evaluation of YOLO26 models. We evaluate nano, small, and medium variants on seven public weed\-detection datasets covering different crops, weed species, field conditions, acquisition setups, and annotation protocols. The models are compared in terms of detection accuracy, model complexity, inference latency, FPS, and model size. In addition to in\-dataset evaluation, we investigate cross\-domain generalization using a unified one\-class weed setup and evaluate multi\-source training using the combined training subsets from all datasets. The results show that YOLO26 achieves strong in\-dataset performance, with YOLO26m obtaining the highest average accuracy and YOLO26s providing the best practical accuracy\-efficiency trade\-off. However, cross\-domain performance decreases substantially, with YOLO26s dropping from an average in\-domain mAP$\_\{50:95\}$ of 0.603 to 0.148 in the off\-domain setting. Multi\-source training improves performance on several datasets, but does not fully eliminate domain shift. Overall, the benchmark highlights the importance of dataset diversity, domain similarity, and target\-domain adaptation for robust weed detection in real\-world precision agriculture applications.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2609.33991v1)

---

<div align="center">

*由 [YOLO-Arxiv-Daily](https://github.com/WangQvQ/YOLO-Arxiv-Daily) 自动生成*

</div>