<div align="center">

# YOLO ArXiv Daily

[![Daily Papers](https://img.shields.io/badge/📅-每日更新-blue)]()
[![arXiv](https://img.shields.io/badge/arXiv-最新论文-red)](https://arxiv.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)

*自动追踪 YOLO 相关最新论文，提供中英文双语摘要*

</div>

---

## 📑 论文列表

> ### 1. Reading Legends on Ancient Coins: An Object Detection Approach for Character Recognition on a Novel Roman Republican Dataset
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-07-28 |
> | 👤 作者 | Hafeez Anwar |
>
> **📄 英文摘要：**
> When it comes to the proper classification of ancient coins with respect to their time and issuer, the textual inscriptions on these coins, also known as legends, are of paramount importance. These legends consist of alphabets or characters still used in English. This paper addresses image based character recognition on ancient Roman Republican coins via a deep learning based object detection strategy. However, legends on these coins pose high variation due to non\-uniform placement, primitive inscription techniques, and wear and tear. Additional challenges include inconsistent imaging conditions such as illumination, orientation, and scale. To accommodate these, we gathered a novel large\-scale dataset of 5,654 Roman Republican coin images, manually annotated with 21 character labels, totaling 38,808 annotations. For recognition, we use You Only Look Once \(YOLO\) variants: YOLOv3, v4, v5, v7, and v8. YOLOv7\-Large achieves the best mAP50 of 90.4%, followed by YOLOv7\-Extended and YOLOv7\-xl with 90.2% and 90.1%, respectively.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2607.25455v1)

---

> ### 2. Construction of entropy satisfying Active Flux\-type methods
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-07-27 |
> | 👤 作者 | Remi Abgrall |
>
> **📄 英文摘要：**
> This paper is devoted to the analysis of the entropy stability properties of Active Fluxyolo\{\-type\} scheme for a hyperbolic system equipped with one entropy inequality. This type of scheme evolves two sets of degrees of freedom: point values that are chosen on the boundary of the elements that cover the computational domain, and the average of the solution in these elements. We show that the only thing to do is to get an entropy inequality for the average values, the point values degrees of freedom do not play any role. We construct a monolithic scheme which is bound preserving of cite\{BP\_Pampa\_VEM\}, non oscillatory following cite\{PampaDG\}, and entropy diminishing. The entropy condition is implemented in Tadmor's frameworkcite\{TadmorEntropy\}, i.e. for the semi\-discrete scheme only. The scheme is tested on the Kurganov\-Popov\-Petrova test case cite\{KPP\} which is known to yolo\{be\} sensitive to the satisfaction of an entropy inequality. We show that our entropy correction is effective: if we do not activate the bound\-preserving nor the non oscillatory condition, we get the correct solution with some spurious wiggles, as expected. Though the development, implementation and tests are done with the triangle version of the scheme, the same method can be used for polygonal meshes, following cite\{BP\_Pampa\_VEM\}.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2607.25111v1)

---

> ### 3. Small\-Pollinator Detection in Cluttered Field Video
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-07-24 |
> | 👤 作者 | Onur Onal |
>
> **📄 英文摘要：**
> Detecting pollinators in field video is challenging: targets are small, visually similar, and observed against cluttered vegetation under blur and occlusion. We present a systematic empirical study of small\-pollinator detection under a practical single\-GPU compute budget. Using the BuzzSpot challenge dataset, we compare YOLO and RF\-DETR models across input resolutions and evaluate sliced inference, class\-gated fusion, size\-routed ensembling, and post\-hoc temporal processing. RF\-DETR Large at 1344\-pixel resolution achieved our best hidden\-test result, reaching 0.405 mAP50:95 and outperforming the 1120\-pixel model \(0.379\) and the best single\-model YOLO26m baseline \(0.366\). The strongest gains came from adopting RF\-DETR and increasing its input resolution, indicating that detector choice and input resolution were more effective levers than added inference\-time complexity; the resolution gain was strongest for small objects and the rarer bumblebee and moth classes. Sliced\-inference fusion, size\-routed ensembling, and warm\-started 1536\-pixel continuation did not surpass this result, while post\-hoc temporal processing did not improve the leaked diagnostic evaluation. Error analysis identified bee\-hoverfly discrimination as the clearest remaining bottleneck: neighboring frames rarely supplied correctly classified hoverfly evidence for post\-hoc correction. These findings motivate learned feature\-level temporal aggregation before the final classification decision.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2607.22913v1)

---

> ### 4. Synthetic data generation framework for quality control automation in gravure printing
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-07-23 |
> | 👤 作者 | Korota Arsène Coulibaly |
>
> **📄 英文摘要：**
> Quality control in printing, particularly in rotogravure printing, still depends on slow, costly, and subjective manual inspection. Automated surface defect detection is critical for maintaining high\-quality standards in rotogravure printing. Deep learning models give prospects for automation. However, training robust deep learning models, such as YOLO or Vision Transformers, is heavily hindered by the extreme scarcity of real\-world industrial defects images. To overcome this limitation, this paper introduces a novel synthetic data generation framework tailored for rotogravure printing quality control. The proposed pipeline automatically generates high\-fidelity images of specific printing defects \(creases, streaks, misregistration, etc.\) and outputs corresponding bounding boxes and annotations. To validate the framework, a synthetic dataset of 7533 images was generated and used to train the state\-of\-the\-art object\-detection model RFDETR. Experimental results demonstrate that the model trained on our synthetic data achieves a Mean Average Precision \(mAP\) of 80.9% on real industrial testing samples. This framework provides a zero\-cost, rapid\-deployment solution for automating defect inspection in printing lines without requiring massive manual data collection.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2607.21577v1)

---

> ### 5. Real\-Time EEG Cap Electrode Detection for Guided Point\-of\-Care Placement
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-07-22 |
> | 👤 作者 | William Lehn\-Schiøler |
>
> **📄 英文摘要：**
> We present a two\-stage vision system that detects EEG cap electrodes in a live webcam stream and validates their anatomical placement in real time. A single\-class YOLO detector localises electrodes; a geometric stage assigns each detection to a named 10\-20 role from facial landmarks. Evaluating under subject\-disjoint leave\-one\-subject\-out \(LOSO\) cross\-validation across five subjects wearing the clinically\-validated Small/Medium/Large caps, the detector attains mAP@.5 = 0.94 \+/\- 0.07 across five held\-out folds \(0.96 pooled\). A dedicated leave\-one\-cap\-out axis, holding out every frame of a cap regardless of subject, leaves Medium and Large mAP@.5 within 0.01 of LOSO \(0.97, 0.97\) while Small drops to 0.72 \+/\- 0.28, a gap confounded with subject familiarity rather than cap style. Geometric augmentation \(rotation, perspective, mixup\) improves in\-plane\-roll robustness and temporal\-electrode recall at no inference cost, and a landmark\-driven head crop extends the usable distance range, lifting mAP@.5 from 0.23 to 0.45 at 0.6 x apparent scale. A compact mobile\-candidate backbone \(YOLOv10n\) keeps the detector at real\-time throughput \(19 FPS\) on a commodity CPU at 640 px.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2607.20142v1)

---

<div align="center">

*由 [YOLO-Arxiv-Daily](https://github.com/WangQvQ/YOLO-Arxiv-Daily) 自动生成*

</div>