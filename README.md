<div align="center">

# YOLO ArXiv Daily

[![Daily Papers](https://img.shields.io/badge/📅-每日更新-blue)]()
[![arXiv](https://img.shields.io/badge/arXiv-最新论文-red)](https://arxiv.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)

*自动追踪 YOLO 相关最新论文，提供中英文双语摘要*

</div>

---

## 📑 论文列表

> ### 1. Explainable and Resource\-Efficient Spatial Reasoning in Multimodal LLMs for Decision\-Critical Applications
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-07-29 |
> | 👤 作者 | Piyush Jain |
>
> **📄 英文摘要：**
> As Multimodal Large Language Models \(MLLMs\) are increasingly deployed in decision\-critical pipelines such as robotics, embodied AI, and safety monitoring, the opacity of their spatial judgments limits operator trust and auditability. MLLMs demonstrate strong reasoning but often struggle with fine\-grained spatial understanding and object hallucination. Prior work, ByDeWay, introduced Layered\-Depth\-Based Prompting \(LDP\), a training\-free framework that mitigates hallucinations by structuring prompts using monocular depth estimation. However, coarse depth layering falls short in resolving object\-to\-object spatial relationships within the same geometric plane, such as projective \("left of", "above"\) and topological \("inside", "touching"\) relations. We propose ByDeWay\-V2, which integrates explicit spatial relational context alongside depth cues, expressed as human\-readable predicates that serve as auditable evidence for downstream decision support. Using an open\-vocabulary object detector \(YOLO\-World\-L\), our framework computes pairwise geometric relations between detected objects and injects them as structured spatial predicates into the MLLM prompt, bridging 3D scene depth and 2D spatial semantics without any training. We evaluate ByDeWay\-V2 on the Visual Spatial Reasoning \(VSR\) and BLINK benchmarks across multiple MLLMs, with hallucination grounding assessed via POPE. On the BLINK spatial subset, ByDeWay\-V2 achieves a 46 percent relative F1 improvement over LDP for Qwen2.5\-VL, and recovers BLIP\-Base's spatial reasoning on VSR from near\-random performance to a competitive F1 of 0.53. Our lightest configuration operates under a strict 40\-token context budget on CPU, showing the framework's suitability for resource\-constrained, real\-time decision\-support settings.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2607.27145v1)

---

> ### 2. From Keypoints to Predictive Distributions: Post\-Hoc Uncertainty for YOLO\-Pose Models
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-07-29 |
> | 👤 作者 | Alexej Klushyn |
>
> **📄 英文摘要：**
> YOLO\-Pose models provide efficient keypoint localization, but do not quantify the associated spatial uncertainty. We introduce a lightweight post\-hoc probabilistic extension that augments a trained YOLO\-Pose model with calibrated bivariate predictive distributions over keypoint locations, centered at the model's original predictions. Concretely, we train additional probabilistic heads with an importance\-weighted negative log\-likelihood to predict an input\-dependent $2times2$ dispersion matrix for each keypoint, followed by Gaussian calibration for broad downstream compatibility or Student\-$t$ calibration for distributional fidelity. Complementing this, we propose an evaluation protocol that combines a suite of distributional calibration diagnostics with average keypoint precision \(AKP\), a keypoint\-level extension of the COCO AP protocol for assessing reliability rankings. Experiments on COCO show that the learned uncertainty estimates enable effective keypoint\-level reliability ranking, Student\-$t$ calibration best captures the empirical residual distribution, and uncertainty\-based pruning removes unreliable keypoints. A central application\-level demonstration is vision\-based aircraft landing, where calibrated covariances for runway keypoints support uncertainty\-aware aircraft position estimation and downstream sensor fusion.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2607.26921v1)

---

> ### 3. Reading Legends on Ancient Coins: An Object Detection Approach for Character Recognition on a Novel Roman Republican Dataset
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

> ### 4. Construction of entropy satisfying Active Flux\-type methods
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

> ### 5. Small\-Pollinator Detection in Cluttered Field Video
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

<div align="center">

*由 [YOLO-Arxiv-Daily](https://github.com/WangQvQ/YOLO-Arxiv-Daily) 自动生成*

</div>