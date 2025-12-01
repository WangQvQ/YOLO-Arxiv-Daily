# 每日从arXiv中获取最新YOLO相关论文


## Identifying bars in galaxies using machine learning / 

发布日期：2025-11-28

作者：Rajit Shrivastava

摘要：This thesis presents an innovative framework for the automated detection and characterization of galactic bars, pivotal structures in spiral galaxies, using the YOLO\-OBB \(You Only Look Once with Oriented Bounding Boxes\) model. Traditional methods for identifying bars are often labor\-intensive and subjective, limiting their scalability for large astronomical surveys. To address this, a synthetic dataset of 1,000 barred spiral galaxy images was generated, incorporating realistic components such as disks, bars, bulges, spiral arms, stars, and observational noise, modeled through Gaussian, Ferrers, and Sersic functions. The YOLO\-OBB model, trained on this dataset for six epochs, achieved robust validation metrics, including a precision of 0.93745, recall of 0.85, and mean Average Precision \(mAP50\) of 0.94173. Applied to 10 real galaxy images, the model extracted physical parameters, such as bar lengths ranging from 2.27 to 9.70 kpc and orientations from 13.41$^circ$ to 134.11$^circ$, with detection confidences between 0.26 and 0.68. These measurements, validated through pixel\-to\-kiloparsec conversions, align with established bar sizes, demonstrating the model's reliability. The methodology's scalability and interpretability enable efficient analysis of complex galaxy morphologies, particularly for dwarf galaxies and varied orientations. Future research aims to expand the dataset to 5,000 galaxies and integrate the Tremaine\-Weinberg method to measure bar pattern speeds, enhancing insights into galaxy dynamics and evolution. This work advances automated morphological analysis, offering a transformative tool for large\-scale astronomical studies.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2511.23383v1)

---


## Hierarchical Feature Integration for Multi\-Signal Automatic Modulation Recognition / 

发布日期：2025-11-28

作者：Yunpeng Qu

摘要：Automatic modulation recognition \(AMR\) is a crucial step in wireless communication systems, which identifies the modulation scheme from detected signals to provide key information for further processing. However, previous work has mainly focused on the identification of a single signal, overlooking the phenomenon of multiple signal superposition in practical channels and the signal detection procedures that must be conducted beforehand. Considering the susceptibility of radio frequency \(RF\) signals to noise interference and significant spectral variations, we propose a novel Hierarchical Feature Integration \(HIFI\)\-YOLO framework for multi\-signal joint detection and modulation recognition. Our HIFI\-YOLO framework, with its unique design of hierarchical feature integration, effectively enhances the representation capability of features in different modules, thereby improving detection performance. We construct a large\-scale AMR dataset specifically tailored for scenarios of the coexistence or overlapping of multiple signals transmitted through channels with realistic propagation conditions, consisting of diverse digital and analog modulation schemes. Extensive experiments on our dataset demonstrate the excellent performance of HIFI\-YOLO in multi\-signal detection and modulation recognition as a joint approach.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2511.23258v1)

---


## SemOD: Semantic Enabled Object Detection Network under Various Weather Conditions / 

发布日期：2025-11-27

作者：Aiyinsi Zuo

摘要：In the field of autonomous driving, camera\-based perception models are mostly trained on clear weather data. Models that focus on addressing specific weather challenges are unable to adapt to various weather changes and primarily prioritize their weather removal characteristics. Our study introduces a semantic\-enabled network for object detection in diverse weather conditions. In our analysis, semantics information can enable the model to generate plausible content for missing areas, understand object boundaries, and preserve visual coherency and realism across both filled\-in and existing portions of the image, which are conducive to image transformation and object recognition. Specific in implementation, our architecture consists of a Preprocessing Unit \(PPU\) and a Detection Unit \(DTU\), where the PPU utilizes a U\-shaped net enriched by semantics to refine degraded images, and the DTU integrates this semantic information for object detection using a modified YOLO network. Our method pioneers the use of semantic data for all\-weather transformations, resulting in an increase between 1.47% to 8.80% in mAP compared to existing methods across benchmark datasets of different weather. This highlights the potency of semantics in image enhancement and object detection, offering a comprehensive approach to improving object detection performance. Code will be available at https://github.com/EnisZuo/SemOD.

中文摘要：


代码链接：https://github.com/EnisZuo/SemOD.

论文链接：[阅读更多](http://arxiv.org/abs/2511.22142v1)

---


## Intelligent Image Search Algorithms Fusing Visual Large Models / 

发布日期：2025-11-25

作者：Kehan Wang

摘要：Fine\-grained image retrieval, which aims to find images containing specific object components and assess their detailed states, is critical in fields like security and industrial inspection. However, conventional methods face significant limitations: manual features \(e.g., SIFT\) lack robustness; deep learning\-based detectors \(e.g., YOLO\) can identify component presence but cannot perform state\-specific retrieval or zero\-shot search; Visual Large Models \(VLMs\) offer semantic and zero\-shot capabilities but suffer from poor spatial grounding and high computational cost, making them inefficient for direct retrieval. To bridge these gaps, this paper proposes DetVLM, a novel intelligent image search framework that synergistically fuses object detection with VLMs. The framework pioneers a search\-enhancement paradigm via a two\-stage pipeline: a YOLO detector first conducts efficient, high\-recall component\-level screening to determine component presence; then, a VLM acts as a recall\-enhancement unit, performing secondary verification for components missed by the detector. This architecture directly enables two advanced capabilities: 1\) State Search: Guided by task\-specific prompts, the VLM refines results by verifying component existence and executing sophisticated state judgments \(e.g., "sun visor lowered"\), allowing retrieval based on component state. 2\) Zero\-shot Search: The framework leverages the VLM's inherent zero\-shot capability to recognize and retrieve images containing unseen components or attributes \(e.g., "driver wearing a mask"\) without any task\-specific training. Experiments on a vehicle component dataset show DetVLM achieves a state\-of\-the\-art overall retrieval accuracy of 94.82%, significantly outperforming detection\-only baselines. It also attains 94.95% accuracy in zero\-shot search for driver mask\-wearing and over 90% average accuracy in state search tasks.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2511.19920v1)

---


## From Pixels to Posts: Retrieval\-Augmented Fashion Captioning and Hashtag Generation / 

发布日期：2025-11-24

作者：Moazzam Umer Gondal

摘要：This paper introduces the retrieval\-augmented framework for automatic fashion caption and hashtag generation, combining multi\-garment detection, attribute reasoning, and Large Language Model \(LLM\) prompting. The system aims to produce visually grounded, descriptive, and stylistically interesting text for fashion imagery, overcoming the limitations of end\-to\-end captioners that have problems with attribute fidelity and domain generalization. The pipeline combines a YOLO\-based detector for multi\-garment localization, k\-means clustering for dominant color extraction, and a CLIP\-FAISS retrieval module for fabric and gender attribute inference based on a structured product index. These attributes, together with retrieved style examples, create a factual evidence pack that is used to guide an LLM to generate human\-like captions and contextually rich hashtags. A fine\-tuned BLIP model is used as a supervised baseline model for comparison. Experimental results show that the YOLO detector is able to obtain a mean Average Precision \(mAP@0.5\) of 0.71 for nine categories of garments. The RAG\-LLM pipeline generates expressive attribute\-aligned captions and achieves mean attribute coverage of 0.80 with full coverage at the 50% threshold in hashtag generation, whereas BLIP gives higher lexical overlap and lower generalization. The retrieval\-augmented approach exhibits better factual grounding, less hallucination, and great potential for scalable deployment in various clothing domains. These results demonstrate the use of retrieval\-augmented generation as an effective and interpretable paradigm for automated and visually grounded fashion content generation.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2511.19149v1)

---

