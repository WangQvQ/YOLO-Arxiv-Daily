# 每日从arXiv中获取最新YOLO相关论文


## BoardVision: Deployment\-ready and Robust Motherboard Defect Detection with YOLO\+Faster\-RCNN Ensemble / 

发布日期：2025-10-16

作者：Brandon Hill

摘要：Motherboard defect detection is critical for ensuring reliability in high\-volume electronics manufacturing. While prior research in PCB inspection has largely targeted bare\-board or trace\-level defects, assembly\-level inspection of full motherboards inspection remains underexplored. In this work, we present BoardVision, a reproducible framework for detecting assembly\-level defects such as missing screws, loose fan wiring, and surface scratches. We benchmark two representative detectors \- YOLOv7 and Faster R\-CNN, under controlled conditions on the MiracleFactory motherboard dataset, providing the first systematic comparison in this domain. To mitigate the limitations of single models, where YOLO excels in precision but underperforms in recall and Faster R\-CNN shows the reverse, we propose a lightweight ensemble, Confidence\-Temporal Voting \(CTV Voter\), that balances precision and recall through interpretable rules. We further evaluate robustness under realistic perturbations including sharpness, brightness, and orientation changes, highlighting stability challenges often overlooked in motherboard defect detection. Finally, we release a deployable GUI\-driven inspection tool that bridges research evaluation with operator usability. Together, these contributions demonstrate how computer vision techniques can transition from benchmark results to practical quality assurance for assembly\-level motherboard manufacturing.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2510.14389v1)

---


## Efficient Few\-Shot Learning in Remote Sensing: Fusing Vision and Vision\-Language Models / 

发布日期：2025-10-15

作者：Jia Yun Chua

摘要：Remote sensing has become a vital tool across sectors such as urban planning, environmental monitoring, and disaster response. While the volume of data generated has increased significantly, traditional vision models are often constrained by the requirement for extensive domain\-specific labelled data and their limited ability to understand the context within complex environments. Vision Language Models offer a complementary approach by integrating visual and textual data; however, their application to remote sensing remains underexplored, particularly given their generalist nature. This work investigates the combination of vision models and VLMs to enhance image analysis in remote sensing, with a focus on aircraft detection and scene understanding. The integration of YOLO with VLMs such as LLaVA, ChatGPT, and Gemini aims to achieve more accurate and contextually aware image interpretation. Performance is evaluated on both labelled and unlabelled remote sensing data, as well as degraded image scenarios which are crucial for remote sensing. The findings show an average MAE improvement of 48.46% across models in the accuracy of aircraft detection and counting, especially in challenging conditions, in both raw and degraded scenarios. A 6.17% improvement in CLIPScore for comprehensive understanding of remote sensing images is obtained. The proposed approach combining traditional vision models and VLMs paves the way for more advanced and efficient remote sensing image analysis, especially in few\-shot learning scenarios.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2510.13993v1)

---


## A Modular Object Detection System for Humanoid Robots Using YOLO / 

发布日期：2025-10-15

作者：Nicolas Pottier

摘要：Within the field of robotics, computer vision remains a significant barrier to progress, with many tasks hindered by inefficient vision systems. This research proposes a generalized vision module leveraging YOLOv9, a state\-of\-the\-art framework optimized for computationally constrained environments like robots. The model is trained on a dataset tailored to the FIRA robotics Hurocup. A new vision module is implemented in ROS1 using a virtual environment to enable YOLO compatibility. Performance is evaluated using metrics such as frames per second \(FPS\) and Mean Average Precision \(mAP\). Performance is then compared to the existing geometric framework in static and dynamic contexts. The YOLO model achieved comparable precision at a higher computational cost then the geometric model, while providing improved robustness.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2510.13625v1)

---


## DEF\-YOLO: Leveraging YOLO for Concealed Weapon Detection in Thermal Imagin / 

发布日期：2025-10-15

作者：Divya Bhardwaj

摘要：Concealed weapon detection aims at detecting weapons hidden beneath a person's clothing or luggage. Various imaging modalities like Millimeter Wave, Microwave, Terahertz, Infrared, etc., are exploited for the concealed weapon detection task. These imaging modalities have their own limitations, such as poor resolution in microwave imaging, privacy concerns in millimeter wave imaging, etc. To provide a real\-time, 24 x 7 surveillance, low\-cost, and privacy\-preserved solution, we opted for thermal imaging in spite of the lack of availability of a benchmark dataset. We propose a novel approach and a dataset for concealed weapon detection in thermal imagery. Our YOLO\-based architecture, DEF\-YOLO, is built with key enhancements in YOLOv8 tailored to the unique challenges of concealed weapon detection in thermal vision. We adopt deformable convolutions at the SPPF layer to exploit multi\-scale features; backbone and neck layers to extract low, mid, and high\-level features, enabling DEF\-YOLO to adaptively focus on localization around the objects in thermal homogeneous regions, without sacrificing much of the speed and throughput. In addition to these simple yet effective key architectural changes, we introduce a new, large\-scale Thermal Imaging Concealed Weapon dataset, TICW, featuring a diverse set of concealed weapons and capturing a wide range of scenarios. To the best of our knowledge, this is the first large\-scale contributed dataset for this task. We also incorporate focal loss to address the significant class imbalance inherent in the concealed weapon detection task. The efficacy of the proposed work establishes a new benchmark through extensive experimentation for concealed weapon detection in thermal imagery.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2510.13326v1)

---


## Detect Anything via Next Point Prediction / 

发布日期：2025-10-14

作者：Qing Jiang

摘要：Object detection has long been dominated by traditional coordinate regression\-based models, such as YOLO, DETR, and Grounding DINO. Although recent efforts have attempted to leverage MLLMs to tackle this task, they face challenges like low recall rate, duplicate predictions, coordinate misalignment, etc. In this work, we bridge this gap and propose Rex\-Omni, a 3B\-scale MLLM that achieves state\-of\-the\-art object perception performance. On benchmarks like COCO and LVIS, Rex\-Omni attains performance comparable to or exceeding regression\-based models \(e.g., DINO, Grounding DINO\) in a zero\-shot setting. This is enabled by three key designs: 1\) Task Formulation: we use special tokens to represent quantized coordinates from 0 to 999, reducing the model's learning difficulty and improving token efficiency for coordinate prediction; 2\) Data Engines: we construct multiple data engines to generate high\-quality grounding, referring, and pointing data, providing semantically rich supervision for training; 3\) Training Pipelines: we employ a two\-stage training process, combining supervised fine\-tuning on 22 million data with GRPO\-based reinforcement post\-training. This RL post\-training leverages geometry\-aware rewards to effectively bridge the discrete\-to\-continuous coordinate prediction gap, improve box accuracy, and mitigate undesirable behaviors like duplicate predictions that stem from the teacher\-guided nature of the initial SFT stage. Beyond conventional detection, Rex\-Omni's inherent language understanding enables versatile capabilities such as object referring, pointing, visual prompting, GUI grounding, spatial referring, OCR and key\-pointing, all systematically evaluated on dedicated benchmarks. We believe that Rex\-Omni paves the way for more versatile and language\-aware visual perception systems.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2510.12798v1)

---

