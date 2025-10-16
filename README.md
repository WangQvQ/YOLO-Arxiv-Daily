# 每日从arXiv中获取最新YOLO相关论文


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


## Enhancing Maritime Domain Awareness on Inland Waterways: A YOLO\-Based Fusion of Satellite and AIS for Vessel Characterization / 

发布日期：2025-10-13

作者：Geoffery Agorku

摘要：Maritime Domain Awareness \(MDA\) for inland waterways remains challenged by cooperative system vulnerabilities. This paper presents a novel framework that fuses high\-resolution satellite imagery with vessel trajectory data from the Automatic Identification System \(AIS\). This work addresses the limitations of AIS\-based monitoring by leveraging non\-cooperative satellite imagery and implementing a fusion approach that links visual detections with AIS data to identify dark vessels, validate cooperative traffic, and support advanced MDA. The You Only Look Once \(YOLO\) v11 object detection model is used to detect and characterize vessels and barges by vessel type, barge cover, operational status, barge count, and direction of travel. An annotated data set of 4,550 instances was developed from $5\{,\}973~mathrm\{mi\}^2$ of Lower Mississippi River imagery. Evaluation on a held\-out test set demonstrated vessel classification \(tugboat, crane barge, bulk carrier, cargo ship, and hopper barge\) with an F1 score of 95.8%; barge cover \(covered or uncovered\) detection yielded an F1 score of 91.6%; operational status \(staged or in motion\) classification reached an F1 score of 99.4%. Directionality \(upstream, downstream\) yielded 93.8% accuracy. The barge count estimation resulted in a mean absolute error \(MAE\) of 2.4 barges. Spatial transferability analysis across geographically disjoint river segments showed accuracy was maintained as high as 98%. These results underscore the viability of integrating non\-cooperative satellite sensing with AIS fusion. This approach enables near\-real\-time fleet inventories, supports anomaly detection, and generates high\-quality data for inland waterway surveillance. Future work will expand annotated datasets, incorporate temporal tracking, and explore multi\-modal deep learning to further enhance operational scalability.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2510.11449v1)

---


## When Does Supervised Training Pay Off? The Hidden Economics of Object Detection in the Era of Vision\-Language Models / 

发布日期：2025-10-13

作者：Samer Al\-Hamadani

摘要：Object detection systems have traditionally relied on supervised learning with manually annotated bounding boxes, achieving high accuracy at the cost of substantial annotation investment. The emergence of Vision\-Language Models \(VLMs\) offers an alternative paradigm enabling zero\-shot detection through natural language queries, eliminating annotation requirements but operating with reduced accuracy. This paper presents the first comprehensive cost\-effectiveness analysis comparing supervised detection \(YOLO\) with zero\-shot VLM inference \(Gemini Flash 2.5\). Through systematic evaluation on 1,000 stratified COCO images and 200 diverse product images spanning consumer electronics and rare categories, combined with detailed Total Cost of Ownership modeling, we establish quantitative break\-even thresholds governing architecture selection. Our findings reveal that supervised YOLO achieves 91.2% accuracy versus 68.5% for zero\-shot Gemini on standard categories, representing a 22.7 percentage point advantage that costs $10,800 in annotation for 100\-category systems. However, this advantage justifies investment only beyond 55 million inferences, equivalent to 151,000 images daily for one year. Zero\-shot Gemini demonstrates 52.3% accuracy on diverse product categories \(ranging from highly web\-prevalent consumer electronics at 75\-85% to rare specialized equipment at 25\-40%\) where supervised YOLO achieves 0% due to architectural constraints preventing detection of untrained classes. Cost per Correct Detection analysis reveals substantially lower per\-detection costs for Gemini \($0.00050 vs $0.143\) at 100,000 inferences despite accuracy deficits. We develop decision frameworks demonstrating that optimal architecture selection depends critically on deployment volume, category stability, budget constraints, and accuracy requirements rather than purely technical performance metrics.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2510.11302v1)

---

