# 每日从arXiv中获取最新YOLO相关论文


## Real\-Time Control and Automation Framework for Acousto\-Holographic Microscopy / 

发布日期：2025-12-03

作者：Hasan Berkay Abdioğlu

摘要：Manual operation of microscopes for repetitive tasks in cell biology is a significant bottleneck, consuming invaluable expert time, and introducing human error. Automation is essential, and while Digital Holographic Microscopy \(DHM\) offers powerful, label\-free quantitative phase imaging \(QPI\), its inherently noisy and low\-contrast holograms make robust autofocus and object detection challenging. We present the design, integration, and validation of a fully automated closed\-loop DHM system engineered for high\-throughput mechanical characterization of biological cells. The system integrates automated serpentine scanning, real\-time YOLO\-based object detection, and a high\-performance, multi\-threaded software architecture using pinned memory and SPSC queues. This design enables the GPU\-accelerated reconstruction pipeline to run fully in parallel with the 50 fps data acquisition, adding no sequential overhead. A key contribution is the validation of a robust, multi\-stage holographic autofocus strategy; we demonstrate that a selected metric \(based on a low\-pass filter and standard deviation\) provides reliable focusing for noisy holograms where conventional methods \(e.g., Tenengrad, Laplacian\) fail entirely. Performance analysis of the complete system identifies the 2.23\-second autofocus operation\-not reconstruction\-as the primary throughput bottleneck, resulting in a 9.62\-second analysis time per object. This work delivers a complete functional platform for autonomous DHM screening and provides a clear, data\-driven path for future optimization, proposing a hybrid brightfield imaging modality to address current bottlenecks.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2512.03539v1)

---


## AfroBeats Dance Movement Analysis Using Computer Vision: A Proof\-of\-Concept Framework Combining YOLO and Segment Anything Model / 

发布日期：2025-12-03

作者：Kwaku Opoku\-Ware

摘要：This paper presents a preliminary investigation into automated dance movement analysis using contemporary computer vision techniques. We propose a proof\-of\-concept framework that integrates YOLOv8 and v11 for dancer detection with the Segment Anything Model \(SAM\) for precise segmentation, enabling the tracking and quantification of dancer movements in video recordings without specialized equipment or markers. Our approach identifies dancers within video frames, counts discrete dance steps, calculates spatial coverage patterns, and measures rhythm consistency across performance sequences. Testing this framework on a single 49\-second recording of Ghanaian AfroBeats dance demonstrates technical feasibility, with the system achieving approximately 94% detection precision and 89% recall on manually inspected samples. The pixel\-level segmentation provided by SAM, achieving approximately 83% intersection\-over\-union with visual inspection, enables motion quantification that captures body configuration changes beyond what bounding\-box approaches can represent. Analysis of this preliminary case study indicates that the dancer classified as primary by our system executed 23% more steps with 37% higher motion intensity and utilized 42% more performance space compared to dancers classified as secondary. However, this work represents an early\-stage investigation with substantial limitations including single\-video validation, absence of systematic ground truth annotations, and lack of comparison with existing pose estimation methods. We present this framework to demonstrate technical feasibility, identify promising directions for quantitative dance metrics, and establish a foundation for future systematic validation studies.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2512.03509v1)

---


## YOLOA: Real\-Time Affordance Detection via LLM Adapter / 

发布日期：2025-12-03

作者：Yuqi Ji

摘要：Affordance detection aims to jointly address the fundamental "what\-where\-how" challenge in embodied AI by understanding "what" an object is, "where" the object is located, and "how" it can be used. However, most affordance learning methods focus solely on "how" objects can be used while neglecting the "what" and "where" aspects. Other affordance detection methods treat object detection and affordance learning as two independent tasks, lacking effective interaction and real\-time capability. To overcome these limitations, we introduce YOLO Affordance \(YOLOA\), a real\-time affordance detection model that jointly handles these two tasks via a large language model \(LLM\) adapter. Specifically, YOLOA employs a lightweight detector consisting of object detection and affordance learning branches refined through the LLM Adapter. During training, the LLM Adapter interacts with object and affordance preliminary predictions to refine both branches by generating more accurate class priors, box offsets, and affordance gates. Experiments on our relabeled ADG\-Det and IIT\-Heat benchmarks demonstrate that YOLOA achieves state\-of\-the\-art accuracy \(52.8 / 73.1 mAP on ADG\-Det / IIT\-Heat\) while maintaining real\-time performance \(up to 89.77 FPS, and up to 846.24 FPS for the lightweight variant\). This indicates that YOLOA achieves an excellent trade\-off between accuracy and efficiency.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2512.03418v1)

---


## Edge\-Native, Behavior\-Adaptive Drone System for Wildlife Monitoring / 

发布日期：2025-12-01

作者：Jenna Kline

摘要：Wildlife monitoring with drones must balance competing demands: approaching close enough to capture behaviorally\-relevant video while avoiding stress responses that compromise animal welfare and data validity. Human operators face a fundamental attentional bottleneck: they cannot simultaneously control drone operations and monitor vigilance states across entire animal groups. By the time elevated vigilance becomes obvious, an adverse flee response by the animals may be unavoidable. To solve this challenge, we present an edge\-native, behavior\-adaptive drone system for wildlife monitoring. This configurable decision\-support system augments operator expertise with automated group\-level vigilance monitoring. Our system continuously tracks individual behaviors using YOLOv11m detection and YOLO\-Behavior classification, aggregates vigilance states into a real\-time group stress metric, and provides graduated alerts \(alert vigilance to flee response\) with operator\-tunable thresholds for context\-specific calibration. We derive service\-level objectives \(SLOs\) from video frame rates and behavioral dynamics: to monitor 30fps video streams in real\-time, our system must complete detection and classification within 33ms per frame. Our edge\-native pipeline achieves 23.8ms total inference on GPU\-accelerated hardware, meeting this constraint with a substantial margin. Retrospective analysis of seven wildlife monitoring missions demonstrates detection capability and quantifies the cost of reactive control: manual piloting results in 14 seconds average adverse behavior duration with 71.9% usable frames. Our analysis reveals operators could have received actionable alerts 51s before animals fled in 57% of missions. Simulating 5\-second operator intervention yields a projected performance of 82.8% usable frames with 1\-second adverse behavior duration,a 93% reduction compared to manual piloting.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2512.02285v1)

---


## Real\-Time On\-the\-Go Annotation Framework Using YOLO for Automated Dataset Generation / 

发布日期：2025-12-01

作者：Mohamed Abdallah Salem

摘要：Efficient and accurate annotation of datasets remains a significant challenge for deploying object detection models such as You Only Look Once \(YOLO\) in real\-world applications, particularly in agriculture where rapid decision\-making is critical. Traditional annotation techniques are labor\-intensive, requiring extensive manual labeling post data collection. This paper presents a novel real\-time annotation approach leveraging YOLO models deployed on edge devices, enabling immediate labeling during image capture. To comprehensively evaluate the efficiency and accuracy of our proposed system, we conducted an extensive comparative analysis using three prominent YOLO architectures \(YOLOv5, YOLOv8, YOLOv12\) under various configurations: single\-class versus multi\-class annotation and pretrained versus scratch\-based training. Our analysis includes detailed statistical tests and learning dynamics, demonstrating significant advantages of pretrained and single\-class configurations in terms of model convergence, performance, and robustness. Results strongly validate the feasibility and effectiveness of our real\-time annotation framework, highlighting its capability to drastically reduce dataset preparation time while maintaining high annotation quality.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2512.01165v1)

---

