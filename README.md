# 每日从arXiv中获取最新YOLO相关论文


## Shadow Erosion and Nighttime Adaptability for Camera\-Based Automated Driving Applications / 

发布日期：2025-04-11

作者：Mohamed Sabry

摘要：Enhancement of images from RGB cameras is of particular interest due to its wide range of ever\-increasing applications such as medical imaging, satellite imaging, automated driving, etc. In autonomous driving, various techniques are used to enhance image quality under challenging lighting conditions. These include artificial augmentation to improve visibility in poor nighttime conditions, illumination\-invariant imaging to reduce the impact of lighting variations, and shadow mitigation to ensure consistent image clarity in bright daylight. This paper proposes a pipeline for Shadow Erosion and Nighttime Adaptability in images for automated driving applications while preserving color and texture details. The Shadow Erosion and Nighttime Adaptability pipeline is compared to the widely used CLAHE technique and evaluated based on illumination uniformity and visual perception quality metrics. The results also demonstrate a significant improvement over CLAHE, enhancing a YOLO\-based drivable area segmentation algorithm.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2504.08551v1)

---


## MMLA: Multi\-Environment, Multi\-Species, Low\-Altitude Aerial Footage Dataset / 

发布日期：2025-04-10

作者：Jenna Kline

摘要：Real\-time wildlife detection in drone imagery is critical for numerous applications, including animal ecology, conservation, and biodiversity monitoring. Low\-altitude drone missions are effective for collecting fine\-grained animal movement and behavior data, particularly if missions are automated for increased speed and consistency. However, little work exists on evaluating computer vision models on low\-altitude aerial imagery and generalizability across different species and settings. To fill this gap, we present a novel multi\-environment, multi\-species, low\-altitude aerial footage \(MMLA\) dataset. MMLA consists of drone footage collected across three diverse environments: Ol Pejeta Conservancy and Mpala Research Centre in Kenya, and The Wilds Conservation Center in Ohio, which includes five species: Plains zebras, Grevy's zebras, giraffes, onagers, and African Painted Dogs. We comprehensively evaluate three YOLO models \(YOLOv5m, YOLOv8m, and YOLOv11m\) for detecting animals. Results demonstrate significant performance disparities across locations and species\-specific detection variations. Our work highlights the importance of evaluating detection algorithms across different environments for robust wildlife monitoring applications using drones.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2504.07744v1)

---


## Few\-Shot Adaptation of Grounding DINO for Agricultural Domain / 

发布日期：2025-04-09

作者：Rajhans Singh

摘要：Deep learning models are transforming agricultural applications by enabling automated phenotyping, monitoring, and yield estimation. However, their effectiveness heavily depends on large amounts of annotated training data, which can be labor and time intensive. Recent advances in open\-set object detection, particularly with models like Grounding\-DINO, offer a potential solution to detect regions of interests based on text prompt input. Initial zero\-shot experiments revealed challenges in crafting effective text prompts, especially for complex objects like individual leaves and visually similar classes. To address these limitations, we propose an efficient few\-shot adaptation method that simplifies the Grounding\-DINO architecture by removing the text encoder module \(BERT\) and introducing a randomly initialized trainable text embedding. This method achieves superior performance across multiple agricultural datasets, including plant\-weed detection, plant counting, insect identification, fruit counting, and remote sensing tasks. Specifically, it demonstrates up to a $sim24%$ higher mAP than fully fine\-tuned YOLO models on agricultural datasets and outperforms previous state\-of\-the\-art methods by $sim10%$ in remote sensing, under few\-shot learning conditions. Our method offers a promising solution for automating annotation and accelerating the development of specialized agricultural AI solutions.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2504.07252v1)

---


## Real\-Time Roadway Obstacle Detection for Electric Scooters Using Deep Learning and Multi\-Sensor Fusion / 

发布日期：2025-04-04

作者：Zeyang Zheng

摘要：The increasing adoption of electric scooters \(e\-scooters\) in urban areas has coincided with a rise in traffic accidents and injuries, largely due to their small wheels, lack of suspension, and sensitivity to uneven surfaces. While deep learning\-based object detection has been widely used to improve automobile safety, its application for e\-scooter obstacle detection remains unexplored. This study introduces a novel ground obstacle detection system for e\-scooters, integrating an RGB camera, and a depth camera to enhance real\-time road hazard detection. Additionally, the Inertial Measurement Unit \(IMU\) measures linear vertical acceleration to identify surface vibrations, guiding the selection of six obstacle categories: tree branches, manhole covers, potholes, pine cones, non\-directional cracks, and truncated domes. All sensors, including the RGB camera, depth camera, and IMU, are integrated within the Intel RealSense Camera D435i. A deep learning model powered by YOLO detects road hazards and utilizes depth data to estimate obstacle proximity. Evaluated on the seven hours of naturalistic riding dataset, the system achieves a high mean average precision \(mAP\) of 0.827 and demonstrates excellent real\-time performance. This approach provides an effective solution to enhance e\-scooter safety through advanced computer vision and data fusion. The dataset is accessible at https://zenodo.org/records/14583718, and the project code is hosted on https://github.com/Zeyang\-Zheng/Real\-Time\-Roadway\-Obstacle\-Detection\-for\-Electric\-Scooters.

中文摘要：


代码链接：https://zenodo.org/records/14583718,，https://github.com/Zeyang-Zheng/Real-Time-Roadway-Obstacle-Detection-for-Electric-Scooters.

论文链接：[阅读更多](http://arxiv.org/abs/2504.03171v1)

---


## LLM\-Guided Evolution: An Autonomous Model Optimization for Object Detection / 

发布日期：2025-04-03

作者：YiMing Yu

摘要：In machine learning, Neural Architecture Search \(NAS\) requires domain knowledge of model design and a large amount of trial\-and\-error to achieve promising performance. Meanwhile, evolutionary algorithms have traditionally relied on fixed rules and pre\-defined building blocks. The Large Language Model \(LLM\)\-Guided Evolution \(GE\) framework transformed this approach by incorporating LLMs to directly modify model source code for image classification algorithms on CIFAR data and intelligently guide mutations and crossovers. A key element of LLM\-GE is the "Evolution of Thought" \(EoT\) technique, which establishes feedback loops, allowing LLMs to refine their decisions iteratively based on how previous operations performed. In this study, we perform NAS for object detection by improving LLM\-GE to modify the architecture of You Only Look Once \(YOLO\) models to enhance performance on the KITTI dataset. Our approach intelligently adjusts the design and settings of YOLO to find the optimal algorithms against objective such as detection accuracy and speed. We show that LLM\-GE produced variants with significant performance improvements, such as an increase in Mean Average Precision from 92.5% to 94.5%. This result highlights the flexibility and effectiveness of LLM\-GE on real\-world challenges, offering a novel paradigm for automated machine learning that combines LLM\-driven reasoning with evolutionary strategies.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2504.02280v1)

---

