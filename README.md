# 每日从arXiv中获取最新YOLO相关论文


## WildLive: Near Real\-time Visual Wildlife Tracking onboard UAVs / 

发布日期：2025-04-14

作者：Nguyen Ngoc Dat

摘要：Live tracking of wildlife via high\-resolution video processing directly onboard drones is widely unexplored and most existing solutions rely on streaming video to ground stations to support navigation. Yet, both autonomous animal\-reactive flight control beyond visual line of sight and/or mission\-specific individual and behaviour recognition tasks rely to some degree on this capability. In response, we introduce WildLive \-\- a near real\-time animal detection and tracking framework for high\-resolution imagery running directly onboard uncrewed aerial vehicles \(UAVs\). The system performs multi\-animal detection and tracking at 17fps\+ for HD and 7fps\+ on 4K video streams suitable for operation during higher altitude flights to minimise animal disturbance. Our system is optimised for Jetson Orin AGX onboard hardware. It integrates the efficiency of sparse optical flow tracking and mission\-specific sampling with device\-optimised and proven YOLO\-driven object detection and segmentation techniques. Essentially, computational resource is focused onto spatio\-temporal regions of high uncertainty to significantly improve UAV processing speeds without domain\-specific loss of accuracy. Alongside, we introduce our WildLive dataset, which comprises 200k\+ annotated animal instances across 19k\+ frames from 4K UAV videos collected at the Ol Pejeta Conservancy in Kenya. All frames contain ground truth bounding boxes, segmentation masks, as well as individual tracklets and tracking point trajectories. We compare our system against current object tracking approaches including OC\-SORT, ByteTrack, and SORT. Our multi\-animal tracking experiments with onboard hardware confirm that near real\-time high\-resolution wildlife tracking is possible on UAVs whilst maintaining high accuracy levels as needed for future navigational and mission\-specific animal\-centric operational autonomy.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2504.10165v1)

---


## Small Object Detection with YOLO: A Performance Analysis Across Model Versions and Hardware / 

发布日期：2025-04-14

作者：Muhammad Fasih Tariq

摘要：This paper provides an extensive evaluation of YOLO object detection models \(v5, v8, v9, v10, v11\) by com\- paring their performance across various hardware platforms and optimization libraries. Our study investigates inference speed and detection accuracy on Intel and AMD CPUs using popular libraries such as ONNX and OpenVINO, as well as on GPUs through TensorRT and other GPU\-optimized frameworks. Furthermore, we analyze the sensitivity of these YOLO models to object size within the image, examining performance when detecting objects that occupy 1%, 2.5%, and 5% of the total area of the image. By identifying the trade\-offs in efficiency, accuracy, and object size adaptability, this paper offers insights for optimal model selection based on specific hardware constraints and detection requirements, aiding practitioners in deploying YOLO models effectively for real\-world applications.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2504.09900v1)

---


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

