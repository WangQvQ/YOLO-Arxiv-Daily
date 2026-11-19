<div align="center">

# YOLO ArXiv Daily

[![Daily Papers](https://img.shields.io/badge/📅-每日更新-blue)]()
[![arXiv](https://img.shields.io/badge/arXiv-最新论文-red)](https://arxiv.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)

*自动追踪 YOLO 相关最新论文，提供中英文双语摘要*

</div>

---

## 📑 论文列表

> ### 1. Enhancing Active Learning Performance with Initial Data Selection Based on Foundation Models
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-11-18 |
> | 👤 作者 | Mohammad Mirzaee |
>
> **📄 英文摘要：**
> In active object detection, selecting the initial labeled pool is essential, especially in complex datasets with expensive labeling and low resource budget. This study presents an active learning framework designed to optimize the initial training cycle of active learning for the object detection task. Our approach diverges from traditional methods by utilizing feature\-based clustering from the YOLO\-World model and refining images based on objectness scores rather than relying on uncertainty sampling in the initial cycle. This approach focuses on finding samples with various features and emphasizes training in areas where the foundation model is most confident. Doing so helps the model better understand different types of objects. For subsequent cycles, we employ a consistency\-based sampling technique to maintain the robustness of the training process. To evaluate our method, we use two complex datasets called Aquarium and NWPU VHR\-10, which are well known for their diverse object sizes and challenging conditions. In the NWPU dataset, our method increases mAP50 by an average of 6.6% in image\-based evaluation. Moreover, the image refinement method shows an improvement of approximately 11% in object\-based evaluation. The results show improvements in model performance and labeling time, demonstrating the advantages of using foundation models in early cycles of active learning. An implementation of the algorithm is available at https://github.com/Mhrshd\-fllh/ALBoost\-FM
>
> **💻 代码链接：** https://github.com/Mhrshd-fllh/ALBoost-FM
>
> 🔗 [阅读论文](http://arxiv.org/abs/2611.24384v1)

---

> ### 2. UnrealEvent: Synthetic Dataset Generation and Annotation for Event\-Based Vision
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-11-17 |
> | 👤 作者 | Yentel Schoovaerts |
>
> **📄 英文摘要：**
> Event\-based cameras provide high temporal resolution, low power consumption, and a high dynamic range, but progress in event\-based object detection remains constrained by the scarcity of annotated training data. To solve this issue, this paper introduces the tool UnrealEvent, a synthetic dataset\-generation pipeline implemented in Unreal Engine 5.5. The pipeline combines a V2E\-inspired event\-frame generator with EasySynth semantic\-rendering passes, allowing rendered level sequences to be converted into paired event representations and YOLO\-format bounding\-box annotations without manual labeling. Events are generated from per\-pixel log\-luminance differences between consecutive frames, with contrast thresholds sampled independently from a normal distribution to emulate the fabrication\-induced threshold mismatch of real sensors. Using this pipeline, we provide a synthetic dataset and evaluate YOLOv8 detectors under synthetic, zero\-shot real, and real fine\-tuning training on eTraM. The results show strong in\-domain synthetic performance but a substantial synthetic\-to\-real gap, especially for pedestrians. Fine\-tuning on a small real subset improves real\-domain performance, while the remaining gap highlights the importance of asset diversity and sensor calibration.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2611.23078v1)

---

> ### 3. RoadAnnoter: An AI\-Assisted Desktop Tool for Semantic Segmentation Annotation of Road\-Scene Images
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-11-17 |
> | 👤 作者 | Nazmul Haque |
>
> **📄 英文摘要：**
> Accurate semantic segmentation of road\-scene images is essential for road safety research; however, polygon\-level annotation remains labor\-intensive and limits labelled dataset development. This study presents RoadAnnoter v1.0, an open\-source desktop application integrating six human\-in\-the\-loop annotation modes: manual polygon delineation, SAM Prompt, Smart Select, Edge Fill, pretrained semantic segmentation using Cityscapes\-trained SegFormer models, and task\-specific segmentation using TinySegNet, a lightweight MobileNetV3\-Small\-based network trained with accumulated annotator\-generated labels. SAM\-based modes use promptable segmentation with SAM and MobileSAM, while Edge Fill combines Canny edge detection with colour\-similarity\-based flood filling in CIELAB colour space. SegFormer generates suggestions for common road\-scene categories, whereas TinySegNet supports adaptation to project\-specific features through iterative training. All six modes use a shared mask\-to\-polygon pipeline comprising morphological processing, Gaussian smoothing, contour extraction, and Douglas\-Peucker simplification; annotations are exported in YOLO segmentation format. RoadAnnoter was used to annotate Google Street View images from recorded road accident locations in Bangladesh for a vision\-language model\-based investigation of free\-flow speed estimation. Qualitative assessment across representative scenes demonstrated the complementary applicability of the annotation modes to diverse feature geometries. Visual comparison with Roboflow indicated similar qualitative boundary characteristics for principal road\-scene regions, although no quantitative accuracy assessment was conducted. RoadAnnoter provides a locally operated framework for developing road\-scene segmentation datasets for transportation engineering and road safety research.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2611.22884v1)

---

> ### 4. Technical Report of Mobile Manipulator Robot for Industrial Environments
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-11-17 |
> | 👤 作者 | Aria Dadnavi |
>
> **📄 英文摘要：**
> This paper presents the design and implementation of Auriga Robotics' mobile manipulator robot, developed at the Robotics and Intelligent Automation Laboratory, Department of Electrical Engineering, Shahid Beheshti University. Hereafter, this platform is referred to as the @Work Robot. The robot targets industrial tasks, with an emphasis on improving efficiency in repetitive or hazardous work environments. It uses a 4\-wheel Mecanum drive for omnidirectional movement and a 5\-degree\-of\-freedom manipulator arm with a 3D\-printed gripper for object handling and navigation. The electronics consist of custom PCB boards built around ESP32 microcontrollers, together with an NVIDIA Jetson Orin NX that handles high\-level processing and real\-time system control. On the software side, Hector SLAM provides mapping and localization, A\* handles path planning, and YOLO performs real\-time object detection. These modules draw on fused sensor data to deliver robust navigation and reliable collision avoidance. Overall, the system shows effective hardware\-software integration, with consistent performance in autonomous navigation, object handling, and task execution in industrial\-like settings.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2611.22852v1)

---

> ### 5. Counting What Matters: Reducing Redundant Object Detections in Overlapping UAV Mapping Imagery for Disaster Response
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-11-16 |
> | 👤 作者 | Max Schulte |
>
> **📄 英文摘要：**
> UAV mapping flights enable rapid aerial reconnaissance of large\-scale incident sites in civil protection. However, the image overlap required for orthophoto generation causes every physical object to be detected multiple times across images, while dynamic objects may be omitted from or duplicated in the resulting orthomosaic. Analysis based on the orthophoto alone is therefore structurally insufficient. We address this by detecting objects on the individual images and merging redundant detections across them. This task lies between multi\-object tracking and re\-identification, as mapping flights form temporally ordered but sparsely sampled sequences with object reappearances across flight strips. We fine\-tune YOLO detectors on a new domain\-specific nadir dataset from real civil protection operations, roughly doubling mAP50 over the previous system, and propose re\-identification approaches based on 3D positions from sparse reconstruction, DINOv3 appearance embeddings, and their combination via constrained greedy clustering. %On four manually annotated real\-world mapping flights, the combined weighted multi\-pass approach achieves the best trade\-off between association accuracy and count error. On four manually annotated real\-world mapping flights, an association\-only evaluation using ground\-truth detections shows that the combined weighted multi\-pass approach provides the best trade\-off between association quality and count error. It is integrated into the open\-source web application ARGUS and visualized as verifiable, correctable object groups for first responders. https://github.com/RoblabWh/argus
>
> **💻 代码链接：** https://github.com/RoblabWh/argus
>
> 🔗 [阅读论文](http://arxiv.org/abs/2611.21076v1)

---

<div align="center">

*由 [YOLO-Arxiv-Daily](https://github.com/WangQvQ/YOLO-Arxiv-Daily) 自动生成*

</div>