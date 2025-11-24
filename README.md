# 每日从arXiv中获取最新YOLO相关论文


## Video\-R4: Reinforcing Text\-Rich Video Reasoning with Visual Rumination / 

发布日期：2025-11-21

作者：Yolo Yunlong Tang

摘要：Understanding text\-rich videos requires reading small, transient textual cues that often demand repeated inspection. Yet most video QA models rely on single\-pass perception over fixed frames, leading to hallucinations and failures on fine\-grained evidence. Inspired by how humans pause, zoom, and re\-read critical regions, we introduce Video\-R4 \(Reinforcing Text\-Rich Video Reasoning with Visual Rumination\), a video reasoning LMM that performs visual rumination: iteratively selecting frames, zooming into informative regions, re\-encoding retrieved pixels, and updating its reasoning state. We construct two datasets with executable rumination trajectories: Video\-R4\-CoT\-17k for supervised practice and Video\-R4\-RL\-30k for reinforcement learning. We propose a multi\-stage rumination learning framework that progressively finetunes a 7B LMM to learn atomic and mixing visual operations via SFT and GRPO\-based RL. Video\-R4\-7B achieves state\-of\-the\-art results on M4\-ViteVQA and further generalizes to multi\-page document QA, slides QA, and generic video QA, demonstrating that iterative rumination is an effective paradigm for pixel\-grounded multimodal reasoning.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2511.17490v1)

---


## A lightweight detector for real\-time detection of remote sensing images / 

发布日期：2025-11-21

作者：Qianyi Wang

摘要：Remote sensing imagery is widely used across various fields, yet real\-time detection remains challenging due to the prevalence of small objects and the need to balance accuracy with efficiency. To address this, we propose DMG\-YOLO, a lightweight real\-time detector tailored for small object detection in remote sensing images. Specifically, we design a Dual\-branch Feature Extraction \(DFE\) module in the backbone, which partitions feature maps into two parallel branches: one extracts local features via depthwise separable convolutions, and the other captures global context using a vision transformer with a gating mechanism. Additionally, a Multi\-scale Feature Fusion \(MFF\) module with dilated convolutions enhances multi\-scale integration while preserving fine details. In the neck, we introduce the Global and Local Aggregate Feature Pyramid Network \(GLAFPN\) to further boost small object detection through global\-local feature fusion. Extensive experiments on the VisDrone2019 and NWPU VHR\-10 datasets show that DMG\-YOLO achieves competitive performance in terms of mAP, model size, and other key metrics.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2511.17147v1)

---


## Cranio\-ID: Graph\-Based Craniofacial Identification via Automatic Landmark Annotation in 2D Multi\-View X\-rays / 

发布日期：2025-11-18

作者：Ravi Shankar Prasad

摘要：In forensic craniofacial identification and in many biomedical applications, craniometric landmarks are important. Traditional methods for locating landmarks are time\-consuming and require specialized knowledge and expertise. Current methods utilize superimposition and deep learning\-based methods that employ automatic annotation of landmarks. However, these methods are not reliable due to insufficient large\-scale validation studies. In this paper, we proposed a novel framework Cranio\-ID: First, an automatic annotation of landmarks on 2D skulls \(which are X\-ray scans of faces\) with their respective optical images using our trained YOLO\-pose models. Second, cross\-modal matching by formulating these landmarks into graph representations and then finding semantic correspondence between graphs of these two modalities using cross\-attention and optimal transport framework. Our proposed framework is validated on the S2F and CUHK datasets \(CUHK dataset resembles with S2F dataset\). Extensive experiments have been conducted to evaluate the performance of our proposed framework, which demonstrates significant improvements in both reliability and accuracy, as well as its effectiveness in cross\-domain skull\-to\-face and sketch\-to\-face matching in forensic science.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2511.14411v1)

---


## LSP\-YOLO: A Lightweight Single\-Stage Network for Sitting Posture Recognition on Embedded Devices / 

发布日期：2025-11-18

作者：Nanjun Li

摘要：With the rise in sedentary behavior, health problems caused by poor sitting posture have drawn increasing attention. Most existing methods, whether using invasive sensors or computer vision, rely on two\-stage pipelines, which result in high intrusiveness, intensive computation, and poor real\-time performance on embedded edge devices. Inspired by YOLOv11\-Pose, a lightweight single\-stage network for sitting posture recognition on embedded edge devices termed LSP\-YOLO was proposed. By integrating partial convolution\(PConv\) and Similarity\-Aware Activation Module\(SimAM\), a lightweight module, Light\-C3k2, was designed to reduce computational cost while maintaining feature extraction capability. In the recognition head, keypoints were directly mapped to posture classes through pointwise convolution, and intermediate supervision was employed to enable efficient fusion of pose estimation and classification. Furthermore, a dataset containing 5,000 images across six posture categories was constructed for model training and testing. The smallest trained model, LSP\-YOLO\-n, achieved 94.2% accuracy and 251 Fps on personal computer\(PC\) with a model size of only 1.9 MB. Meanwhile, real\-time and high\-accuracy inference under constrained computational resources was demonstrated on the SV830C \+ GC030A platform. The proposed approach is characterized by high efficiency, lightweight design and deployability, making it suitable for smart classrooms, rehabilitation, and human\-computer interaction applications.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2511.14322v1)

---


## Hardware optimization on Android for inference of AI models / 

发布日期：2025-11-17

作者：Iulius Gherasim

摘要：The pervasive integration of Artificial Intelligence models into contemporary mobile computing is notable across numerous use cases, from virtual assistants to advanced image processing. Optimizing the mobile user experience involves minimal latency and high responsiveness from deployed AI models with challenges from execution strategies that fully leverage real time constraints to the exploitation of heterogeneous hardware architecture. In this paper, we research and propose the optimal execution configurations for AI models on an Android system, focusing on two critical tasks: object detection \(YOLO family\) and image classification \(ResNet\). These configurations evaluate various model quantization schemes and the utilization of on device accelerators, specifically the GPU and NPU. Our core objective is to empirically determine the combination that achieves the best trade\-off between minimal accuracy degradation and maximal inference speed\-up.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2511.13453v1)

---

