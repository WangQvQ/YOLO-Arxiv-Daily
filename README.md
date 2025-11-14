# 每日从arXiv中获取最新YOLO相关论文


## Robust Object Detection with Pseudo Labels from VLMs using Per\-Object Co\-teaching / 

发布日期：2025-11-13

作者：Uday Bhaskar

摘要：Foundation models, especially vision\-language models \(VLMs\), offer compelling zero\-shot object detection for applications like autonomous driving, a domain where manual labelling is prohibitively expensive. However, their detection latency and tendency to hallucinate predictions render them unsuitable for direct deployment. This work introduces a novel pipeline that addresses this challenge by leveraging VLMs to automatically generate pseudo\-labels for training efficient, real\-time object detectors. Our key innovation is a per\-object co\-teaching\-based training strategy that mitigates the inherent noise in VLM\-generated labels. The proposed per\-object coteaching approach filters noisy bounding boxes from training instead of filtering the entire image. Specifically, two YOLO models learn collaboratively, filtering out unreliable boxes from each mini\-batch based on their peers' per\-object loss values. Overall, our pipeline provides an efficient, robust, and scalable approach to train high\-performance object detectors for autonomous driving, significantly reducing reliance on costly human annotation. Experimental results on the KITTI dataset demonstrate that our method outperforms a baseline YOLOv5m model, achieving a significant mAP@0.5 boost \($31.12%$ to $46.61%$\) while maintaining real\-time detection latency. Furthermore, we show that supplementing our pseudo\-labelled data with a small fraction of ground truth labels \($10%$\) leads to further performance gains, reaching $57.97%$ mAP@0.5 on the KITTI dataset. We observe similar performance improvements for the ACDC and BDD100k datasets.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2511.09955v1)

---


## DKDS: A Benchmark Dataset of Degraded Kuzushiji Documents with Seals for Detection and Binarization / 

发布日期：2025-11-12

作者：Rui\-Yang Ju

摘要：Kuzushiji, a pre\-modern Japanese cursive script, can currently be read and understood by only a few thousand trained experts in Japan. With the rapid development of deep learning, researchers have begun applying Optical Character Recognition \(OCR\) techniques to transcribe Kuzushiji into modern Japanese. Although existing OCR methods perform well on clean pre\-modern Japanese documents written in Kuzushiji, they often fail to consider various types of noise, such as document degradation and seals, which significantly affect recognition accuracy. To the best of our knowledge, no existing dataset specifically addresses these challenges. To address this gap, we introduce the Degraded Kuzushiji Documents with Seals \(DKDS\) dataset as a new benchmark for related tasks. We describe the dataset construction process, which required the assistance of a trained Kuzushiji expert, and define two benchmark tracks: \(1\) text and seal detection and \(2\) document binarization. For the text and seal detection track, we provide baseline results using multiple versions of the You Only Look Once \(YOLO\) models for detecting Kuzushiji characters and seals. For the document binarization track, we present baseline results from traditional binarization algorithms, traditional algorithms combined with K\-means clustering, and Generative Adversarial Network \(GAN\)\-based methods. The DKDS dataset and the implementation code for baseline methods are available at https://ruiyangju.github.io/DKDS.

中文摘要：


代码链接：https://ruiyangju.github.io/DKDS.

论文链接：[阅读更多](http://arxiv.org/abs/2511.09117v1)

---


## Hardware\-Aware YOLO Compression for Low\-Power Edge AI on STM32U5 for Weeds Detection in Digital Agriculture / 

发布日期：2025-11-11

作者：Charalampos S. Kouzinopoulos

摘要：Weeds significantly reduce crop yields worldwide and pose major challenges to sustainable agriculture. Traditional weed management methods, primarily relying on chemical herbicides, risk environmental contamination and lead to the emergence of herbicide\-resistant species. Precision weeding, leveraging computer vision and machine learning methods, offers a promising eco\-friendly alternative but is often limited by reliance on high\-power computational platforms. This work presents an optimized, low\-power edge AI system for weeds detection based on the YOLOv8n object detector deployed on the STM32U575ZI microcontroller. Several compression techniques are applied to the detection model, including structured pruning, integer quantization and input image resolution scaling in order to meet strict hardware constraints. The model is trained and evaluated on the CropAndWeed dataset with 74 plant species, achieving a balanced trade\-off between detection accuracy and efficiency. Our system supports real\-time, in\-situ weeds detection with a minimal energy consumption of 51.8mJ per inference, enabling scalable deployment in power\-constrained agricultural environments.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2511.07990v1)

---


## FPGA\-Accelerated RISC\-V ISA Extensions for Efficient Neural Network Inference on Edge Devices / 

发布日期：2025-11-10

作者：Arya Parameshwara

摘要：Edge AI deployment faces critical challenges balancing computational performance, energy efficiency, and resource constraints. This paper presents FPGA\-accelerated RISC\-V instruction set architecture \(ISA\) extensions for efficient neural network inference on resource\-constrained edge devices. We introduce a custom RISC\-V core with four novel ISA extensions \(FPGA.VCONV, FPGA.GEMM, FPGA.RELU, FPGA.CUSTOM\) and integrated neural network accelerators, implemented and validated on the Xilinx PYNQ\-Z2 platform. The complete system achieves 2.14x average latency speedup and 49.1% energy reduction versus an ARM Cortex\-A9 software baseline across four benchmark models \(MobileNet V2, ResNet\-18, EfficientNet Lite, YOLO Tiny\). Hardware implementation closes timing with \+12.793 ns worst negative slack at 50 MHz while using 0.43% LUTs and 11.4% BRAM for the base core and 38.8% DSPs when accelerators are active. Hardware verification confirms successful FPGA deployment with verified 64 KB BRAM memory interface and AXI interconnect functionality. All performance metrics are obtained from physical hardware measurements. This work establishes a reproducible framework for ISA\-guided FPGA acceleration that complements fixed\-function ASICs by trading peak performance for programmability.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2511.06955v1)

---


## Semantic\-Guided Natural Language and Visual Fusion for Cross\-Modal Interaction Based on Tiny Object Detection / 

发布日期：2025-11-07

作者：Xian\-Hong Huang

摘要：This paper introduces a cutting\-edge approach to cross\-modal interaction for tiny object detection by combining semantic\-guided natural language processing with advanced visual recognition backbones. The proposed method integrates the BERT language model with the CNN\-based Parallel Residual Bi\-Fusion Feature Pyramid Network \(PRB\-FPN\-Net\), incorporating innovative backbone architectures such as ELAN, MSP, and CSP to optimize feature extraction and fusion. By employing lemmatization and fine\-tuning techniques, the system aligns semantic cues from textual inputs with visual features, enhancing detection precision for small and complex objects. Experimental validation using the COCO and Objects365 datasets demonstrates that the model achieves superior performance. On the COCO2017 validation set, it attains a 52.6% average precision \(AP\), outperforming YOLO\-World significantly while maintaining half the parameter consumption of Transformer\-based models like GLIP. Several test on different of backbones such ELAN, MSP, and CSP further enable efficient handling of multi\-scale objects, ensuring scalability and robustness in resource\-constrained environments. This study underscores the potential of integrating natural language understanding with advanced backbone architectures, setting new benchmarks in object detection accuracy, efficiency, and adaptability to real\-world challenges.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2511.05474v1)

---

