# 每日从arXiv中获取最新YOLO相关论文


## YOLOE: Real\-Time Seeing Anything / 

发布日期：2025-03-10

作者：Ao Wang

摘要：Object detection and segmentation are widely employed in computer vision applications, yet conventional models like YOLO series, while efficient and accurate, are limited by predefined categories, hindering adaptability in open scenarios. Recent open\-set methods leverage text prompts, visual cues, or prompt\-free paradigm to overcome this, but often compromise between performance and efficiency due to high computational demands or deployment complexity. In this work, we introduce YOLOE, which integrates detection and segmentation across diverse open prompt mechanisms within a single highly efficient model, achieving real\-time seeing anything. For text prompts, we propose Re\-parameterizable Region\-Text Alignment \(RepRTA\) strategy. It refines pretrained textual embeddings via a re\-parameterizable lightweight auxiliary network and enhances visual\-textual alignment with zero inference and transferring overhead. For visual prompts, we present Semantic\-Activated Visual Prompt Encoder \(SAVPE\). It employs decoupled semantic and activation branches to bring improved visual embedding and accuracy with minimal complexity. For prompt\-free scenario, we introduce Lazy Region\-Prompt Contrast \(LRPC\) strategy. It utilizes a built\-in large vocabulary and specialized embedding to identify all objects, avoiding costly language model dependency. Extensive experiments show YOLOE's exceptional zero\-shot performance and transferability with high inference efficiency and low training cost. Notably, on LVIS, with 3$times$ less training cost and 1.4$times$ inference speedup, YOLOE\-v8\-S surpasses YOLO\-Worldv2\-S by 3.5 AP. When transferring to COCO, YOLOE\-v8\-L achieves 0.6 AP$^b$ and 0.4 AP$^m$ gains over closed\-set YOLOv8\-L with nearly 4$times$ less training time. Code and models are available at https://github.com/THU\-MIG/yoloe.

中文摘要：


代码链接：https://github.com/THU-MIG/yoloe.

论文链接：[阅读更多](http://arxiv.org/abs/2503.07465v1)

---


## HGO\-YOLO: Advancing Anomaly Behavior Detection with Hierarchical Features and Lightweight Optimized Detection / 

发布日期：2025-03-10

作者：Qizhi Zheng

摘要：Accurate and real\-time object detection is crucial for anomaly behavior detection, especially in scenarios constrained by hardware limitations, where balancing accuracy and speed is essential for enhancing detection performance. This study proposes a model called HGO\-YOLO, which integrates the HGNetv2 architecture into YOLOv8. This combination expands the receptive field and captures a wider range of features while simplifying model complexity through GhostConv. We introduced a lightweight detection head, OptiConvDetect, which utilizes parameter sharing to construct the detection head effectively. Evaluation results show that the proposed algorithm achieves a mAP@0.5 of 87.4% and a recall rate of 81.1%, with a model size of only 4.6 MB and a frame rate of 56 FPS on the CPU. HGO\-YOLO not only improves accuracy by 3.0% but also reduces computational load by 51.69% \(from 8.9 GFLOPs to 4.3 GFLOPs\), while increasing the frame rate by a factor of 1.7. Additionally, real\-time tests were conducted on Raspberry Pi4 and NVIDIA platforms. These results indicate that the HGO\-YOLO model demonstrates superior performance in anomaly behavior detection.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2503.07371v1)

---


## Mitigating Hallucinations in YOLO\-based Object Detection Models: A Revisit to Out\-of\-Distribution Detection / 

发布日期：2025-03-10

作者：Weicheng He

摘要：Object detection systems must reliably perceive objects of interest without being overly confident to ensure safe decision\-making in dynamic environments. Filtering techniques based on out\-of\-distribution \(OoD\) detection are commonly added as an extra safeguard to filter hallucinations caused by overconfidence in novel objects. Nevertheless, evaluating YOLO\-family detectors and their filters under existing OoD benchmarks often leads to unsatisfactory performance. This paper studies the underlying reasons for performance bottlenecks and proposes a methodology to improve performance fundamentally. Our first contribution is a calibration of all existing evaluation results: Although images in existing OoD benchmark datasets are claimed not to have objects within in\-distribution \(ID\) classes \(i.e., categories defined in the training dataset\), around 13% of objects detected by the object detector are actually ID objects. Dually, the ID dataset containing OoD objects can also negatively impact the decision boundary of filters. These ultimately lead to a significantly imprecise performance estimation. Our second contribution is to consider the task of hallucination reduction as a joint pipeline of detectors and filters. By developing a methodology to carefully synthesize an OoD dataset that semantically resembles the objects to be detected, and using the crafted OoD dataset in the fine\-tuning of YOLO detectors to suppress the objectness score, we achieve a 88% reduction in overall hallucination error with a combined fine\-tuned detection and filtering system on the self\-driving benchmark BDD\-100K. Our code and dataset are available at: https://gricad\-gitlab.univ\-grenoble\-alpes.fr/dnn\-safety/m\-hood.

中文摘要：


代码链接：https://gricad-gitlab.univ-grenoble-alpes.fr/dnn-safety/m-hood.

论文链接：[阅读更多](http://arxiv.org/abs/2503.07330v1)

---


## OpenRSD: Towards Open\-prompts for Object Detection in Remote Sensing Images / 

发布日期：2025-03-08

作者：Ziyue Huang

摘要：Remote sensing object detection has made significant progress, but most studies still focus on closed\-set detection, limiting generalization across diverse datasets. Open\-vocabulary object detection \(OVD\) provides a solution by leveraging multimodal associations between text prompts and visual features. However, existing OVD methods for remote sensing \(RS\) images are constrained by small\-scale datasets and fail to address the unique challenges of remote sensing interpretation, include oriented object detection and the need for both high precision and real\-time performance in diverse scenarios. To tackle these challenges, we propose OpenRSD, a universal open\-prompt RS object detection framework. OpenRSD supports multimodal prompts and integrates multi\-task detection heads to balance accuracy and real\-time requirements. Additionally, we design a multi\-stage training pipeline to enhance the generalization of model. Evaluated on seven public datasets, OpenRSD demonstrates superior performance in oriented and horizontal bounding box detection, with real\-time inference capabilities suitable for large\-scale RS image analysis. Compared to YOLO\-World, OpenRSD exhibits an 8.7% higher average precision and achieves an inference speed of 20.8 FPS. Codes and models will be released.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2503.06146v1)

---


## Fine\-Tuning Florence2 for Enhanced Object Detection in Un\-constructed Environments: Vision\-Language Model Approach / 

发布日期：2025-03-06

作者：Soumyadeep Ro

摘要：Artificial intelligence has progressed through the development of Vision\-Language Models \(VLMs\), which integrate text and visual inputs to achieve comprehensive understanding and interaction in various contexts. Enhancing the performance of these models such as the transformer based Florence 2 on specialized tasks like object detection in complex and unstructured environments requires fine\-tuning. The goal of this paper is to improve the efficiency of the Florence 2 model in challenging environments by finetuning it. We accomplished this by experimenting with different configurations, using various GPU types \(T4, L4, A100\) and optimizers such as AdamW and SGD. We also employed a range of learning rates and LoRA \(Low Rank Adaptation\) settings. Analyzing the performance metrics, such as Mean Average Precision \(mAP\) scores,reveals that the finetuned Florence 2 models performed comparably to YOLO models, including YOLOv8, YOLOv9, and YOLOv10. This demonstrates how transformer based VLMs can be adapted for detailed object detection tasks. The paper emphasizes the capability of optimized transformer based VLMs to address specific challenges in object detection within unstructured environments, opening up promising avenues for practical applications in demanding and complex settings.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2503.04918v1)

---

