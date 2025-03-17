# 每日从arXiv中获取最新YOLO相关论文


## Comparative Analysis of Advanced AI\-based Object Detection Models for Pavement Marking Quality Assessment during Daytime / 

发布日期：2025-03-14

作者：Gian Antariksa

摘要：Visual object detection utilizing deep learning plays a vital role in computer vision and has extensive applications in transportation engineering. This paper focuses on detecting pavement marking quality during daytime using the You Only Look Once \(YOLO\) model, leveraging its advanced architectural features to enhance road safety through precise and real\-time assessments. Utilizing image data from New Jersey, this study employed three YOLOv8 variants: YOLOv8m, YOLOv8n, and YOLOv8x. The models were evaluated based on their prediction accuracy for classifying pavement markings into good, moderate, and poor visibility categories. The results demonstrated that YOLOv8n provides the best balance between accuracy and computational efficiency, achieving the highest mean Average Precision \(mAP\) for objects with good visibility and demonstrating robust performance across various Intersections over Union \(IoU\) thresholds. This research enhances transportation safety by offering an automated and accurate method for evaluating the quality of pavement markings.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2503.11008v1)

---


## Object detection characteristics in a learning factory environment using YOLOv8 / 

发布日期：2025-03-13

作者：Toni Schneidereit

摘要：AI\-based object detection, and efforts to explain and investigate their characteristics, is a topic of high interest. The impact of, e.g., complex background structures with similar appearances as the objects of interest, on the detection accuracy and, beforehand, the necessary dataset composition are topics of ongoing research. In this paper, we present a systematic investigation of background influences and different features of the object to be detected. The latter includes various materials and surfaces, partially transparent and with shiny reflections in the context of an Industry 4.0 learning factory. Different YOLOv8 models have been trained for each of the materials on different sized datasets, where the appearance was the only changing parameter. In the end, similar characteristics tend to show different behaviours and sometimes unexpected results. While some background components tend to be detected, others with the same features are not part of the detection. Additionally, some more precise conclusions can be drawn from the results. Therefore, we contribute a challenging dataset with detailed investigations on 92 trained YOLO models, addressing some issues on the detection accuracy and possible overfitting.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2503.10356v1)

---


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

