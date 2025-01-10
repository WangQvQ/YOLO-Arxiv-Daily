# 每日从arXiv中获取最新YOLO相关论文


## Video Summarisation with Incident and Context Information using Generative AI / 

发布日期：2025-01-08

作者：Ulindu De Silva

摘要：The proliferation of video content production has led to vast amounts of data, posing substantial challenges in terms of analysis efficiency and resource utilization. Addressing this issue calls for the development of robust video analysis tools. This paper proposes a novel approach leveraging Generative Artificial Intelligence \(GenAI\) to facilitate streamlined video analysis. Our tool aims to deliver tailored textual summaries of user\-defined queries, offering a focused insight amidst extensive video datasets. Unlike conventional frameworks that offer generic summaries or limited action recognition, our method harnesses the power of GenAI to distil relevant information, enhancing analysis precision and efficiency. Employing YOLO\-V8 for object detection and Gemini for comprehensive video and text analysis, our solution achieves heightened contextual accuracy. By combining YOLO with Gemini, our approach furnishes textual summaries extracted from extensive CCTV footage, enabling users to swiftly navigate and verify pertinent events without the need for exhaustive manual review. The quantitative evaluation revealed a similarity of 72.8%, while the qualitative assessment rated an accuracy of 85%, demonstrating the capability of the proposed method.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2501.04764v1)

---


## Efficient License Plate Recognition in Videos Using Visual Rhythm and Accumulative Line Analysis / 

发布日期：2025-01-08

作者：Victor Nascimento Ribeiro

摘要：Video\-based Automatic License Plate Recognition \(ALPR\) involves extracting vehicle license plate text information from video captures. Traditional systems typically rely heavily on high\-end computing resources and utilize multiple frames to recognize license plates, leading to increased computational overhead. In this paper, we propose two methods capable of efficiently extracting exactly one frame per vehicle and recognizing its license plate characters from this single image, thus significantly reducing computational demands. The first method uses Visual Rhythm \(VR\) to generate time\-spatial images from videos, while the second employs Accumulative Line Analysis \(ALA\), a novel algorithm based on single\-line video processing for real\-time operation. Both methods leverage YOLO for license plate detection within the frame and a Convolutional Neural Network \(CNN\) for Optical Character Recognition \(OCR\) to extract textual information. Experiments on real videos demonstrate that the proposed methods achieve results comparable to traditional frame\-by\-frame approaches, with processing speeds three times faster.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2501.04750v1)

---


## Combining YOLO and Visual Rhythm for Vehicle Counting / 

发布日期：2025-01-08

作者：Victor Nascimento Ribeiro

摘要：Video\-based vehicle detection and counting play a critical role in managing transport infrastructure. Traditional image\-based counting methods usually involve two main steps: initial detection and subsequent tracking, which are applied to all video frames, leading to a significant increase in computational complexity. To address this issue, this work presents an alternative and more efficient method for vehicle detection and counting. The proposed approach eliminates the need for a tracking step and focuses solely on detecting vehicles in key video frames, thereby increasing its efficiency. To achieve this, we developed a system that combines YOLO, for vehicle detection, with Visual Rhythm, a way to create time\-spatial images that allows us to focus on frames that contain useful information. Additionally, this method can be used for counting in any application involving unidirectional moving targets to be detected and identified. Experimental analysis using real videos shows that the proposed method achieves mean counting accuracy around 99.15% over a set of videos, with a processing speed three times faster than tracking based approaches.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2501.04534v1)

---


## SCC\-YOLO: An Improved Object Detector for Assisting in Brain Tumor Diagnosis / 

发布日期：2025-01-07

作者：Runci Bai

摘要：Brain tumors can result in neurological dysfunction, alterations in cognitive and psychological states, increased intracranial pressure, and the occurrence of seizures, thereby presenting a substantial risk to human life and health. The You Only Look Once\(YOLO\) series models have demonstrated superior accuracy in object detection for medical imaging. In this paper, we develop a novel SCC\-YOLO architecture by integrating the SCConv attention mechanism into YOLOv9. The SCConv module reconstructs an efficient convolutional module by reducing spatial and channel redundancy among features, thereby enhancing the learning of image features. We investigate the impact of intergrating different attention mechanisms with the YOLOv9 model on brain tumor image detection using both the Br35H dataset and our self\-made dataset\(Brain\_Tumor\_Dataset\). Experimental results show that on the Br35H dataset, SCC\-YOLO achieved a 0.3% improvement in mAp50 compared to YOLOv9, while on our self\-made dataset, SCC\-YOLO exhibited a 0.5% improvement over YOLOv9. SCC\-YOLO has reached state\-of\-the\-art performance in brain tumor detection. Source code is available at : https://jihulab.com/healthcare\-information\-studio/SCC\-YOLO/\-/tree/master

中文摘要：


代码链接：https://jihulab.com/healthcare-information-studio/SCC-YOLO/-/tree/master

论文链接：[阅读更多](http://arxiv.org/abs/2501.03836v1)

---


## Identifying Surgical Instruments in Pedagogical Cataract Surgery Videos through an Optimized Aggregation Network / 

发布日期：2025-01-05

作者：Sanya Sinha

摘要：Instructional cataract surgery videos are crucial for ophthalmologists and trainees to observe surgical details repeatedly. This paper presents a deep learning model for real\-time identification of surgical instruments in these videos, using a custom dataset scraped from open\-access sources. Inspired by the architecture of YOLOV9, the model employs a Programmable Gradient Information \(PGI\) mechanism and a novel Generally\-Optimized Efficient Layer Aggregation Network \(Go\-ELAN\) to address the information bottleneck problem, enhancing Minimum Average Precision \(mAP\) at higher Non\-Maximum Suppression Intersection over Union \(NMS IoU\) scores. The Go\-ELAN YOLOV9 model, evaluated against YOLO v5, v7, v8, v9 vanilla, Laptool and DETR, achieves a superior mAP of 73.74 at IoU 0.5 on a dataset of 615 images with 10 instrument classes, demonstrating the effectiveness of the proposed model.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2501.02618v1)

---

