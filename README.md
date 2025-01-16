# 每日从arXiv中获取最新YOLO相关论文


## Detecting Wildfire Flame and Smoke through Edge Computing using Transfer Learning Enhanced Deep Learning Models / 

发布日期：2025-01-15

作者：Giovanny Vazquez

摘要：Autonomous unmanned aerial vehicles \(UAVs\) integrated with edge computing capabilities empower real\-time data processing directly on the device, dramatically reducing latency in critical scenarios such as wildfire detection. This study underscores Transfer Learning's \(TL\) significance in boosting the performance of object detectors for identifying wildfire smoke and flames, especially when trained on limited datasets, and investigates the impact TL has on edge computing metrics. With the latter focusing how TL\-enhanced You Only Look Once \(YOLO\) models perform in terms of inference time, power usage, and energy consumption when using edge computing devices. This study utilizes the Aerial Fire and Smoke Essential \(AFSE\) dataset as the target, with the Flame and Smoke Detection Dataset \(FASDD\) and the Microsoft Common Objects in Context \(COCO\) dataset serving as source datasets. We explore a two\-stage cascaded TL method, utilizing D\-Fire or FASDD as initial stage target datasets and AFSE as the subsequent stage. Through fine\-tuning, TL significantly enhances detection precision, achieving up to 79.2% mean Average Precision \(mAP@0.5\), reduces training time, and increases model generalizability across the AFSE dataset. However, cascaded TL yielded no notable improvements and TL alone did not benefit the edge computing metrics evaluated. Lastly, this work found that YOLOv5n remains a powerful model when lacking hardware acceleration, finding that YOLOv5n can process images nearly twice as fast as its newer counterpart, YOLO11n. Overall, the results affirm TL's role in augmenting the accuracy of object detectors while also illustrating that additional enhancements are needed to improve edge computing performance.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2501.08639v1)

---


## Benchmarking YOLOv8 for Optimal Crack Detection in Civil Infrastructure / 

发布日期：2025-01-12

作者：Woubishet Zewdu Taffese

摘要：Ensuring the structural integrity and safety of bridges is crucial for the reliability of transportation networks and public safety. Traditional crack detection methods are increasingly being supplemented or replaced by advanced artificial intelligence \(AI\) techniques. However, most of the models rely on two\-stage target detection algorithms, which pose concerns for real\-time applications due to their lower speed. While models such as YOLO \(You Only Look Once\) have emerged as transformative tools due to their remarkable speed and accuracy. However, the potential of the latest YOLOv8 framework in this domain remains underexplored. This study bridges that gap by rigorously evaluating YOLOv8's performance across five model scales \(nano, small, medium, large, and extra\-large\) using a high\-quality Roboflow dataset. A comprehensive hyperparameter optimization was performed, testing six state\-of\-the\-art optimizers\-Stochastic Gradient Descent, Adaptive Moment Estimation, Adam with Decoupled Weight Decay, Root Mean Square Propagation, Rectified Adam, and Nesterov\-accelerated Adam. Results revealed that YOLOv8, optimized with Stochastic Gradient Descent, delivered exceptional accuracy and speed, setting a new benchmark for real\-time crack detection. Beyond its immediate application, this research positions YOLOv8 as a foundational approach for integrating advanced computer vision techniques into infrastructure monitoring. By enabling more reliable and proactive maintenance of aging bridge networks, this work paves the way for safer, more efficient transportation systems worldwide.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2501.06922v1)

---


## YO\-CSA\-T: A Real\-time Badminton Tracking System Utilizing YOLO Based on Contextual and Spatial Attention / 

发布日期：2025-01-11

作者：Yuan Lai

摘要：The 3D trajectory of a shuttlecock required for a badminton rally robot for human\-robot competition demands real\-time performance with high accuracy. However, the fast flight speed of the shuttlecock, along with various visual effects, and its tendency to blend with environmental elements, such as court lines and lighting, present challenges for rapid and accurate 2D detection. In this paper, we first propose the YO\-CSA detection network, which optimizes and reconfigures the YOLOv8s model's backbone, neck, and head by incorporating contextual and spatial attention mechanisms to enhance model's ability in extracting and integrating both global and local features. Next, we integrate three major subtasks, detection, prediction, and compensation, into a real\-time 3D shuttlecock trajectory detection system. Specifically, our system maps the 2D coordinate sequence extracted by YO\-CSA into 3D space using stereo vision, then predicts the future 3D coordinates based on historical information, and re\-projects them onto the left and right views to update the position constraints for 2D detection. Additionally, our system includes a compensation module to fill in missing intermediate frames, ensuring a more complete trajectory. We conduct extensive experiments on our own dataset to evaluate both YO\-CSA's performance and system effectiveness. Experimental results show that YO\-CSA achieves a high accuracy of 90.43% mAP@0.75, surpassing both YOLOv8s and YOLO11s. Our system performs excellently, maintaining a speed of over 130 fps across 12 test sequences.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2501.06472v1)

---


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

