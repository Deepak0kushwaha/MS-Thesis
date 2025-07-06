# MS-Thesis
Blockages in centrifugal pumps reduce performance and risk breakdown, so timely detection is essential. This study uses 2D CWT of suction pressure signals and a modified InceptionV3 CNN (ReLUs + ELUs) to classify 35 conditions, raising accuracy from 82% to 95%. Enhanced precision, recall, and F1 underscore its value for predictive maintenance.

# Abstract
Blockages in a centrifugal pump reduce the performance of the pump and can result in operational breakdown if not identified on time. Thus, it is crucial to identify blockages in the pump to ensure its reliable working. This study presents a modified InceptionV3 Deep Convolutional Neural Network (CNN) model to classify blockage faults and their severity in centrifugal pumps based on Suction pressure signal, highlighting its importance in predictive analytics. Initially, the methodology involves plotting two-dimensional Continuous Wavelet Transform (CWT) plots of pressure signals acquired from the pump test rig. The InceptionV3 model is employed to classify blockages in the pump. Validation and test accuracies of InceptionV3 are achieved as 81.71 % and 81.82%, respectively at 2000 RPM. Afterward, the blockage fault diagnosis capabilities of InceptionV3 are improved by integrating Exponential Linear Units (ELUs) with Rectified Linear Units (ReLUs) in the model’s architecture. It is found that validation and test accuracies increased to 95.48 % and 95.36 %, respectively, in modified InceptionV3 on 35 classes (one healthy and 34 faulty conditions) at 2000RPM. The robustness of the modified InceptionV3 model is highlighted by comparing test accuracy, precision, recall, and F1 score to the original InceptionV3 model. The proposed methodology demonstrates the modified InceptionV3 model’s efficacy in classifying various blockage conditions, making it a potent tool for predictive maintenance in industrial settings.

# Architectural Modifications
The architectural modifications are shown in my published research paper is shown below.

![image](https://github.com/user-attachments/assets/b372e79b-560a-46d9-b77d-1346607859a1)



