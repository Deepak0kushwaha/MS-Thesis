# MS-Thesis
Blockages in centrifugal pumps reduce performance and risk breakdown, so timely detection is essential. This study uses 2D CWT of suction pressure signals and a modified InceptionV3 CNN (ReLUs + ELUs) to classify 35 conditions, raising accuracy from 82% to 95%. Enhanced precision, recall, and F1 underscore its value for predictive maintenance.

# Abstract
Blockages in a centrifugal pump reduce the performance of the pump and can result in operational breakdown if not identified on time. Thus, it is crucial to identify blockages in the pump to ensure its reliable working. This study presents a modified InceptionV3 Deep Convolutional Neural Network (CNN) model to classify blockage faults and their severity in centrifugal pumps based on Suction pressure signal, highlighting its importance in predictive analytics. Initially, the methodology involves plotting two-dimensional Continuous Wavelet Transform (CWT) plots of pressure signals acquired from the pump test rig. The InceptionV3 model is employed to classify blockages in the pump. Validation and test accuracies of InceptionV3 are achieved as 81.71 % and 81.82%, respectively at 2000 RPM. Afterward, the blockage fault diagnosis capabilities of InceptionV3 are improved by integrating Exponential Linear Units (ELUs) with Rectified Linear Units (ReLUs) in the model’s architecture. It is found that validation and test accuracies increased to 95.48 % and 95.36 %, respectively, in modified InceptionV3 on 35 classes (one healthy and 34 faulty conditions) at 2000RPM. The robustness of the modified InceptionV3 model is highlighted by comparing test accuracy, precision, recall, and F1 score to the original InceptionV3 model. The proposed methodology demonstrates the modified InceptionV3 model’s efficacy in classifying various blockage conditions, making it a potent tool for predictive maintenance in industrial settings.

# Blockage faults classification Methodology

![image](https://github.com/user-attachments/assets/e9305d49-d0f1-462e-b421-e7309dc8fe1a)


# Architectural Modifications
The architectural modifications in my published research paper is shown below.

![image](https://github.com/user-attachments/assets/b372e79b-560a-46d9-b77d-1346607859a1)


#Results and Discussions

![image](https://github.com/user-attachments/assets/f631c198-2cfb-4f15-bacd-64b3451edb04)

![image](https://github.com/user-attachments/assets/4a6d7200-eb6f-42fe-b6bf-676641a03e8d)

![image](https://github.com/user-attachments/assets/c55aed27-084d-4d5b-83f7-ed3207d535e2)


#Full article
Deepak Kumar, Nagendra Singh Ranawat, Pavan Kumar Kankar, Ankur Miglani, InceptionV3 based blockage fault diagnosis of centrifugal pump, Advanced Engineering Informatics, Volume 65, Part A, 2025, 103181, ISSN 1474-0346,
https://doi.org/10.1016/j.aei.2025.103181. (https://www.sciencedirect.com/science/article/pii/S1474034625000746)
