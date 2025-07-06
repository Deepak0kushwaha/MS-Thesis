# MS-Thesis
Blockages in centrifugal pumps reduce performance and risk breakdown, so timely detection is essential. This study uses 2D CWT of suction pressure signals and a modified InceptionV3 CNN (ReLUs + ELUs) to classify 35 conditions, raising accuracy from 82% to 95%. Enhanced precision, recall, and F1 underscore its value for predictive maintenance.

# Abstract
Blockages in a centrifugal pump reduce the performance of the pump and can result in operational breakdown if not identified on time. Thus, it is crucial to identify blockages in the pump to ensure its reliable working. This study presents a modified InceptionV3 Deep Convolutional Neural Network (CNN) model to classify blockage faults and their severity in centrifugal pumps based on Suction pressure signal, highlighting its importance in predictive analytics. Initially, the methodology involves plotting two-dimensional Continuous Wavelet Transform (CWT) plots of pressure signals acquired from the pump test rig. The InceptionV3 model is employed to classify blockages in the pump. Validation and test accuracies of InceptionV3 are achieved as 81.71 % and 81.82%, respectively at 2000 RPM. Afterward, the blockage fault diagnosis capabilities of InceptionV3 are improved by integrating Exponential Linear Units (ELUs) with Rectified Linear Units (ReLUs) in the model’s architecture. It is found that validation and test accuracies increased to 95.48 % and 95.36 %, respectively, in modified InceptionV3 on 35 classes (one healthy and 34 faulty conditions) at 2000RPM. The robustness of the modified InceptionV3 model is highlighted by comparing test accuracy, precision, recall, and F1 score to the original InceptionV3 model. The proposed methodology demonstrates the modified InceptionV3 model’s efficacy in classifying various blockage conditions, making it a potent tool for predictive maintenance in industrial settings.

# Blockage faults classification Methodology

![image](https://github.com/user-attachments/assets/e9305d49-d0f1-462e-b421-e7309dc8fe1a)

# CWT

The CWT is a critical signal processing tool, which is very good at capturing the different frequency components of a non-stationary signal. These wavelets are instrumental as they are localized in both the time and frequency domains.

![image](https://github.com/user-attachments/assets/5acb0ffe-988b-4c68-84c1-90ae7dea254e)

![image](https://github.com/user-attachments/assets/d844fc8a-6a39-4105-9450-bd40d90ad473)

![image](https://github.com/user-attachments/assets/b19081ab-ee00-4fd3-82df-df40a96cc1da)
![image](https://github.com/user-attachments/assets/f5f8388c-dd3a-4abc-bd22-5a6c2bf35713)
![image](https://github.com/user-attachments/assets/f8afe77b-e22b-4d7c-a597-f15496de2b50)

![image](https://github.com/user-attachments/assets/f1c58471-70b5-4e27-a8cf-e7cd2222e0cc)
![image](https://github.com/user-attachments/assets/769da8a3-9e4e-4519-939c-9ddf58024566)
![image](https://github.com/user-attachments/assets/2c616a92-253d-4cee-aa9d-03b9b9661d22)

![image](https://github.com/user-attachments/assets/24cd2dc2-bf47-473c-8965-4fe12276aaf3)
![image](https://github.com/user-attachments/assets/a49caaa5-88ce-4d31-8c7c-c25cc3ff83d9)
![image](https://github.com/user-attachments/assets/01bb0464-da9a-44c4-baab-318a5fefa7a9)




# Architectural Modifications
The architectural modifications in my published research paper is shown below.

![image](https://github.com/user-attachments/assets/b372e79b-560a-46d9-b77d-1346607859a1)




# Results and Discussions

![image](https://github.com/user-attachments/assets/683f8bd8-0cde-42e0-a291-62ddac38da7a)

![image](https://github.com/user-attachments/assets/7c6b6d6e-b331-4d48-9669-bdc751a2d766)

![image](https://github.com/user-attachments/assets/4a1a0ddd-5fce-4ff7-a2bc-51630c87d80b)


# Full article
Deepak Kumar, Nagendra Singh Ranawat, Pavan Kumar Kankar, Ankur Miglani, InceptionV3 based blockage fault diagnosis of centrifugal pump, Advanced Engineering Informatics, Volume 65, Part A, 2025, 103181, ISSN 1474-0346,
https://doi.org/10.1016/j.aei.2025.103181. (https://www.sciencedirect.com/science/article/pii/S1474034625000746)
