# Oil Spill Detection

Trained a <strong> ResNet-18 convolutional neural network (CNN) </strong> to detect marine oil spills, implementing data augmentation techniques with TensorFlow.
## Problem Statement <!--- do not change this line -->

Oil spills have continued to cause severe and lasting damage to the environment. For reference, in the Deepwater Horizon oil spill (the largest oil spill in history) **4.9 million barrels** (210 million gallons) of oil were released in the Gulf of Mexico.

This disaster alone is estimated to have killed:
- Up to 5 trillion fish
 - 84,000 birds
 - At least 59,000 sea turtles <br>
_(Source: National Oceanic and Atmospheric Administration)_

We believe <strong> computer vision technology </strong> can help mitigate wildlife damage by identifying oil spills in water body images, while also supporting tourism, shipping, and the fishing industry.

## Key Results <!--- do not change this line -->

1. Used transfer learning to train a ResNet-18 CNN on a 300 image dataset to classify oil spill imagery.
2. Applied image transformations (augmentation) to create varied training samples from existing images.
3. Achieved ~86% validation accuracy.

## Model Predicting User Uploaded Images


<img width="70%" height="70%" alt="ResNet-18 model predicted no oil spill correctly while testing" src="https://github.com/user-attachments/assets/0d9f60fa-6b80-44bf-8b29-c41c578f019d" />

> ResNet-18 model predicting **No Oil Spill** correctly.

<br>

<img width="70%" height="70%" alt="ResNet-18 model predicted an oil spill correctly while testing" src="https://github.com/user-attachments/assets/30477691-a0a9-45a0-9456-1e43c18bc241" />

> ResNet-18 model predicting **Oil Spill** correctly.

<br>

<img width="512" height="425" alt="image" src="https://github.com/user-attachments/assets/945e077f-7018-4c2f-8c02-46295c438d33" />

> Shows correct classification of most **Oil Spill** and **No Oil Spill** images after applying transfer learning and fine-tuning.



## Methodologies <!--- do not change this line -->

To accomplish this, we utilized the ResNet-18 model to classify images of water bodies with supervised learning. Techniques that were incorporated:

- **Transfer Learning**: Pretrained on ImageNet, replacing the final layer for binary classification.
- **Image Normalization**: Improved feature extraction and pattern recognition.
- **Image Augmentation**: Increased dataset variety via flips, color changes, and resizing.
- **Freezing Layers**: Preserved pretrained knowledge while fine-tuning top layers.
- **Dropout Regularization**: Reduced overfitting.
- **Early Stopping**: Prevented unnecessary training once performance plateaued.
- **Differential Learning Rates**: Higher LR for new layers, lower for pretrained ones.
- **Learning Rate Scheduler**: Adjusted LR during training for stability.

## Data Sources <!--- do not change this line -->

[Marine Oil Spill Detection Dataset](https://www.kaggle.com/datasets/afzalofficial/marine-oil-spill-detection) <br>
<br>
[NOAA statistics on Deepwater Horizon](https://oceanservice.noaa.gov/education/tutorial-coastal/oil-spills/os04.html?utm_source=chatgpt.com)


## Technologies Used <!--- do not change this line -->

- *Python*
- *pandas*
- *NumPy*
- *PyTorch*
- *Matplotlib*

## Authors <!--- do not change this line -->
*This project was completed in collaboration with:*
- *Jason Qin ([jq2406@nyu.edu](mailto:jq2406@nyu.edu))*
- *Chelsea Nguyen ([chelsea.nguyen001@umb.edu](mailto:chelsea.nguyen001@umb.edu))*
- *Xavier Rush ([xcrush@aggies.ncat.edu](mailto:xcrush@aggies.ncat.edu))*
