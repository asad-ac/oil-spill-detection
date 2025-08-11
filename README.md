# Oil Spill Detection

Trained a ResNet-18 convolutional neural network (CNN) to detect marine oil spills, implementing data augmentation techniques with TensorFlow.
## Problem Statement <!--- do not change this line -->

Oil spills have continued to cause severe and lasting damage to the environment. For reference, in the Deepwater Horizon oil spill (the largest oil spill in history) 4.9 million barrels (210 million gallons) of oil were released in the Gulf of Mexico. This oil spill alone is estimated to have killed up to 5 trillion fish, 84 thousand birds, and at least 59 thousand sea turtles (National Oceanic and Atmospheric Administration). We believe computer vision technology can help mitigate wildlife damage by identifying oil spills in water body images, while also supporting tourism, shipping, and the fishing industry.

## Key Results <!--- do not change this line -->

1. Used transfer learning to train a ResNet-18 CNN on a 300 image dataset to classify oil spill imagery.
2. Applied image transformations (augmentation) to create varied training samples from existing images.
3. Achieved ~86% validation accuracy after training.

## Model Predicting User Uploaded Images

<img width="65%" height="65%" alt="ResNet-18 Model Predicted no oil spill correctly while testing" src="https://github.com/user-attachments/assets/0d9f60fa-6b80-44bf-8b29-c41c578f019d" />

#

<img width="65%" height="65%" alt="ResNet-18 Model Predicted an oil spill correctly while testing" src="https://github.com/user-attachments/assets/30477691-a0a9-45a0-9456-1e43c18bc241" />

## Methodologies <!--- do not change this line -->

To accomplish this, we utilized the ResNet-18 model to classify images of water bodies with supervised learning. We designed our model to use image normalization to allow the model to identify patterns more easily. Other techniques we incorporated:
- Image augmentation
- Freezing layers
- Early stopping
- Dropout regularization
- Differential learning rates: higher LR for new layers, lower for pre-trained
- Learning rate scheduler

## Data Sources <!--- do not change this line -->

*Kaggle Datasets: [Marine Oil Spill Detection](https://www.kaggle.com/datasets/afzalofficial/marine-oil-spill-detection)*

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
