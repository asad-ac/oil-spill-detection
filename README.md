# Oil Spill Detection

Trained a <strong> ResNet-18 convolutional neural network (CNN) </strong> to detect marine oil spills, implementing data augmentation techniques with TensorFlow.
## Problem Statement <!--- do not change this line -->

Oil spills have continued to cause severe and lasting damage to the environment. For reference, in the Deepwater Horizon oil spill (the largest oil spill in history) **4.9 million barrels** (210 million gallons) of oil were released in the Gulf of Mexico.

This disaster alone is estimated to have killed:
- Up to **5 trillion** fish
- **84,000** birds
- At least **59,000** sea turtles <br>
_(Source: National Oceanic and Atmospheric Administration)_

We believe <strong> computer vision technology </strong> can help mitigate wildlife damage by identifying oil spills in water body images, while also supporting tourism, shipping, and the fishing industry.

## Key Results <!--- do not change this line -->

1. Trained a ResNet-18 CNN using transfer learning on a **300-image marine oil spill dataset**. 
2. Enhanced dataset diversity by applying image augmentation techniques (flips, color adjustments, resizing) to generate varied training samples from existing images.
3. Achieved ~**86% validation accuracy**, correctly classifying most oil spill and non-oil spill images.

## Model Predicting User Uploaded Images

<img width="70%" height="70%" alt="ResNet-18 model predicted no oil spill correctly while testing" src="https://github.com/user-attachments/assets/0d9f60fa-6b80-44bf-8b29-c41c578f019d" />

> **Prediction:** No Oil Spill ✅ 

<br>

<img width="70%" height="70%" alt="ResNet-18 model predicted an oil spill correctly while testing" src="https://github.com/user-attachments/assets/30477691-a0a9-45a0-9456-1e43c18bc241" />

> **Prediction:** Oil Spill ✅ 

<br>

## Confusion Matrix

<img width="512" height="425" alt="image" src="https://github.com/user-attachments/assets/945e077f-7018-4c2f-8c02-46295c438d33" />

> **Confusion Matrix — ResNet-18 with Transfer Learning:** Correctly classified most oil spill and no oil spill images (~86% validation accuracy).

## Classification Report

<img width="1056" height="384" alt="Precision, recall, and F1-score for Oil Spill and No Oil Spill classes; achieved 82.22% test accuracy and +32.22% improvement over baseline" src="https://github.com/user-attachments/assets/e5296428-1b7a-42ce-8d79-12f39c47e5c2" />

> **Classification Report — ResNet-18 with Transfer Learning:** Precision, recall, and F1-score for both classes; achieved 82.22% test accuracy and +32.22% improvement over baseline.


## Methodologies <!--- do not change this line -->

To accomplish this, we utilized the ResNet-18 model to classify images of water bodies with supervised learning. 

Techniques that were incorporated:

- **Transfer Learning**: Pretrained on ImageNet, replacing the final layer for binary classification.
- **Image Normalization**: Scaled RGB values between 0 and 1, then applied ImageNet mean and standard deviation for each channel to match ResNet-18’s pretrained input distribution.
- **Image Augmentation**: Increased dataset variety via flips, color changes, and resizing.
- **Freezing Layers**: Preserved pretrained knowledge while fine-tuning top layers.
- **Dropout Regularization**: Reduced overfitting.
- **Early Stopping**: Prevented unnecessary training once performance plateaued.
- **Differential Learning Rates**: Higher LR for new layers, lower for pretrained ones.
- **Learning Rate Scheduler**: Adjusted LR during training for stability.

## Limitations & Future Work
- **Dataset Size & Diversity**: Limited image diversity (mostly ocean scenes) could bias predictions.
- **Class Imbalance**: Some oil spill images contain text overlays, which may influence results.
- **Future Improvements**:
  - Expand dataset with varied water body types and conditions.
  - Revisit ResNet-50 to improve accuracy.
  - Deploy as a web app for real-time oil spill detection.

## Data Sources <!--- do not change this line -->

[Marine Oil Spill Detection Dataset](https://www.kaggle.com/datasets/afzalofficial/marine-oil-spill-detection)
<br>
[NOAA statistics on Deepwater Horizon](https://oceanservice.noaa.gov/education/tutorial-coastal/oil-spills/os04.html)

## Technologies Used <!--- do not change this line -->

- *Python*
- *Pandas*
- *NumPy*
- *PyTorch*
- *Matplotlib*

## Authors <!--- do not change this line -->
*This project was completed in collaboration with:*
- *Jason Qin ([jq2406@nyu.edu](mailto:jq2406@nyu.edu))*
- *Chelsea Nguyen ([chelsea.nguyen001@umb.edu](mailto:chelsea.nguyen001@umb.edu))*
- *Xavier Rush ([xcrush@aggies.ncat.edu](mailto:xcrush@aggies.ncat.edu))*
