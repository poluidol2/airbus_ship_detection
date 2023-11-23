# Ship Detection using U-Net with Dice+BCE Loss

This repository contains the implementation of a ship detection model using U-Net architecture along with a combined loss function of Dice Coefficient and Binary Cross-Entropy (BCE) on highly unbalanced data.

## Overview

The solution aims to detect ships in satellite or aerial images using the U-Net architecture, which is a popular choice for semantic segmentation tasks due to its ability to capture intricate details. Additionally, to handle the issue of data imbalance in the dataset, a combined loss function of Dice and BCE has been utilized.

## File Structure

- `train.py`: Py file containing model training 
- `model.py`: Py file containing model architecture 
- `data_prep.py`: Py file containing data preprocessing
- `inference.py`: Py file containing model evaluation/inference
- `loses.py`: Py file containing loses
- `data_exploration.ipyng`: Jupyter Noteboo containing data exploration
  
## Requirements

    matplotlib==3.7.3
    numpy==1.24.3
    pandas==2.0.3
    scikit_learn==1.2.2
    tensorflow==2.13.0

## Data

The dataset consists of paired images: the main image and its corresponding mask.

![image](https://github.com/poluidol2/airbus_ship_detection/assets/112002795/0908abfa-eb62-405f-8742-025cd54e8061)

You can access the dataset here - https://www.kaggle.com/c/airbus-ship-detection/data

Masks are represented with RLE-encoding and stored in train_ship_segmentations_v2.csv file. Images stored in train_v2/ and  test_v2/ folders

The file data_exploration.ipynb provides a detailed exploration of the dataset.

It's important to note that this dataset is highly unbalanced, with the majority of images containing no ships. A plot illustrating the distribution of ships within images is included

![image](https://github.com/poluidol2/airbus_ship_detection/assets/112002795/2dd2ec0f-386a-425b-803c-8fd15398bde3)

Additionally, the ship area percentage within images is notably small, reaching a maximum of 4%. 
This needs to be taken into account when selecting appropriate loss functions for the model.

## Data preprocessing

The file data_prep.py includes code for dataset preprocessing. It performs several tasks:

1. Decoding Masks from RLE to Image Format: This process involves converting masks encoded in Run-Length Encoding (RLE) to an image format. It's a crucial step for translating the encoded representation of masks back into an understandable image format for model training and evaluation.

2. Rescaling and Normalization: Images and masks are rescaled to a specific size or resolution to ensure uniformity and compatibility with the model architecture.

3. Random Flipping Augmentation: The code includes random flipping augmentation. This technique introduces variety into the dataset by horizontally or vertically flipping images randomly during training.

4. Balancing Dataset by Removing Images with Zero Ships: Additionally, the dataset is balanced by removing a portion of images that do not contain any ships. This step ensures a more balanced distribution between images with and without ships.
   
## Model

### Overview

The U-Net architecture is a widely used deep learning model for biomedical image segmentation and has been adapted for various computer vision tasks due to its ability to capture fine-grained details and features. This implementation specifically targets ship detection within images.

### Architecture Details

#### Encoder Utilities
- **Conv2D Block**: This function adds two convolutional layers with specified parameters.
- **Encoder Block**: It combines two convolutional blocks and performs downsampling via max-pooling with dropout.
- **Encoder Function**: Defines the downsampling path by stacking multiple encoder blocks.

#### Bottleneck
- **Bottleneck Function**: Defines the bottleneck convolutions, extracting additional features before the upsampling layers.

#### Decoder Utilities
- **Decoder Block**: This function defines a single decoder block of the U-Net, including deconvolution (transpose convolution) and concatenation with features from an encoder block.
- **Decoder Function**: Chains together four decoder blocks, progressively upsampling and merging features from the encoder blocks.

#### U-Net Model
- **Model Definition**: Connects the encoder, bottleneck, and decoder to create the U-Net architecture.
- **Input Shape**: Specifies the input shape for the model.
- **Output Channels**: Defines the number of output channels (classes) for the pixel-wise label map. We use 1 channel because we have only one class for segmentation.

## Loses


In the context of highly unbalanced data, such as in ship detection tasks within images, employing appropriate loss functions is crucial to ensure effective training and model performance. The following loss functions are selected specifically for addressing the challenges posed by imbalanced datasets in semantic segmentation tasks.

### Dice Coefficient Loss

- **Definition**: The Dice coefficient is a statistical measure used to gauge the similarity between two samples. In the context of semantic segmentation, the Dice coefficient loss quantifies the overlap between predicted and ground truth segmentation masks.
- **Advantages for Unbalanced Data**: The Dice coefficient loss is effective for imbalanced datasets as it emphasizes the overlap or intersection between predicted and true positive regions, mitigating the impact of the majority class dominating the loss.

### Binary Cross-Entropy (BCE) Loss

- **Definition**: Binary Cross-Entropy loss, also known as Log Loss, measures the dissimilarity between predicted and actual pixel-wise classifications in binary segmentation tasks.
- **Advantages for Unbalanced Data**: BCE loss penalizes misclassifications by assigning higher weights to incorrectly classified pixels, thereby helping the model focus on accurately classifying both positive and negative classes.

### Combined Dice+BCE Loss

- **Usage Rationale**: The combined use of Dice and BCE losses leverages the strengths of both metrics. The Dice loss focuses on capturing fine-grained details and mitigating the impact of class imbalance, while BCE loss emphasizes accurate classification in pixel-wise segmentation tasks.
- **Advantages**: This combined loss strategy aims to strike a balance between capturing intricate details (Dice) and ensuring accurate classification (BCE), which is particularly beneficial when dealing with highly unbalanced datasets like ship detection.

### Considerations for Sparse Ship Area Percentage
- **Impact on Training**: Datasets with a notably small ship area percentage pose challenges for model training due to limited positive instances, which might not be adequately captured by conventional loss functions.
- **Addressing Sparse Instances**: The chosen combination of Dice and BCE losses aims to mitigate this challenge by encouraging the model to effectively learn from sparse positive instances and their pixel-wise distributions.


## Results

## Acknowledgments



## License


## Contact

