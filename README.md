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
- **Output Channels**: Defines the number of output channels (classes) for the pixel-wise label map.



## Loses

## Results

## Acknowledgments



## License


## Contact

