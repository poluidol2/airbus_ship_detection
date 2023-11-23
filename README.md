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

You can access the dataset here - https://www.kaggle.com/c/airbus-ship-detection/data

The file data_exploration.ipynb provides a detailed exploration of the dataset.

It's important to note that this dataset is highly unbalanced, with the majority of images containing no ships. A plot illustrating the distribution of ships within images is included

![image](https://github.com/poluidol2/airbus_ship_detection/assets/112002795/2dd2ec0f-386a-425b-803c-8fd15398bde3)

Additionally, the ship area percentage within images is notably small, reaching a maximum of 4%. 
This needs to be taken into account when selecting appropriate loss functions for the model.




## Usage



## Results

Include insights or quantitative results achieved by the model. For example:
- Training/validation loss and accuracy curves.
- Evaluation metrics on the test set.
- Visualizations showing the model's predictions compared to ground truth.

## Acknowledgments

If you used any external resources, libraries, or referenced other works, acknowledge them here.

## License

Specify the license for this project if applicable.

## Contact

Provide contact information if someone wants to reach out
