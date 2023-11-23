import tensorflow as tf
from model import unet  # Import your model creation function
from data_prep import train_df_prep, train_val_dataset_prep  # Import functions to get train and validation data
from loses import dice_coeff, combined_loss


def main():

    # Set up hyperparameters and configurations

    SEGMENTATION_PATH = ''  # Path to segmentation csv
    TRAIN_IMAGES_PATH = ''  # Path to train images folder
    batch_size = 256
    epochs = 10
    image_shape = (768, 768)  # Image shape
    model_input_shape = (768, 768)  # Model input shape (rescaled image shape)
    learning_rate = 0.001


    # Get the train and validation datasets
    train_df = train_df_prep(SEGMENTATION_PATH)
    train_data, validation_data = train_val_dataset_prep(train_df, image_shape,
                                                         model_input_shape, TRAIN_IMAGES_PATH, batch_size)


    # Create the model
    model = unet(model_input_shape)
    iou_metric = tf.keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5)

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss=combined_loss,
                  metrics=[dice_coeff, 'binary_accuracy', iou_metric])

    # Train the model
    model.fit(train_data,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=validation_data)

    # Save the trained model
    model.save('saved_models/trained_model.h5')  # Save the trained model to a file


if __name__ == "__main__":
    main()
