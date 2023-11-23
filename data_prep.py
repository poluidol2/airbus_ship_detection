from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import pandas as pd


# Function to decode RLE encoding into images
def get_mask(image_id, image_shape, segmentation_df):
    rows = segmentation_df[segmentation_df['ImageId'] == image_id]

    ships_count = len(rows)
    mask = np.zeros((image_shape[0] * image_shape[1]), dtype=np.uint8)

    for i in range(ships_count):
        encoded_mask = rows.iloc[i]['EncodedPixels']

        if isinstance(encoded_mask, float):  # Check for NaN values
            return mask.reshape((image_shape[0], image_shape[1], 1))
        encoded_mask = np.array(encoded_mask.split(), dtype=int)

        rle_data = np.reshape(encoded_mask, (-1, 2))

        for pixel, shift in rle_data:
            mask[pixel - 1:pixel + shift] = 255

    return mask.reshape((1, image_shape[1], image_shape[0])).T


# Preparing train df
def train_df_prep(segmentation_path):

    segmentation_df = pd.read_csv(segmentation_path)

    corrupted_images = ['6384c3e78.jpg','13703f040.jpg', '14715c06d.jpg',  '33e0ff2d5.jpg',
                        '4d4e09f2a.jpg', '877691df8.jpg', '8b909bb20.jpg', 'a8d99130e.jpg',
                        'ad55c3143.jpg', 'c8260c541.jpg', 'd6c7f17c7.jpg', 'dc3e7c901.jpg',
                        'e44dffe88.jpg', 'ef87bad36.jpg', 'f083256d8.jpg']

    corrupted_images_present = segmentation_df[segmentation_df['ImageId'].isin(corrupted_images)]
    # drop corrupted images
    segmentation_df = segmentation_df.drop(corrupted_images_present.index)

    # balance images with zero ships
    rows_with_nan = segmentation_df[segmentation_df['EncodedPixels'].isnull()]
    random_sample_indexes = rows_with_nan.sample(frac=0.90).index
    balanced_segmentation_df = segmentation_df.drop(random_sample_indexes)

    # create ShipsCount column to later use it for stratify
    balanced_segmentation_df['ShipsCount'] = balanced_segmentation_df.groupby('ImageId')['ImageId'].transform('count')
    balanced_segmentation_df.loc[balanced_segmentation_df['EncodedPixels'].isnull(), 'ShipsCount'] = 0
    train_df = balanced_segmentation_df[['ImageId', 'ShipsCount']].drop_duplicates()

    return train_df

#Creating datasets
def train_val_dataset_prep(train_df, image_shape, input_shape, train_image_path, batch_size):

    BATCH_SIZE = batch_size
    BUFFER_SIZE = 1000

    train_data, val_data = train_test_split(list(train_df['ImageId']),
                                        test_size=0.05,
                                        stratify=train_df['ShipsCount'],
                                        random_state=42)

    def train_generator():
        for image_id in train_data:
            yield load_image_train(image_id, image_shape, input_shape, train_image_path, train_df)

    def validation_generator():
        for image_id in val_data:
            yield load_image_val(image_id, image_shape, input_shape, train_image_path, train_df)

    train_dataset = tf.data.Dataset.from_generator(train_generator,
                                                   output_signature=(
                                                       tf.TensorSpec(shape=(input_shape[0], input_shape[1], 3),
                                                                     dtype=tf.float32),
                                                       tf.TensorSpec(shape=(input_shape[0], input_shape[1], 1),
                                                                     dtype=tf.float32)))

    validation_dataset = tf.data.Dataset.from_generator(validation_generator,
                                                        output_signature=(
                                                        tf.TensorSpec(shape=(input_shape[0], input_shape[1], 3),
                                                                          dtype=tf.float32),
                                                        tf.TensorSpec(shape=(input_shape[0], input_shape[1], 1),
                                                                          dtype=tf.float32)))



    # shuffle and group the train set into batches
    train_dataset = train_dataset.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    # do a prefetch to optimize processing
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    validation_dataset = validation_dataset.batch(BATCH_SIZE)

    return train_dataset, validation_dataset


def random_flip(input_image, input_mask):
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    return input_image, input_mask


def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask = tf.cast(input_mask, tf.float32) / 255.0
    return input_image, input_mask

@tf.function
def load_and_preprocess_image(image_path, input_shape):
    input_image = tf.io.read_file(image_path)
    input_image = tf.image.decode_png(input_image, channels=3)
    input_image = tf.image.resize(input_image, input_shape, method='nearest')
    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image

@tf.function
def load_and_preprocess_mask(image_name, image_shape, input_shape, train_df):
    input_mask = get_mask(image_name, image_shape, train_df)
    input_mask = tf.image.resize(input_mask, input_shape, method='nearest')
    input_mask = tf.cast(input_mask, tf.float32) / 255.0
    return input_mask

@tf.function
def load_image_train(image_name, image_shape, input_shape, train_path, train_df):
    '''resizes, normalizes, and flips the training data'''
    input_image = load_and_preprocess_image(train_path + image_name)
    input_mask = load_and_preprocess_mask(image_name, image_shape, input_shape, train_df)

    input_image, input_mask = random_flip(input_image, input_mask)

    return input_image, input_mask


def load_image_val(image_name, image_shape, input_shape, train_path, train_df):
    '''resizes and normalizes the test data'''
    input_image = load_and_preprocess_image(train_path + image_name)
    input_mask = load_and_preprocess_mask(image_name, image_shape, input_shape, train_df)

    return input_image, input_mask

