import sys
import os
import pickle

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


def plot_confusion_matrix(cm, label_names, title='Confusion matrix', cmap=plt.cm.Blues):
    """This function create a confusion matrix from the data
    Keyword arguments:
        cm -- confusion matrix
        label_names -- array with name of each class
        title -- title for the plot
        cmap -- color to the confusion matrix
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(np.unique(label_names)))
    plt.xticks(tick_marks, np.unique(label_names), rotation=45)
    plt.yticks(tick_marks, np.unique(label_names))
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def read_images(path_folder, im_size, pixel_max=255.0):
    """This function read all image inside of folder, create a 3D dataset where the first dimension is a image,
    the second dimension is the row and the third dimension is the column from the image  

    :param path_folder: folder where search for images
    :param im_size: image's size
    :param pixel_max: image's Maximum intensity 
    :return dataset: 3D matrix (num_images x row_image x column_image) 
    """

    # return all images.png (file) into folder
    images = os.listdir(path_folder)
    print(full_path + '. Fist image: ' + images[0])
    dataset = np.ndarray(shape=(len(images), im_size, im_size), dtype=np.float32)
    print('Dataset size: ' + str(np.shape(dataset)))

    # read each image from folder vowel and save it in a ndarray matrix (num_images, rows, cols)
    idx = 0
    for image in images:
        # load image in gray scale and normalize
        try:
            img = Image.open(os.path.join(full_path, image))
            img_as_np = np.array(img.convert('L'))
            dataset[idx, :, :] = (img_as_np.astype(float) - pixel_max / 2) / pixel_max
        except IOError as e:
            print("the image could not be read: " + image + "... skipping it")
            print(e)
        idx += 1
    return dataset


def save_dataset(data, folder, file_name):
    """Function to save data in the given file_name and in specific folder

    :param data: dataset to save
    :param folder: folder where save the data
    :param file_name: name of the file
    :return bool: True if the process was success. False in other case 
    """

    # check whether the file exist
    if not os.path.isdir(folder):
        path_after, folder_to_create = os.path.split(folder)
        os.mkdir(os.path.join(path_after, folder_to_create))

    # try to save the file
    try:
        with open(os.path.join(folder, file_name), 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

        return True
    # exception if the system can save the file
    except Exception as e:
        print('Unable to save data in', file_name, ':', e)
        return False


def split_data(words: dict, percentage=0.6):
    train_data = np.ndarray
    train_label = np.ndarray
    validation_data = np.ndarray
    validation_label = np.ndarray
    len_t = 0
    len_v = 0
    index = 1
    for lbl, word in words.items():
        # create a random vector for the length of the image
        rand_perm = np.random.permutation(range(0, len(word)))
        # 60% for training
        rand_t = rand_perm[:int(np.floor(len(rand_perm) * 0.6))]
        len_t += len(rand_t)
        # 40% for validation
        rand_v = rand_perm[int(np.ceil(len(rand_perm) * .6)):]
        len_v += len(rand_v)
        # if is the first iteration save data in the final matrix
        if index == 1:
            train_data = word[rand_t, :, :]
            print(word[rand_t[0], :, :])
            train_label = np.ones((len(rand_t), 1)) * lbl
            validation_data = word[rand_v, :, :]
            validation_label = np.ones((len(rand_v), 1)) * lbl
            index += 1
        # else, it's necessary concatenate data by ROW
        else:
            aux_train = word[rand_t, :, :]
            print(word[rand_t[0], :, :])
            train_data = np.concatenate((train_data, aux_train), axis=0)
            train_label = np.concatenate((train_label, np.ones((len(rand_t), 1)) * lbl), axis=0)
            aux_validation = word[rand_v, :, :]
            validation_data = np.concatenate((validation_data, aux_validation), axis=0)
            validation_label = np.concatenate((validation_label, np.ones((len(rand_v), 1)) * lbl), axis=0)
            index += 1
    print(str(len_t) + ": " + str(len_v))
    return train_data, train_label, validation_data, validation_label


def concatenate_test_data(words: dict):
    test = np.concatenate([words[k] for k in sorted(words.keys())])
    labels = np.concatenate([np.ones([len(words[k]), 1]) * k for k in sorted(words.keys())])
    return test, labels


path_train_data = 'D:\\git\\databases\\images\\notMNIST_large'
path_test_data = 'D:\\git\\databases\\images\\notMNIST_small'
root = 'D:\\git\\image_tensorflow\\image_processing'
# set root path
os.chdir(root)
# image size 28x28
image_size = 28
pixel_depth = 255.0
train_save_folder = 'train_data'
test_save_folder = 'test_data'

# create a list with all image path
full_images_paths = list()
if not os.path.isdir(os.path.join(root, train_save_folder)):
    os.mkdir(os.path.join(root, train_save_folder))
if not os.path.isdir(os.path.join(root, test_save_folder)):
    os.mkdir(os.path.join(root, test_save_folder))

# get all folders (file) into path (folder by each vowel)
vowel_folders = os.listdir(path_train_data)
for vowel in vowel_folders:
    # full path for vowel image
    full_path = os.path.join(path_train_data, vowel)

    images_set = read_images(path_folder=full_path, im_size=image_size, pixel_max=pixel_depth)
    was = save_dataset(data=images_set, folder=os.path.join(root, train_save_folder), file_name=vowel + '.pickle')


vowel_folders = os.listdir(path_test_data)
for vowel in vowel_folders:
    # full path for vowel image
    full_path = os.path.join(path_test_data, vowel)

    images_set = read_images(path_folder=full_path, im_size=image_size, pixel_max=pixel_depth)
    was = save_dataset(data=images_set, folder=os.path.join(root, test_save_folder), file_name=vowel + '.pickle')



#######################################################################
# Use the pickle data set
#######################################################################
# over here we suppose that the system never use the data before

# create an empty dataset for training data

dataset_names = ['A.pickle', 'B.pickle', 'C.pickle', 'D.pickle', 'E.pickle',
                 'F.pickle', 'G.pickle', 'H.pickle', 'I.pickle', 'J.pickle']
train_save_folder = 'train_data'
test_save_folder = 'test_data'
letter_train = dict()
letter_test = dict()
for label, pickle_file in enumerate(dataset_names):
    try:
        # load data for the given vowel
        database_path = os.path.join(root, train_save_folder)
        with open(os.path.join(database_path, pickle_file), 'rb') as f:
            if int(sys.version_info[0]) >= 3:
                letter_train[label] = pickle.load(f)
            else:
                letter_train[label] = pickle.load(f)
    except Exception as ex:
        print('Unable to process data from', pickle_file, ':', ex)
        raise

    try:
        # load data for the given vowel
        database_path = os.path.join(root, test_save_folder)
        with open(os.path.join(database_path, pickle_file), 'rb') as f:
            if int(sys.version_info[0]) >= 3:
                letter_test[label] = pickle.load(f)
            else:
                letter_test[label] = pickle.load(f)
    except Exception as ex:
        print('Unable to process data from', pickle_file, ':', ex)
        raise


train_data, train_label, validation_data, validation_label = split_data(words=letter_train, percentage=0.6)

test_data, test_label = concatenate_test_data(words=letter_test)
n = 18000
plt.imshow(test_data[n, :, :])
plt.title(test_label[n])
plt.waitforbuttonpress()
# save a final file with all data
pickle_file = 'notMNIST.pickle'

try:
    f = open(os.path.join(root, pickle_file), 'wb')
    save = {
        'train_dataset': train_data,
        'train_labels': train_label,
        'valid_dataset': validation_data,
        'valid_labels': validation_label,
        'test_dataset': test_data,
        'test_labels': test_label
    }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
except Exception as ex:
    print('Unable to save data to', pickle_file, ':', ex)
    raise
