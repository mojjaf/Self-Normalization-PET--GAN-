import os 
import numpy as np 
import numpy as np
import tensorflow as tf
from mainargs import get_args
import h5py
from pathlib import Path
from glob import glob

args = get_args()
IMG_HEIGHT=args.image_size
IMG_WIDTH=args.image_size
INPUT_CHANNELS=args.input_channel
OUTPUT_CHANNELS = args.output_channel


def progressbar(progress,total):
    percent= 100*(progress/ float(total))
    bar= '=' * int(percent) + '-' * (100- int(percent))
    print(f"\r |{bar}| {percent:.2f}%", end="\r")


# Normalizing the images in each slice to [-1, 1]
def normalise_per_channel(image):
    rank = 4
    ch_mins = np.amin(image, axis=tuple(range(rank - 1)))
    ch_maxs = np.amax(image, axis=tuple(range(rank - 1)))
    ch_range = ch_maxs - ch_mins
    
    #idx = np.where(ch_range == 0)
    #ch_mins[idx] = 0
    #ch_range[idx] = 1
    img = (image - ch_mins) / ch_range
    image_norm = 2*img - 1
    
    return image_norm



# Normalizing the images in each channel to [-1, 1]

def normalize_per_slice(image):
    image_norm = np.zeros_like(image)
    #print(image.shape)
    w, h, c= image.shape
    for i in range(c):
        im_range=np.max(image[...,i])-np.min(image[...,i])
        img=(image[...,i]-np.min(image[...,i]))/im_range                                
        image_norm[...,i] = 2*img - 1
    
    return image_norm


def normalize_tensor(input_image, real_image):
    input_image=normalise_per_channel(input_image)
    real_image=normalise_per_channel(real_image)

    return input_image, real_image


def load(image_file):
    # Read and decode an image file to a uint8 tensor
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)

    # Split each image tensor into two tensors:
    # - one with a normalized image
    # - one with an un-normalized image 
    w = tf.shape(image)[1]
    w = w // 2
    real_image = image[:, :w, :]
    input_image  = image[:, w:, :]

    # Convert both images to float32 tensors
    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    return input_image, real_image

def load_h5(image_file):
    # Read and decode an image file to a uint8 tensor
    img = h5py.File(image_file,'r')
    key = list(img.keys())[0]
    image = img[key][:]
    image = np.array(image)
    image = tf.cast(image, tf.float32)
    return image


def load_image_data_from_directory(image_dirA,image_dirB,height, width):
    input_image_list= []#np.zeros((ids,image_size, image_size, 3), dtype=float)
    real_image_list= []#np.zeros((ids,image_size, image_size, 3), dtype=float)
    image_suffixes = (".h5",".mat",".jpeg", ".jpg", ".png", ".tif")
    image_paths = [p for p in Path(image_dirA).glob("**/*") if p.suffix.lower() in image_suffixes]
    progressbar(0, len(image_paths))
    i=0
    for image_fileA,image_fileB in zip(os.listdir(image_dirA),os.listdir(image_dirB)):
        #print(image_path)
        ImgA= load_h5(os.path.join(image_dirA, image_fileA)) #input image
        ImgB= load_h5(os.path.join(image_dirB, image_fileB)) #target image
        ImgA=np.array(ImgA)
        ImgB=np.array(ImgB)
        #print(" input and output images with original shape",np.shape(ImgA), np.shape(ImgB))
        if OUTPUT_CHANNELS ==1:
            ImgB=np.expand_dims(ImgB, axis=-1)


        if INPUT_CHANNELS ==1:
            ImgA=np.expand_dims(ImgA, axis=-1)
       # print("input and output tensors dim",np.shape(ImgA), np.shape(ImgB))

        input_image,real_image=resize(ImgA, ImgB, height, width)
        input_image,real_image=normalize_tensor(input_image,real_image)
        input_image_list.append(input_image)
        real_image_list.append(real_image)
        progressbar(i+1, len(image_paths))
        i+=1
    return input_image_list,real_image_list

def load_files_from_directory(image_dirA,image_dirB,height, width):
    input_image_list= []#np.zeros((ids,image_size, image_size, 3), dtype=float)
    real_image_list= []#np.zeros((ids,image_size, image_size, 3), dtype=float)
    image_suffixes = (".h5",".mat",".jpeg", ".jpg", ".png", ".tif")
    image_filenames = [p for p in Path(image_dirA).glob("**/*") if p.suffix.lower() in image_suffixes]
    progressbar(0, len(image_filenames))
    i=0
    for image_file in (os.listdir(image_dirA)):
        #print(image_file)
        ImgA= load_h5(os.path.join(image_dirA, image_file)) #input image
        ImgB= load_h5(os.path.join(image_dirB, image_file)) #target image file name is equal to the input image file name
        ImgA=np.array(ImgA)
        ImgB=np.array(ImgB)
        #print(" input and output images with original shape",np.shape(ImgA), np.shape(ImgB))
        if OUTPUT_CHANNELS ==1:
            ImgB=np.expand_dims(ImgB, axis=-1)


        if INPUT_CHANNELS ==1:
            ImgA=np.expand_dims(ImgA, axis=-1)
       # print("input and output tensors dim",np.shape(ImgA), np.shape(ImgB))

        input_image,real_image=resize(ImgA, ImgB, height, width)
        input_image,real_image=normalize_tensor(input_image,real_image)
        input_image_list.append(input_image)
        real_image_list.append(real_image)
        progressbar(i+1, len(image_filenames))
        i+=1
    return input_image_list,real_image_list



def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image

# Normalizing the  8bit images to [-1, 1]
def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image


def load_image_train(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = resize(input_image, real_image,
                                   IMG_HEIGHT, IMG_WIDTH)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image

def load_image_test(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = resize(input_image, real_image,
                                   IMG_HEIGHT, IMG_WIDTH)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image



def generate_images(model, input_image, target):
    prediction = model(input_image, training=True)
    plt.figure(figsize=(15, 15))

    display_list = [input_image[0], target[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # Getting the pixel values in the [0, 1] range to plot.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()


