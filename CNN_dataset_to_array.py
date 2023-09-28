
import os
import numpy as np
from PIL import Image

#import matplotlib.pyplot as plt
#from matplotlib.ticker import (MultipleLocator, FormatStrFormatter)
#import tensorflow as tf
#from tensorflow.python.keras.models import Sequential
#from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
#from sklearn.model_selection import KFold
#import seaborn as sn


# Set the path to the directory containing the images
data_dir = r"C:\Users\HP\OneDrive - Technion\Desktop\Computer_homeworks\semester 6\project\Cannabis images\Project_cropped_images"
#data_dir = r"C:\Users\HP\OneDrive - Technion\Desktop\Computer_homeworks\semester 6\project\Cannabis images\simulation_cropped_images"
#data_dir = r"C:\Users\arthurSoussan\project_cannabis\simulation_cropped_cannabis"
#data_dir = r"D:\Users Data\arthurSoussan\Desktop\cannabis_cropped_images"
# Define the image dimensions and number of classes
img_height, img_width = 224, 224
num_classes = 2

# Load the images into memory and create labels
X = []
y = []
for i, class_name in enumerate(os.listdir(data_dir)):
    class_path = os.path.join(data_dir, class_name)
    print("Class number ", i, "is the file ", class_name)
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        print("i: ", i, "img path:", img_path)
        if (os.path.isfile(img_path)):
          image = Image.open(img_path)
          img = image.resize((img_height, img_width))
          X.append(np.array(img) / 255.0)
          y.append(i)

# Convert the data to numpy arrays
X = np.array(X)
y = np.array(y)


x_array = X
y_array = y

print("x array shape: ", x_array.shape)
print("y array shape: ", y_array.shape)

np.save("x_array_4d", x_array)
np.save("y_array", y_array)

print("the arrays are saved!")
x_loaded_array = np.load('x_array_4d.npy')
y_loaded_array = np.load('y_array.npy')

print("the array is loaded!")
print("x array loaded shape: ", x_loaded_array.shape)
print("y array loaded shape: ", y_loaded_array.shape)


# checking if both the Arrays are same
if (x_loaded_array == x_array).all():
    if (y_loaded_array == y_array).all():
        print("All elements are same")
else:
    print("All elements are not same")