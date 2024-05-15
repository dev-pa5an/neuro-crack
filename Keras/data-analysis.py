import os
import numpy as np
import matplotlib.pyplot as plt
import skillsnetwork

from PIL import Image

negative_images = os.listdir('resources/data/Negative')
negative_images.sort()
# image_data = Image.open('resources/data/Negative/{}'.format(negative_images[0]))
# #print(plt.imshow(image_data))
negative_images_dir = ['resources/data/Negative/{}'.format(image) for image in negative_images]
# print(len(negative_images_dir)) #20000

positive_images = os.listdir('resources/data/Positive')
positive_images.sort()
positive_images_dir = ['.resources/data/Positive/{}'.format(image) for image in negative_images]
# print(len(positive_images_dir))