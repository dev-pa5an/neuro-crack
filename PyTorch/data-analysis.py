from PIL import Image
import matplotlib.pyplot as plt
import os
import glob
import torch
from torch.utils.data import Dataset
import skillsnetwork 

def show_data(data_sample, shape = (28, 28)):
    plt.imshow(data_sample[0].numpy().reshape(shape), cmap='gray')
    plt.title('y = ' + data_sample[1])

directory="resources/data"
negative='Negative'
negative_file_path=os.path.join(directory,negative)
negative_files=[os.path.join(negative_file_path,file) for file in  os.listdir(negative_file_path) if file.endswith(".jpg")]
negative_files.sort()

positive="Positive"
positive_file_path=os.path.join(directory,positive)
positive_files=[os.path.join(positive_file_path,file) for file in  os.listdir(positive_file_path) if file.endswith(".jpg")]
positive_files.sort()

number_of_samples = len(positive_files) + len(negative_files)
Y = torch.zeros([number_of_samples])

# As we are using the tensor Y for classification we cast it to a LongTensor.
Y = Y.type(torch.LongTensor)
print(Y.type())

# print(negative_files[0:3])
# print(positive_files[0:3])

# image1 = Image.open(negative_files[0])

# plt.imshow(image1)
# plt.title("1st Image With No Cracks")
# plt.show()