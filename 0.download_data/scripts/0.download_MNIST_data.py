#!/usr/bin/env python
# coding: utf-8

# This noteboook downloads the MNIST dataset

# In[1]:


import pathlib
import pickle

# mnist data parser import
import mnist

# In[2]:


mnist.datasets_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"

# make the mnist directory
mnist_path = pathlib.Path("../../data/mnist")
mnist_path.mkdir(parents=True, exist_ok=True)

# package specific path to mnist data
mnist.temporary_dir = lambda: str(mnist_path)


# In[3]:


train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()


# In[4]:


print("Saving data into a pickle file ...")
data = {
    "train_image": train_images,
    "train_label": train_labels,
    "test_image": test_images,
    "test_label": test_labels,
}
with open("../../data/mnist/MNIST.pkl", "wb") as file_handle:
    pickle.dump(data, file_handle)
