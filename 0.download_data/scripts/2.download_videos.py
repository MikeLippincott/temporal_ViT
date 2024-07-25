#!/usr/bin/env python
# coding: utf-8

# This notebook downloads timelapse videos of landscapes from [Martin Setvak's website](https://www.setvak.cz/timelapse/timelapse.html).
# The purpose of downloading these videos is to use them as training data for a temporal Vision Transformer (ViT) model.

# In[1]:


import pathlib
import time

import numpy as np
import requests
import tqdm
from bs4 import BeautifulSoup

# In[2]:


# set the path to where the data will be stored
# this is hardcoded and an absolute path this must be changed for your system
output_path = pathlib.Path(
    "/home/lippincm/Desktop/18TB/timelapse_data_landscapes/raw_videos"
).resolve()
output_path.mkdir(parents=True, exist_ok=True)


# ### Setting up a web scraper to capture the video URLs

# In[3]:


# set up the web scraping
url = "https://www.setvak.cz/timelapse/"
list_of_years = [
    2024,
    2023,
    2022,
    2021,
    2020,
    2019,
    2018,
    2017,
    2016,
    2015,
    "2015a_Tenerife",
    2014,
    2013,
    "2013a_HoheTauern",
    2012,
    2011,
    2010,
    "2010a_USA",
    2009,
    2008,
    2007,
    2006,
]
reponses_dict = {}
for year in list_of_years:
    reponses_dict[year] = requests.get(f"{url}{year}.html")
    reponses_dict[year].raise_for_status()


# In[4]:


# parse the html from the dictionary
list_of_links = []
for year in reponses_dict:
    if reponses_dict[year].status_code != 200:
        print(f"Error: {year}")
        continue
    soup = BeautifulSoup(reponses_dict[year].content, "html.parser")
    # convert the soup to a string
    soup_str = str(soup)
    for line in soup_str.split("\n"):
        if str(line).find(".mp4") != -1:
            if "href" not in line:
                print(line)
            else:
                list_of_links.append(
                    f"https://www.setvak.cz/timelapse/{BeautifulSoup(line, 'html.parser').find_all('a')[0].get('href')}"
                )


# In[5]:


for link in list_of_links:
    if ".mp4" not in link:
        # remove the link from the list
        list_of_links.remove(link)
        continue


# In[6]:


# download the videos carefully - this will take a while and a lot of space
# we do not want to download the same video twice
# we do not want to download the video if it is already in the folder
# we do not want to get blacklisted either
for link in tqdm.tqdm(list_of_links):
    # get the name of the video
    video_name = link.split("/")[-1]
    # set the path to the video file
    video_path = output_path / f"{video_name}"
    # check if the video is already downloaded
    if video_path.exists():
        continue
    # download the video
    try:
        video_response = requests.get(link)
        video_response.raise_for_status()
        # save the video
        with open(video_path, "wb") as f:
            f.write(video_response.content)
    except requests.exceptions.HTTPError as e:
        print(f"Error: {e}")
        continue

    time.sleep(
        np.random.randint(1, 5)
    )  # sleep for a random amount of time to avoid getting blacklisted


# In[7]:


# # https://www.cs.cmu.edu/~walt/license.html
WALT_LINK = "https://drive.google.com/drive/folders/1qb7EUiMJ_fCjqDn2b6pos9QUVlt5L0Rr?usp=sharing"

# lastly, download the WALT dataset manually from the google drive link


# ## Download the moments in time MiT dataset
# * https://arxiv.org/abs/1801.03150 - Mathew Monfort et al. "Moments in Time Dataset: one million videos for event understanding" 2018
#

# In[8]:


MiT_url = "http://moments.csail.mit.edu/splits/Moments_in_Time_Raw_v2.zip"
# download manually
