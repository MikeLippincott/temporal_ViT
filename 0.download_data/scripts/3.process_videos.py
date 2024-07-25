#!/usr/bin/env python
# coding: utf-8

# This notebook processes the files to be used for pre-training models.
# The data need to be in a specific format to be used for the model.
# Each of the zipped files will be unzipped and the data will be processed to be used for the model.
# I will also extract each frame from each video in the dataset and save it as an image.

# In[1]:


import pathlib
import shutil
import zipfile

import cv2
import pandas as pd
import py7zr
import tifffile
import tqdm

# In[2]:


# set processes to run
run_convert_videos = True
run_unzip_files = True


# ### Functions

# In[3]:


def unzip_files(zip_file: pathlib.Path, output_dir: pathlib.Path) -> None:
    """
    This function unzips a file from a directory to a specified directory.

    Parameters
    ----------
    zip_file : pathlib.Path
        The path to a specific zip file.
    output_dir : pathlib.Path
        The path to the directory where the unzipped files will be saved.

    Returns
    -------
    None
    """
    if zip_file.is_dir():
        unzip_files(zip_file, output_dir)
    elif zip_file.is_file():
        try:
            if zip_file.suffix == ".zip":
                # if the zip is already unzipped, skip it
                if not (output_dir / zip_file.stem).exists():
                    with zipfile.ZipFile(zip_file, "r") as zip_ref:
                        zip_ref.extractall(pathlib.Path(output_dir / zip_file.stem))
            elif zip_file.suffix == ".7z":
                if not (output_dir / zip_file.stem).exists():
                    with py7zr.SevenZipFile(zip_file, mode="r") as z7:
                        z7.extractall(pathlib.Path(output_dir / zip_file.stem))
            else:
                print(f"Unrecognized file type: {zip_file.suffix}")
        except Exception as e:
            print(f"Error extracting {zip_file}: {e}")


# In[4]:


def unnest_list(list_of_lists: list) -> list:
    """This function unnests a list of lists.
    or a list of lists of lists, etc.

    Parameters
    ----------
    list_of_lists : list
        A list of lists to be unnested

    Returns
    -------
    list
        A single list with all elements from the input list
    """
    output_list = []
    for l in list_of_lists:
        if isinstance(l, list):
            output_list.extend(unnest_list(l))
        else:
            output_list.append(l)
    return output_list


# In[5]:


def get_files_from_dir(dir: pathlib.Path) -> list:
    """
    This function gets all files from a directory and its subdirectories.

    Parameters
    ----------
    dir : pathlib.Path
        The path to the directory.

    Returns
    -------
    list
        A list of all files in the directory and its subdirectories.
    """
    list_of_files = list(dir.glob("*"))
    output_list = []
    for dir in list_of_files:
        if dir.is_file():
            output_list.append(dir)
        elif dir.is_dir():
            output_list.append(get_files_from_dir(dir))
    return output_list


# In[6]:


def get_dir_of_files(dir: pathlib.Path) -> list:
    """This function gets all directories that contain files.
    Specifically, it gets all directories that contain files that are not .zip or .7z files.

    Parameters
    ----------
    dir : pathlib.Path
        The path to the directory.

    Returns
    -------
    list
        A list of all directories that contain files that are not .zip or .7z files.
    """
    list_of_files = list(dir.glob("*"))
    output_list = []
    for dir in list_of_files:
        # get the number of files in the directory
        if (
            len(
                [
                    i
                    for i in dir.glob("*")
                    if i.is_file()
                    and not (
                        i.name.endswith(".zip")
                        or i.name.endswith(".7z")
                        or i.name.endswith("csv")
                        or i.name.endswith("txt")
                    )
                ]
            )
            > 0
        ):
            output_list.append(dir)
        elif dir.is_dir():
            output_list.append(get_dir_of_files(dir))
    return output_list


# ### Data Processing Steps:
# 1. Unzip the files
# 2. Move the files to the correct directory
# 3. Extract the frames from the videos and save them as images

# ## Unzip the files

# In[7]:


if run_unzip_files:
    # set path to the directory containing the zips
    all_zips_Path = pathlib.Path(
        "/home/lippincm/Desktop/18TB/timelapse_data_landscapes/raw_zips/"
    ).resolve(strict=True)
    # get a list of all ".zip and .7z" files in the directory
    list_of_zips = list(all_zips_Path.glob("*.zip")) + list(all_zips_Path.glob("*.7z"))
    print(len(list_of_zips))

    # unzip the files and save them in the same directory
    output_dir = pathlib.Path(
        "/home/lippincm/Desktop/18TB/timelapse_data_landscapes/unzipped/"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for z in tqdm.tqdm(list_of_zips):
        unzip_files(z, output_dir)

    # get a list of all dirs in the unzipped directory for nested zip files
    list_of_extracted_files = get_files_from_dir(output_dir)
    list_of_extracted_files = unnest_list(list_of_extracted_files)
    # unzip the files and save them in the same directory
    list_of_extracted_files = [
        f for f in list_of_extracted_files if f.suffix == ".zip" or f.suffix == ".7z"
    ]
    print(len(list_of_extracted_files))
    for f in tqdm.tqdm(list_of_extracted_files):
        unzip_files(f, f.parent)


# ## Organize files for frame extraction

# In[8]:


# get a list of all dirs in the unzipped directory for nested zip files
output_list = get_dir_of_files(output_dir)
# unnest the list of lists
output_list = unnest_list(output_list)
print(len(output_list))
# loop through the list of directories and move the files to the appropriate directory
for dir in output_list:
    # get all files in the directory
    files = list(dir.glob("*"))
    if files[0].suffix == ".jpg":
        new_dir_path = pathlib.Path(
            f"/home/lippincm/Desktop/18TB/timelapse_data_landscapes/tiff_frames/{dir.name}"
        )
        if not new_dir_path.exists():
            shutil.move(dir, new_dir_path)
    elif files[0].suffix == ".mp4":
        new_dir_path = pathlib.Path(
            f"/home/lippincm/Desktop/18TB/timelapse_data_landscapes/raw_videos/{dir.name}"
        )
        if not new_dir_path.exists():
            shutil.move(dir, new_dir_path)
    else:
        print(f"Unrecognized file type: {files[0].suffix}")


# ## Convert the videos to images

# In[9]:


if run_convert_videos:
    # set the path to the directory containing the files
    all_videos_Path = pathlib.Path(
        "/home/lippincm/Desktop/18TB/timelapse_data_landscapes/raw_videos/"
    ).resolve(strict=True)
    # ouput directory for the tiff files of frames
    output_dir = pathlib.Path(
        "/home/lippincm/Desktop/18TB/timelapse_data_landscapes/tiff_frames/"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # get a list of all the files in the directory
    list_of_files = get_files_from_dir(all_videos_Path)
    list_of_files = unnest_list(list_of_files)
    print(len(list_of_files))
    list_of_files = [f for f in list_of_files if f.suffix == ".mp4"]
    print(len(list_of_files))
    metadata_dict = {
        "file_name": [],
        "file_path": [],
        "file_size": [],
        "fps": [],
        "frame_count": [],
        "duration": [],
        "width": [],
        "height": [],
        "num_pixels": [],
    }

    for video in tqdm.tqdm(list_of_files):
        # extract metadata from the video
        cap = cv2.VideoCapture(str(video))
        metadata_dict["file_name"].append(video.name)
        metadata_dict["file_path"].append(video)
        metadata_dict["file_size"].append(video.stat().st_size)
        metadata_dict["fps"].append(cap.get(cv2.CAP_PROP_FPS))
        metadata_dict["frame_count"].append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        metadata_dict["duration"].append(
            metadata_dict["frame_count"][-1] / metadata_dict["fps"][-1]
        )
        metadata_dict["width"].append(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        metadata_dict["height"].append(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        metadata_dict["num_pixels"].append(
            metadata_dict["width"][-1] * metadata_dict["height"][-1]
        )
        video_output_path = pathlib.Path(f"{output_dir}/{video.stem}").resolve()
        video_output_path.mkdir(parents=True, exist_ok=True)

        # load the video
        cap = cv2.VideoCapture(str(video))
        # iterate over the frames and save them as tiff files
        for frame in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            cap.set(cv2.CAP_PROP_FRAME_COUNT, frame)
            ret, img = cap.read()
            img_file_path = pathlib.Path(
                f"{video_output_path}/{frame}_{video.stem}.tiff"
            ).resolve()
            if ret:
                if not img_file_path.exists():
                    tifffile.imwrite(
                        img_file_path,
                        img,
                    )
        cap.release()
    # write the metadata to a csv file
    metadata_df = pd.DataFrame(metadata_dict)
    metadata_df.to_csv(output_dir / "metadata_of_tiffs.csv", index=False)


# In[10]:


metadata_df.head()
