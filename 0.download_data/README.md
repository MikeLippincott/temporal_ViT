# Download pre-training data
This module downloads data for pre-training the model.
## Automated download of data
We download the following datasets through the scripts automatically:
### MNIST
- This is a collection of images of handwritten digits.
- To download the dataset, run the [0.download_MNIST_data.ipynb notebook](./notebooks/0.download_MNIST_data.ipynb).
- This notebook will download the dataset and save it in the `data` directory of the project.
Further, it will process the images to create a video of the digits in a scrolling window fashion.
### Martin Setvak's timelapse landscape videos
- [Martin Setvak's website](https://www.setvak.cz/timelapse/timelapse.html)
- Martin has an excellent collection of timelapse videos of landscapes.
We use these videos to pre-train the model.
We scrape the videos from his website and download them to the local machine.
- To download the videos, run the [1.download_timelapse_videos.py](./1.download_timelapse_videos.py) script.

## Manual download of data
We download the following datasets manually:
### WALT - Watch and Learn Time-Lapse Dataset
- This is a curation of time-lapse videos of urban scenes from street cameras.
- Dataset must be manually downloaded from [here]("https://drive.google.com/drive/folders/1qb7EUiMJ_fCjqDn2b6pos9QUVlt5L0Rr?usp=sharing")
### Moments in Time Dataset
- This is a collection of videos of everyday activities.
These videos are about 3 seconds long.
- Dataset must be manually downloaded from [here]("http://moments.csail.mit.edu/")

### Manual download instructions
- Download the datasets from the links provided above.
- Place the zippped files in the `raw_zips` directory of the project.
- Then run the [3.process_vidoes.ipynb notebook](./notebooks/3.process_videos.ipynb)


