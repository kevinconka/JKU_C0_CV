The code can be found in the following git repo: https://github.com/kevinconka/JKU_C0_CV

The python notebooks contain instructions to run the detection and tracking in Google Colab

- YOLOv5_Train_Custom_Data.ipynb - Used to fine-tune the YOLOv5 with our annotated dataset
- YOLOv5_OCSORT.ipynb - Simple notebook to test YOLOv5 + tracker and possible to do only detection

You can also check "all_track.sh" for a glimpse of how all the "tracked_videos" were produced.

Note: Image datasets and videos are not included here due to file size limitations. It is possible to use the YOLOv5_OCSORT.ipynb notebook by simply uploading a video to Colab and adjusting the path in the right cells.
