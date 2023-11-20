# Drone Detection & Tracking using YOLOv8 
## _How To Use_

Steps:
First run:
```python
myenv\Scripts\activate
```
1. Then Determine if you want to use the dataset and videos provided.
\
a. If you want to use your own videos to test open the region for Download Videos and change the URL's in the vid_urls array to include your videos or just have a link to your video.
\
b. If you want to use your own dataset you will need to replace the current dataset_url to the zip file of your dataset. If you will need to edit/remove a few lines in the download dataset region to make it more optimal for your dataset.

2. If you kept the same dataset and you see drone_obj_det.pt in the same folder as Assignment_3_UAV_GK285.py then you can run the program and it will download anything that it is currently missing and needed to run then it will show you the AI detecting the drone.
\
a. If you chose to also change your dataset I would advise moving the drone_obj_det.pt file into the Old_drone_det Models folder this way you can always go back to it. Then check the data augmentations region to make any edits if you wanted to change add or remove augmentations to the dataset.

3. After running there should be a detection folder that has images with the drone and dots for each time the AI predicted the drone, the size to determine the distance the drone was and the space between dots to get an idea of how fast the drone was moving, lastly the color to see the difference in time between dots.
\
a. If you just trained a model with your dataset it will automatically get the last.pt in the model/weights folder and make that your new drone_obj_det.pt so that if you change the video or anything else you will not need to retrain the model it will always choose that model to run. If you would like to re-train it move that model into the old_model folder or delete it and rerun the program.

If you do not use the enviroment provided you will need to manually pip install all the required libraries.

If you wish to look at the dataset currently being used you can download it here [Drone_Dataset](https://drive.google.com/u/4/uc?id=16CMtbV2XoZvIrVLOOjlzICNesZmGbQM_&export=download&confirm=t&uuid=e54f6130-6999-414c-aea2-555b49e873ed&at=AB6BwCBRCmV2MdO8SgQv15mSXYuB:1699477691972) It should be noted that this dataset is the non-modified dataset. This is because I use this python script to make the augmentations to the dataset. So every time I re-train my model I usually created a new dataset for it too.