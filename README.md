# project_cannabis
Project Nice or spice? Cannabis detection by Microscope inspection in GIP Lab Technion

# Using CNN Classifier:


# Annotating images:

To annotate images we used the Makesense tool: https://www.makesense.ai/
Annotate by drawing bounding boxes, assign a label to each bounding box. You can then export your annotated dataset in YOLO format.
You can also import an already annotated YOLO dataset. For that you need to provide the images, the annotation files for each image, as well as a label.txt file that contains label names.

# Using YOLO detection on custom dataset:
## Installation of YOLOv4: we used this very complete tutorial, the process can be tricky and long, follow steps carefully
https://www.youtube.com/watch?v=WK_2bpWj35A&t=1038s

## YOLOv4 GitHub Repository (darknet by AlexeyAB):
https://github.com/AlexeyAB/darknet

## Training YOLOv4 on custom dataset with Google Colab: we used this tutorial
https://medium.com/analytics-vidhya/train-a-custom-yolov4-object-detector-using-google-colab-61a659d4868

## Training YOLOv4 on custom dataset on Windows (after previous installation): we used this tutorial
https://medium.com/geekculture/train-a-custom-yolov4-object-detector-on-windows-fe5332b0ca95

## Using our YOLO model on GIP lab computers:

We created a yolov4 folder on Desktop. Inside you can find the darknet folder containing all YOLO code and files as well as darknet executable to run YOLO. You can also find the training folder containing different backup files to run different training (different cfg files, different .names and .obj files to run with different parameters, different folders to find the different weights we obtained for each training). They are ordered by training batches. Each batch contains more and more images.

Batch 2 contains 300 annotated images of true Cannabis only.
Batch 3 contains 450 annoated images of true Cannabis and 500 annotated images of fake Cannabis.
Batch 4 is batch 3 but augmented (for each Batch 3 image there is an augmented version), all annotated.

If you want to re-run a training, you can follow the given tutorial, using desired cfg, obj, names, and weight file.
