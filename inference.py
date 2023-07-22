##########################################
#
##INFERENCE
#
####################################################
from os import listdir
from xml.etree import ElementTree as ET
import numpy as np
from numpy import zeros
from numpy import asarray
from mrcnn.utils import Dataset
from matplotlib import pyplot
from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes
from PIL import Image
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.utils import compute_ap
from matplotlib.patches import Rectangle
import mrcnn
import mrcnn.config
import mrcnn.model
import mrcnn.visualize
import cv2
import os
from os import listdir
from xml.etree import ElementTree as ET
import numpy as np
from numpy import zeros
from numpy import asarray
from mrcnn.utils import Dataset
from matplotlib import pyplot
from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes
from mrcnn.utils import compute_ap
from PIL import Image
from mrcnn.config import Config
from mrcnn.model import MaskRCNN

#import re

# class that defines and loads the kangaroo dataset
class RootsDataset(Dataset):
    # load the dataset definitions
   #global child, width,height,boxes
    def load_dataset(self, dataset_dir, is_train=True):
        # define classes
        self.add_class("dataset", 1, "primary_root")
        #self.add_class("dataset", 2, "banana")
        #self.add_class("dataset", 3, "orange")

        # define data locations
        images_dir = dataset_dir + '/images/'
        annotations_dir = dataset_dir + '/annotations_new/'


		# find all images
        for filename in listdir(images_dir):
            #print(filename)
			# extract image id
            image_id = filename[:-4]
			#print('IMAGE ID: ',image_id)

			# skip all images after 3525 if we are building the train set
            if is_train and int(image_id) >= 3800:
                continue
			# skip all images before 115 if we are building the test/val set
            if not is_train and int(image_id) <3800:
                continue
            img_path = images_dir + filename
            ann_path = annotations_dir + image_id + '.xml'
			# add to dataset
            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path, class_ids = [1])


dataset_dir='D:\Mayank\BTP_2.0\Object_detection\Root_detection\Roots'
# test/val set
test_set = RootsDataset()
test_set.load_dataset(dataset_dir, is_train=True)
test_set.prepare()
print('*************Test*****************: %d' % len(test_set.image_ids))
import skimage
image_id=3000
print(test_set.image_info[image_id]['path'])
# load the masks and the class ids
mask, class_ids = test_set.load_mask(image_id)
#print(mask.shape[0],class_ids)
# extract bounding boxes from the masks
bbox = extract_bboxes(mask)
# define the prediction configuration
class PredictionConfig(Config):
	# define the name of the configuration
	NAME = "roots_cfg"
	# number of classes (background + 3 roots)
	NUM_CLASSES = 1 + 1
	# simplify GPU config
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1
 

CLASS_NAMES=['primary_root']
# create config
cfg = PredictionConfig()
# define the model
model = MaskRCNN(mode='inference', model_dir='logs', config=cfg)
# load model weights
model.load_weights('logs\mask_rcnn_roots_cfg_0005.h5', by_name=True)


#Test on a few images
import skimage
import cv2
#Test on a single image

#root_img = skimage.io.imread("datasets/renamed_to_numbers/images/184.jpg") #Try 028, 120, 222, 171

#Download a new image for testing...
#https://c2.peakpx.com/wallpaper/603/971/645/root-root-bowl-roots-apple-wallpaper-preview.jpg

#for taking out a image from test set 
#root_img = skimage.io.imread(test_set.image_info[image_id]['path'])

# for inferencing for gray images 

#gray = skimage.io.imread(r"C:\Users\LAB-502-08\Desktop\Rootimages_13092021\20210820\IMG_9970.JPG",0)
#root_img = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)

#for normal rgb image rendering
root_img = skimage.io.imread(r"C:\Users\LAB-502-08\Desktop\IMG_9972.JPG")
r = model.detect([root_img])[0] 
print(r['masks'][0])
print(r)
mrcnn.visualize.display_instances(image=root_img, 
                                 boxes=r['rois'], 
                                 masks=r['masks'], 
                                 class_ids=r['class_ids'], 
                                 class_names=CLASS_NAMES)
pyplot.imshow(root_img)
ax = pyplot.gca()
class_names = ['primary_root']
class_id_counter=1
for box in r['rois']:
    #print(box)
#get coordinates
    detected_class_id = r['class_ids'][class_id_counter-1]
    #print(detected_class_id)
    #print("Detected class is :", class_names[detected_class_id-1])
    y1, x1, y2, x2 = box
    #calculate width and height of the box
    width, height = x2 - x1, y2 - y1
    print('width of box ',width,' Height of box ',height)
    #create the shape
    ax.annotate(class_names[detected_class_id-1], (x1, y1), color='black', weight='bold', fontsize=10, ha='center', va='center')
    rect = Rectangle((x1, y1), width, height, fill=False, color='red',)
#draw the box
    ax.add_patch(rect)
    class_id_counter+=1
#show the figure
pyplot.show()
detected = model.detect([root_img])[0]
pred_mask=detected['masks']
pred_box= extract_bboxes(pred_mask)
pred_class_ids= detected['class_ids']
pred_scores= detected['scores']
print(pred_scores)
print(detected.values())
mAP,precisions,recalls,overlaps= compute_ap(bbox,class_ids,mask,pred_box,pred_class_ids,pred_scores,pred_mask)
print('mAP',mAP)
print('precision',precisions)
print('recalls',recalls)
print('overlaps',overlaps)
