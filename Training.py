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
import cv2 
from PIL import Image 
import PIL 
import re
import os
import tensorflow as tf

# class that defines and loads the kangaroo dataset
class RootsDataset(Dataset):
    # load the dataset definitions
   #global child, width,height,boxes 
    def load_dataset(self, dataset_dir, is_train=True):
        # define classes
        self.add_class("dataset", 1, "primary_root")
        
        # define data locations
        images_dir = dataset_dir + '\\images\\'
        annotations_dir = dataset_dir + '\\annotations_new\\'
       
             
		# find all images
        for filename in listdir(images_dir):
            print(filename)
			# extract image id
            image_id = filename[:-4]
			#print('IMAGE ID: ',image_id)
			
			# skip all images after 3525 if we are building the train set
            if is_train and int(image_id) >= 3796:
                continue
			# skip all images before 115 if we are building the test/val set
            if not is_train and int(image_id) < 3796:
                continue
            img_path = images_dir + filename
            ann_path = annotations_dir + image_id + '.xml'
			# add to dataset
            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path, class_ids = [1])

    # function to return the size of image 
    def sizeOfImage(self,filename):
        # get image
        img = Image.open(filename) 
        # get width and height
        width = img.width
        height = img.height
        # retuns the width and height of the image
        return width, height

    # Extrating the bounding boxes
    def extract_bounding_boxes(self,filename):
        bounding_boxes=list()
        tree = ET.parse(filename)
        root = tree.getroot()
        k=0 
        x=[]
        y=[]
        x_min=[]
        y_min=[]
        x_max=[]
        y_max=[]
        width=[]
        height=[]
        coors=[]
        print(len(root[1]))
        while k< len(root[1]):# plant counter
            #print('*************Root',i+1,'***************')
            for i in range(len(root[1][k])): # corresponding root counter
                for j in range(len(root[1][k][i][0][1])): # corresponding coordinate counter
                  x_coor= root[1][k][i][0][1][j].attrib['x']
                  y_coor= root[1][k][i][0][1][j].attrib['y']
                  #print(x,y)
                  x.append(float(x_coor))
                  y.append(float(y_coor))
                x_max.append(max(x))
                x_min.append(min(x))
                y_max.append(max(y))
                y_min.append(min(y))
                coors=[x,y,x_min[0],y_min[0],x_max[0],y_max[0]]
                
                x=[]
                y=[]
                x_max=[]
                x_min=[]
                y_max=[]
                y_min=[]
                width=[]
                height=[]
                bounding_boxes.append(coors)
            k=k+1 
            return bounding_boxes

    # Loading the mask
    def load_mask(self,image_id):
      info= self.image_info[image_id]
      path= info['annotation']
      path_to_image=info['path']
      boxes= self.extract_bounding_boxes(path)
      w,h= self.sizeOfImage(path_to_image) 
      masks = zeros([h, w, len(boxes)], dtype='uint8')
      class_ids = list()
      for i in range(len(boxes)):
        box = boxes[i]
        x_coors=box[0]
        y_coors=box[1]
        count=-1
        county=-1
        for x,y in zip(x_coors,y_coors):         
          # if x is in incraesing order then do masking from previous x to current x 
          # else do masking from current x to previous mask
          if int(round(x_coors[1]))>= int(round(x_coors[0])): # For x is in  increasing order , if this cond got true it wont be going in else part
            #print('hello!!  up')  
            if count==-1:  
              masks[int(round(y_coors[county+1])):int(round(y)),int(round(x_coors[count+1])):int(round(x)), i]=1 
              count==0
              county==0
            else:  
              masks[int(round(y_coors[county])):int(round(y)),int(round(x_coors[count])):int(round(x)), i]=1
              count+=1 
              county+=1
            
          else: # for x is in decreasing order 
            if count==-1:   
               masks[int(round(y_coors[county+1])):int(round(y)),int(round(x)):int(round(x_coors[count+1])), i]=1 
               count==0
               county==0
            else:  
               masks[int(round(y_coors[county])):int(round(y)),int(round(x)):int(round(x_coors[count])), i]=1   
               count+=1   
               county+=1  
        class_ids.append(self.class_names.index('primary_root'))
      return masks, asarray(class_ids, dtype='int32')
    
	# load an image reference
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']

## define a configuration for the model
class RootsConfig(Config):
   NAME = "Roots_cfg"
   GPU_COUNT = 1
   IMAGES_PER_GPU = 1
   # define the name of the configuration
   # number of classes (background + roots)
   NUM_CLASSES = 1 + 1
   # number of training steps per epoch
   STEPS_PER_EPOCH = 3796
   VALIDATION_STEPS= 120

dataset_dir='D:\Mayank\BTP_2.0\Object_detection\Root_detection\Roots'
# Trainset
train_set = RootsDataset()
train_set.load_dataset(dataset_dir, is_train=True)
train_set.prepare()

# test/val set
test_set = RootsDataset()
test_set.load_dataset(dataset_dir, is_train=False)
test_set.prepare()

# load the masks and the class ids
mask, class_ids = train_set.load_mask(image_id)

# extract bounding boxes from the masks
bbox = extract_bboxes(mask)

# display image with masks and bounding boxes
display_instances(image, bbox, mask, class_ids, train_set.class_names)

# prepare config
config = RootsConfig()
config.display()
print('config class made')
ROOT_DIR = os.path.abspath("./")
# Directory to save logs and trained model
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs_2")
print(DEFAULT_LOGS_DIR)
###############
print('model making')
# define the model
model = MaskRCNN(mode='training', model_dir="logs", config=config)
# load weights (mscoco) and exclude the output layers
model.load_weights(r"D:\Mayank\BTP_2.0\Object_detection_TF_2-20230707T165828Z-001\Object_detection_TF_2\Mask-RCNN-TF2\Root_detection\mrcnn\mask_rcnn_coco.h5", by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
print('model training started')
# train weights (output layers or 'heads')
model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=20, layers='heads')
model_path = 'root_mask_rcnn_trained.h5'
model.keras_model.save_weights(model_path)
print('model trained')
