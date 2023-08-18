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

    #Code to determine the length of Roots back from RSML Files
    #extract bounding boxes from an annotation file
    
    #def distance(x1,x2,y1,y2):
    #    dis=((x2-x1)**2+(y2-y1)**2)**0.5
    #    return dis
    #def length(self,filename):
    #    # Load the RSML file
    #    #tree = ET.parse(rsml_file_path)
    #    #root = tree.getroot()
    #    #print(len(root[1][0])) # returns number of primary roots
    #    #print(root[1][0])
    #    i=0 
    #    root_length=[]
    #    length=0
    #    while i< len(root[1][0]):
    #        #print('*************Root',i+1,'***************')
    #        for j in range(len(root[1][0][i][0][0])-1):
    #          x1= root[1][0][i][0][0][j].attrib['x']
    #          x2= root[1][0][i][0][0][j+1].attrib['x']
    #          y1= root[1][0][i][0][0][j].attrib['y']
    #          y2= root[1][0][i][0][0][j+1].attrib['y']
    #          #print(x1,x2,y1,y2)
    #          length=length+self.distance(float(x1),float(x2),float(y1),float(y2))
    #         # print(length,j)
    #        root_length.append(length)
    #        length=0
    #        i=i+1
    #        
    #    return root_length
    def sizeOfImage(self,filename):
        # get image
        #info= self.image_info[image_id]
        #path= info['path'] 
        img = Image.open(filename) 
        # get width and height
        width = img.width
        height = img.height
          
        # display width and height
        #print("The height of the image is: ", height)
        #print("The width of the image is: ", width)
        return width, height
    def extract_bounding_boxes(self,filename):
        bounding_boxes=list()
        tree = ET.parse(filename)
        root = tree.getroot()
        #print(len(root[1][0])) # returns number of primary roots
        #print(root[1][0])
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
                #width.append(x_max[0]-x_min[0])
                #height.append(y_max[0]-y_min[0])
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
        #image_id = int(filename[:-4])
        #info= self.image_info[filename]
        #image_id= info['image_id']    
          
        return bounding_boxes
    def load_mask(self,image_id):
      info= self.image_info[image_id]
      path= info['annotation']
      #print(path)  
      path_to_image=info['path']
      #print(path_to_image)
      boxes= self.extract_bounding_boxes(path)
      w,h= self.sizeOfImage(path_to_image) 
      #print(boxes)
      masks = zeros([h, w, len(boxes)], dtype='uint8')
      #print(masks.shape)
      #print(len(boxes))
      class_ids = list()
      for i in range(len(boxes)):
        box = boxes[i]
        #print('*************************** box*************************',i)
        #print(box)
        #print(masks[:,:,i].shape,'shape of Mask',i)
        #row_s, row_e = int(box[1]), int(box[3])
        #col_s, col_e = int(box[0]), int(box[2])
        #print(row_s,row_e,col_s,col_e)
        # box[4] will have the name of the class 
        #if (box[4] == 'apple'):
        x_coors=box[0]
        y_coors=box[1]
        count=-1
        county=-1
        for x,y in zip(x_coors,y_coors):         
          #print(x,y)  
          #if x is in incraesing order then do masking from previous x to current x 
          # esle do masking from current x to previous mask
          #masks[int(round(y)),int(round(x)), i] = 1  
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
            #print('hello!! down')  
            if count==-1:   
               masks[int(round(y_coors[county+1])):int(round(y)),int(round(x)):int(round(x_coors[count+1])), i]=1 
               count==0
               county==0
            else:  
               masks[int(round(y_coors[county])):int(round(y)),int(round(x)):int(round(x_coors[count])), i]=1   
               count+=1   
               county+=1  
        class_ids.append(self.class_names.index('primary_root'))
        
        #print(masks[:,:,i].shape,'shape of Mask',i)
      return masks, asarray(class_ids, dtype='int32')
    
        

	# load an image reference
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']

dataset_dir='D:\Mayank\BTP_2.0\Object_detection\Root_detection\Roots'

train_set = RootsDataset()
train_set.load_dataset(dataset_dir, is_train=True)
train_set.prepare()

print('************Train**************: %d' % len(train_set.image_ids))

# test/val set
test_set = RootsDataset()
test_set.load_dataset(dataset_dir, is_train=False)
test_set.prepare()
print('*************Test*****************: %d' % len(test_set.image_ids))
#import random
import cv2 
#num=random.randint(0, len(train_set.image_ids))
# define image id
from PIL import Image 
import PIL 
#folder=r"D:\Mayank\BTP_2.0\Object_detection\Root_detection\Roots\masks"
image_id = 3
#for image_id in range(len(train_set.image_ids)):
#image_id = 155
print(train_set.image_info[image_id])
id=int(train_set.image_info[image_id]['id']) 
#print(id)
## load the image
image = train_set.load_image(image_id)
#print(image_id)

# load the masks and the class ids
mask, class_ids = train_set.load_mask(image_id)
#print(mask.shape[0],class_ids)
# extract bounding boxes from the masks
bbox = extract_bboxes(mask)
#print(mask,'*******mask')
#print(bbox.shape[0],mask.shape[-1],class_ids.shape[0])
#print(image[:,:,0])
# display image with masks and bounding boxes
display_instances(image, bbox, mask, class_ids, train_set.class_names)





## define a configuration for the model
class RootsConfig(Config):
   NAME = "Roots_cfg"
   GPU_COUNT = 1
   IMAGES_PER_GPU = 1
   # define the name of the configuration
   
   # number of classes (background + 3 fruits)
   NUM_CLASSES = 1 + 1
   # number of training steps per epoch
   STEPS_PER_EPOCH = 3796
   VALIDATION_STEPS= 120
# prepare config
config = RootsConfig()
config.display()
print('config class made')
import os
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
import tensorflow as tf
print('model training started')
# train weights (output layers or 'heads')
model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=20, layers='heads')
#model_path = 'root_mask_rcnn_trained.h5'
#model.keras_model.save_weights(model_path)
print('model trained')
