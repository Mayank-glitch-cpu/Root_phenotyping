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

# class that defines and loads the root dataset
class RootsDataset(Dataset):
    # load the dataset definitions
    def load_dataset(self, dataset_dir, is_train=True):
        # define classes
        self.add_class("dataset", 1, "primary_root")
        
        # find all image directories in Root Images folder
        # Each subdirectory contains: image_XXXX.jpg, image_XXXX.rsml, metadata.json
        for subdir in sorted(listdir(dataset_dir)):
            subdir_path = os.path.join(dataset_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue
                
            # Look for image files in the subdirectory
            for filename in listdir(subdir_path):
                if filename.endswith('.jpg'):
                    # extract image id from filename (e.g., image_0000.jpg -> 0000)
                    image_id = filename.replace('image_', '').replace('.jpg', '')
                    
                    try:
                        image_id_int = int(image_id)
                    except ValueError:
                        continue
                    
                    # skip all images after 3795 if we are building the train set
                    if is_train and image_id_int >= 3796:
                        continue
                    # skip all images before 3796 if we are building the test/val set
                    if not is_train and image_id_int < 3796:
                        continue
                    
                    img_path = os.path.join(subdir_path, filename)
                    # RSML file has same name but .rsml extension
                    ann_path = os.path.join(subdir_path, filename.replace('.jpg', '.rsml'))
                    
                    # Check if annotation file exists
                    if not os.path.exists(ann_path):
                        print(f"Warning: Annotation file not found for {image_id}: {ann_path}")
                        continue
                    
                    # add to dataset
                    self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path, class_ids=[1])

    # function to return the size of image 
    def sizeOfImage(self,filename):
        # get image
        img = Image.open(filename) 
        # get width and height
        width = img.width
        height = img.height
        # retuns the width and height of the image
        return width, height

    # Extracting the bounding boxes from RSML files
    def extract_bounding_boxes(self, filename):
        bounding_boxes = list()
        tree = ET.parse(filename)
        root = tree.getroot()
        
        # RSML structure: root -> scene -> plant -> root -> geometry -> polyline -> point
        # Find all plants in the scene
        scene = root.find('scene')
        if scene is None:
            print(f"Warning: No scene found in {filename}")
            return bounding_boxes
            
        plants = scene.findall('plant')
        if len(plants) == 0:
            print(f"Warning: No plants found in {filename}")
            return bounding_boxes
            
        for plant in plants:
            # Find all roots in this plant
            roots = plant.findall('root')
            for root_elem in roots:
                geometry = root_elem.find('geometry')
                if geometry is None:
                    continue
                    
                # Get polyline points
                polyline = geometry.find('polyline')
                if polyline is None:
                    continue
                    
                points = polyline.findall('point')
                if len(points) == 0:
                    continue
                
                x_coords = []
                y_coords = []
                
                for point in points:
                    x_coords.append(float(point.attrib['x']))
                    y_coords.append(float(point.attrib['y']))
                
                if len(x_coords) > 0 and len(y_coords) > 0:
                    x_min = min(x_coords)
                    x_max = max(x_coords)
                    y_min = min(y_coords)
                    y_max = max(y_coords)
                    
                    # Store as [x_coords, y_coords, x_min, y_min, x_max, y_max]
                    coors = [x_coords, y_coords, x_min, y_min, x_max, y_max]
                    bounding_boxes.append(coors)
        
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
              count=0
              county=0
            else:  
              masks[int(round(y_coors[county])):int(round(y)),int(round(x_coors[count])):int(round(x)), i]=1
              count+=1 
              county+=1
            
          else: # for x is in decreasing order 
            if count==-1:   
               masks[int(round(y_coors[county+1])):int(round(y)),int(round(x)):int(round(x_coors[count+1])), i]=1 
               count=0
               county=0
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

# Set dataset directory - update this path to match your Root Images directory
ROOT_DIR = os.path.abspath("./")
dataset_dir = os.path.join(ROOT_DIR, "Root Images")

# Trainset
print("Loading training dataset...")
train_set = RootsDataset()
train_set.load_dataset(dataset_dir, is_train=True)
train_set.prepare()
print(f"Training set: {len(train_set.image_ids)} images")

# test/val set
print("Loading validation dataset...")
test_set = RootsDataset()
test_set.load_dataset(dataset_dir, is_train=False)
test_set.prepare()
print(f"Validation set: {len(test_set.image_ids)} images")

# Optional: Visualize a sample image (commented out by default)
# Uncomment the following lines to visualize a sample image with masks
# if len(train_set.image_ids) > 0:
#     image_id = train_set.image_ids[0]
#     # load the masks and the class ids
#     mask, class_ids = train_set.load_mask(image_id)
#     # load the image
#     image = train_set.load_image(image_id)
#     # extract bounding boxes from the masks
#     bbox = extract_bboxes(mask)
#     # display image with masks and bounding boxes
#     display_instances(image, bbox, mask, class_ids, train_set.class_names)

# prepare config
config = RootsConfig()
# Update STEPS_PER_EPOCH based on actual training set size
if len(train_set.image_ids) > 0:
    config.STEPS_PER_EPOCH = len(train_set.image_ids)
if len(test_set.image_ids) > 0:
    config.VALIDATION_STEPS = max(1, len(test_set.image_ids) // 10)
config.display()
print('Config class created')

# Directory to save logs and trained model
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
os.makedirs(DEFAULT_LOGS_DIR, exist_ok=True)
print(f'Logs directory: {DEFAULT_LOGS_DIR}')

###############
print('Creating model...')
# define the model
model = MaskRCNN(mode='training', model_dir=DEFAULT_LOGS_DIR, config=config)

# Load COCO weights - you may need to download this file
# The weights file should be placed in the mrcnn directory or update the path below
coco_weights_path = os.path.join(ROOT_DIR, "mrcnn", "mask_rcnn_coco.h5")
if not os.path.exists(coco_weights_path):
    # Try alternative location
    coco_weights_path = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
    if not os.path.exists(coco_weights_path):
        print(f"Warning: COCO weights file not found at {coco_weights_path}")
        print("Please download mask_rcnn_coco.h5 and place it in the project directory")
        print("Download from: https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5")
        # You can uncomment the following to download automatically:
        # from mrcnn.utils import download_trained_weights
        # download_trained_weights(coco_weights_path)
else:
    print(f'Loading COCO weights from {coco_weights_path}...')
    model.load_weights(coco_weights_path, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
    print('COCO weights loaded')

print('Starting model training...')
# train weights (output layers or 'heads')
model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=20, layers='heads')
model_path = os.path.join(ROOT_DIR, 'root_mask_rcnn_trained.h5')
model.keras_model.save_weights(model_path)
print(f'Model trained and saved to {model_path}')
