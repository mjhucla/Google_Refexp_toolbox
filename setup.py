"""Setup files for Google Refexp Toolbox

This script has the following functions:
1. Download Google Refexp Data
2. Download and compile coco toolbox
3. Download coco annotations
4. Download coco images
5. Align Google Refexp Data with coco annotations
"""

import os
import sys
import json
import numpy as np
import logging
import copy

logger = logging.getLogger('DatasetRefexpGoogle')
FORMAT = "[%(filename)s:line %(lineno)4s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.INFO)

## Step1. Download Google Refexp Data
while True:
  ch = raw_input('Do you want to download Google Refexp Datset? (Y/N)')
  if ch == 'Y':
    os.system('wget https://storage.googleapis.com/refexp/google_refexp_dataset_release.zip')
    os.system('unzip google_refexp_dataset_release.zip')
    os.system('rm -f google_refexp_dataset_release.zip')
    print 'Google Refexp Dataset is now available at ./google_refexp_dataset_release'
  if ch == 'Y' or ch == 'N':
    break
  else:
    print 'Please type Y or N'

## Step2. Download coco toolbox
while True:
  ch = raw_input('Do you want to download and install MS COCO toolbox? (Y/N)')
  if ch == 'Y':
    os.system('cd ./external && git clone https://github.com/pdollar/coco.git')
    os.system('cd ./external/coco/PythonAPI && python setup.py build_ext --inplace')
    print 'COCO toolbox is installed at ./external/coco/'
  if ch == 'Y' or ch == 'N':
    break
  else:
    print 'Please type Y or N'
    
## Step3. Download coco annotations
while True:
  ch = raw_input('Do you want to download MS COCO train2014 annotations? (Y/N)')
  if ch == 'Y':
    os.system('cd ./external/coco && wget http://msvocds.blob.core.windows.net/'
              'annotations-1-0-3/instances_train-val2014.zip && '
              'unzip instances_train-val2014.zip')
    os.system('rm -f ./external/coco/instances_train-val2014.zip')
    print ('COCO train2014 annotations are downloaded at '
           './external/annotations/instances_train2014.json')
  if ch == 'Y' or ch == 'N':
    break
  else:
    print 'Please type Y or N'
    
## Step4. Download coco images
while True:
  ch = raw_input('Do you want to download MS COCO train2014 images? (Y/N)')
  if ch == 'Y':
    # Download images
    os.system('mkdir ./external/coco/images')
    os.system('cd ./external/coco/images && wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip && unzip train2014.zip && rm -f train2014.zip')
    print 'COCO train2014 images are downloaded at ./external/coco/images/train2014/'
  if ch == 'Y' or ch == 'N':
    break
  else:
    print 'Please type Y or N'

## Step5. Align our dataset with MS COCO
# Check whether MS COCO toolbox are installed
assert os.path.isdir('./external/coco/PythonAPI'), ('Please download MS COCO'
    'first (run this script again and choose "Y" for MS COCO downloading)')
assert os.path.isdir('./external/coco/PythonAPI/build'), ('Please install'
    'MS COCO python API first (go to ./external/coco/PythonAPI/ and run setup.py)')
assert os.path.isfile('./external/coco/annotations/instances_train2014.json'), ('Please'
    'download MS COCO train2014 annotation first')
    
# Create a symbolic link from pycocotools to google_refexp_lib
pycoco_dir = os.path.join(os.getcwd(), 'external', 'coco', 'PythonAPI', 'pycocotools')
os.system('cd google_refexp_py_lib && ln -sf %s pycocotools' % pycoco_dir)

# Load MS COCO data
logger.info('Start to load MS COCO file')
sys.path.append('./external/coco/PythonAPI')
from pycocotools.coco import COCO
coco = COCO('./external/coco/annotations/instances_train2014.json')

# Align our dataset with MS COCO
logger.info('Start to align Google Refexp Data with ms coco annotations')
dataset_template = './google_refexp_dataset_release/google_refexp_%s_201511.json'
split_set_names = ('train', 'val')

for set_name in split_set_names:
  logger.info('Start to align Google Refexp %s set with ms coco' % set_name)
  dataset_path_o = dataset_template % set_name
  dataset_path_n = dataset_path_o[:-5] + '_coco_aligned.json'
  with open(dataset_path_o) as fin:
    dataset = json.load(fin)
  
  # The new dataset that concatenates coco and refexp.
  coco_aligned_dataset = {}
  # Maps image id to image object that concatenates the image info from both coco and refexp.
  images = {}
  # Maps annotation id to annotation object that concatenates the annotation info from both coco and refexp.
  annotations = {}
  # Maps refering expression id to refexp object that contains refering expression info from Google refexp.
  refexps = {}
  
  # Annotations from GoogleRefexp.
  googleRefexp_annotations = dataset['annotations']
  googleRefexp_refexps = dataset['refexps']
  
  for ann in googleRefexp_annotations:
    if images.get(ann['image_id'], None) is None:
      # Take image object from coco.
      img_coco = copy.deepcopy(coco.loadImgs(ann['image_id'])[0])
      # Extend with the region candidates info from GoogleRefexp.
      img_coco['region_candidates'] = ann['region_candidates']
      # Change id to more specific image_id
      img_coco['image_id'] = img_coco['id']
      img_coco.pop('id', None)
      # Add the new image object to the dictionary of image ids to image objects.
      images[ann['image_id']] = img_coco
    
    if annotations.get(ann['annotation_id'], None) is None:
      # Take annotation from coco.
      ann_coco = copy.deepcopy(coco.loadAnns(ann['annotation_id'])[0])
      # Change id to more specific annotation_id
      ann_coco['annotation_id'] = ann_coco['id']
      ann_coco.pop('id', None)
      # Extend annotation with the Google refering expression ids.
      ann_coco['refexp_ids'] = ann['refexp_ids']
      # Add the new annotation object to the dictionary of annotation ids to annotation objects.
      annotations[ann['annotation_id']] = ann_coco
    else:
      anno_coco_o = annotations[ann['annotation_id']]
      anno_coco_o['refexp_ids'] += ann['refexp_ids']
  
  # Set key in refering expression dictionary to be the refering expression id.
  for ref in googleRefexp_refexps:
    refexps[ref['refexp_id']] = ref
    
  # Set the images field to the dictionary of images created above.
  coco_aligned_dataset['images']  = images
  # Set the annotations field to the dictionary of annotations created above.
  coco_aligned_dataset['annotations'] = annotations
  # Set the refering expressions field in the dictionary of referering expressions created above.
  coco_aligned_dataset['refexps'] = refexps
  
  # Write data to file.
  with open(dataset_path_n, 'w') as fout:
    json.dump(coco_aligned_dataset, fout)
