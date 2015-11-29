"""Python class for the evaluation of Google Refexp dataset.

This script contains two python classes:
1. GoogleRefexpEvalComprehension
  -  Use precision@k score to evaluate comprehension task performance
  -  Can evaluate generation task through an end-to-end way
2. GoogleRefexpEvalGeneration
  -  Use Amazon Mechanical Turker (AMT) to compare generated refexps with GT
     with the following steps (step a, c, f covered by the class):
     a. Generate csv files for AMT
     b. Generate images and masked images
     c. Upload these images to a server (e.g. Amazon S3) so that the image are 
        publicly accessible
     d. Create a AMT project with the interface at 
        ./cache_evaluation/AMT_interface/AMT_template_generated_vs_GT.html
     e. Upload csv files and start AMT job
     f. Download annotated json file and calculate the score

TO CHECK:
GoogleRefexp.getAnnIds(): get COCO object ids
GoogleRefexp.getRefexpIds(): get referring expression ids
GoogleRefexp.getRefexpAnns(): get google refexp annotations for a list of annotation_id
GoogleRefexp.getGtBoxes(): currently assume a dictionary with key of id, value of a list for bbox

TODO:
Comprehention:
-  A script that can visualize predicted bboxes whose iou satistied a constrain
"""

import json
import os
import copy
import random
import sys
import numpy
import csv
from scipy import misc
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

from refexp import Refexp # Need to check - change
import common_utils as cu

class RefexpEvalComprehension(object):
  def __init__(self, refexp_dataset_path, coco_data_path):
    """Constructor for GoogleRefexpEvalComprehension class for evaluation.
    
    Args:
      refexp_dataset_path: path for the Google Refexp dataset file
      coco_data_path: path for the original coco dataset file (e.g. 'instances_train2014.json')
    """
    # handle refexp dataset file
    assert refexp_dataset_path, "Refexp dataset file missing!"
    self.refexp_dataset_path = refexp_dataset_path
    print 'Loading Google Refexp dataset file for the comprehension task.'
    self.refexp_dataset = Refexp(refexp_dataset_path, coco_data_path) # Need to check - change
    self.gt_ann_ids_set = frozenset(self.refexp_dataset.getAnnIds()) # Need to check - change
    self.gt_refexp_ids_set = frozenset(self.refexp_dataset.getRefexpIds()) # Need to check - change
    
    # reset evaluation state
    self.reset_eval_state()
    
  def reset_eval_state(self):
    """Reset evaluation state."""
    self.pred_results_path = None
    self.pred_results = None
    self.flag_already_eval = False
    
  def evaluate(self, pred_results_path,
               thresh_iou=0.5,
               thresh_k=1,
               flag_ignore_non_existed_object=False,
               flag_ignore_non_existed_gt_refexp=False,
               flag_missing_objects_verbose=False,
               flag_missing_refexps_verbose=False):
    """Evaluate the predicted results for the comprehension task.
    
    Args:
      pred_results_path: path for the predicted results with the format
          described in ./cache_evaluation/format_comprehension_eval.md
      thresh_iou: threshold of the IoU ratio of the evaluation
      thresh_k: precision@k
      flag_ignore_non_existed_object: if set True, the evaluation process
          continues with an warning when encountered non existed objects in 
          self.refexp_dataset. Otherwise stops.
      flag_ignore_non_existed_gt_refexp: if set True, the evaluation process  
          continues when encountered non existed GT referring expressions.
          Otherwise stops.
      flag_missing_objects_verbose: if set true, will list the ids of all the 
          missing objects in self.refexp_dataset
      flag_missing_refexps_verbose: if set true, will list the ids of all the 
          missing referring expressions in self.refexp_dataset
          
    Returns:
      A two element tuple. The first element is precision@k. The second
      element is the predicted results (a dictionary) with an added field
      'best_iou' of the best iou for the top k bounding boxes.
    """
    # Load predicted results
    self.reset_eval_state()
    print 'Loading predicted result file for the comprehension task.'
    with open(pred_results_path) as fin:
      self.pred_results = json.load(fin)
    
    # evaluation
    pred_ann_ids_set = set()
    pred_refexp_ids_set = set()
    score = 0.0
    num_valid_pred = 0
    for pred_elem in self.pred_results:
      # validate the predicted results
      assert 'annotation_id' in pred_elem, 'Object annotation id missing!'
      assert 'predicted_bounding_boxes' in pred_elem, \
             'list of predicted bounding boxes missing!'
      ann_id = pred_elem['annotation_id']
      gt_bbox = self._get_GT_bbox_with_annotation_id(ann_id) # Need to check - change
      if gt_bbox is None:
        if flag_ignore_non_existed_object:
          print ('Ignore COCO annotation id %d which does not exist in '
                 'Refexp dataset file for evaluation' % ann_id)
          pred_elem['best_iou'] = 0.0
          continue
        else:
          print ('COCO annotation id %d does not exist in Refexp '
                 'dataset file for evaluation!' % ann_id)
          raise
      if ('refexp_id' in pred_elem) and not(pred_elem['refexp_id'] in self.gt_refexp_ids_set):
        if flag_ignore_non_existed_gt_refexp:
          print ('Ignore refexp id %d which does not exist in '
                 'Refexp dataset file for evaluation' % pred_elem['refexp_id'])
          pred_elem['best_iou'] = 0.0
          continue
        else:
          print ('refexp id %d does not exist in Refexp '
                 'dataset file for evaluation!' % pred_elem['refexp_id'])
          raise
      pred_ann_ids_set.add(ann_id)
      if 'refexp_id' in pred_elem:
        pred_refexp_ids_set.add(pred_elem['refexp_id'])
      num_valid_pred += 1
          
      # check whether it is a correct prediction
      pred_bboxes = pred_elem['predicted_bounding_boxes']
      best_iou = 0.0
      for k in xrange(min(thresh_k, len(pred_bboxes))):
        iou = cu.iou_bboxes(pred_bboxes[k], gt_bbox)
        best_iou = max(best_iou, iou)
      if best_iou >= thresh_iou:
        score += 1.0
      pred_elem['best_iou'] = best_iou
    score /= num_valid_pred
      
    # warning for missing objects and refexps
    gt_ann_ids_left_set = self.gt_ann_ids_set - pred_ann_ids_set
    gt_refexp_ids_left_set = self.gt_refexp_ids_set - pred_refexp_ids_set
    if gt_ann_ids_left_set:
      print ('Missing %d objects in the refexp dataset file in the predicted '
             'file' % len(gt_ann_ids_left_set))
      if flag_missing_objects_verbose:
        print ('The missing object annotation ids are:')
        print gt_ann_ids_left_set  # TODO pretty print format
    if gt_refexp_ids_left_set:
      print ('Missing %d refexps in the refexp dataset file in the predicted '
             'file' % len(gt_refexp_ids_left_set))
      if flag_missing_refexps_verbose:
        print ('The missing refexp ids are:')
        print gt_refexp_ids_left_set  # TODO pretty print format
      
    # summarize the results
    print 'The average prec@%d score is %.3f' % (thresh_k, score)
    return (score, self.pred_results)
    
  def _get_GT_bbox_with_annotation_id(self, ann_id):
    if not ann_id in self.gt_ann_ids_set:
      return None
    anns = self.refexp_dataset.loadAnns(ids = [ann_id])
    if len(anns) == 0:
      return None
    assert len(anns) == 1
    return anns[0]['bbox']
    
  def visualize_top_predicted_bbox(self, pred_sample, coco_image_dir):
    """Visualize the top predicted bounding box."""
    assert 'annotation_id' in pred_sample, 'Object annotation id missing!'
    assert 'predicted_bounding_boxes' in pred_sample, \
           'list of predicted bounding boxes missing!'
    if not pred_sample['predicted_bounding_boxes']:
      print 'Empty predicted bounding boxes.'
      return
      
    bbox_pred_top = pred_sample['predicted_bounding_boxes'][0]
    ann_id = pred_sample['annotation_id']
    ann = self.refexp_dataset.loadAnns(ids=[ann_id])[0]
    image_id = ann['image_id']
    img_coco = self.refexp_dataset.loadImgs(ids=[image_id])[0]
    iou = cu.iou_bboxes(bbox_pred_top, ann['bbox'])
    
    if 'refexp' in pred_sample or 'refexp_id' in pred_sample:
      print 'The Referring expression input to the model is:'
      if 'refexp' in pred_sample:
        print '  ' + pred_sample['refexp']
      else:
        refexp_tmp = self.refexp_dataset.loadRefexps(ids=pred_sample['refexp_id'])[0]
        print '  ' + refexp_tmp['raw']
    
    I = misc.imread(os.path.join(coco_image_dir, (img_coco['file_name'])))
    ax = plt.imshow(I)
    ax = plt.axis('off')
    ax = plt.title('IoU: %.3f, green bbox: GT, red bbox: predicted' % iou)
    cu.draw_bbox(plt.gca(), ann['bbox'], edge_color='green')
    cu.draw_bbox(plt.gca(), bbox_pred_top, edge_color='red')
    
    
class RefexpEvalGeneration(object):
  def __init__(self, refexp_dataset_path, coco_data_path):
    """Constructor for GoogleRefexpEvalGeneration class for evaluation.
    
    Args:
      refexp_dataset_path: path for the Google Refexp dataset file
    """
    # handle refexp dataset file
    assert refexp_dataset_path, "Refexp dataset file missing!"
    self.refexp_dataset_path = refexp_dataset_path
    print 'Loading Google Refexp dataset file for the generation task.'
    self.refexp_dataset = Refexp(refexp_dataset_path, coco_data_path) # Need to check - change
    self.gt_ann_ids_set = frozenset(self.refexp_dataset.getAnnIds()) # Need to check - change
    
  def generate_AMT_csv_and_images(self, pred_results_path, 
                                  public_image_url_prefix,
                                  AMT_csv_path,
                                  num_refexp_group=5,
                                  flag_generate_images=True,
                                  coco_image_dir=None,
                                  local_image_dir=None):
    """Generate a csv file and images for AMT evaluation.
    
    Args:
      pred_results_path: path for the predicted results with the format
          described in ./cache_evaluation/format_generation_eval.md
      public_image_url_prefix: image url prefix for the publicly accessible
          images. AMTurkers should be able to access images with this url prefix
          (see details in README.md, AMT section)
      AMT_csv_path: path for the generated csv file.
      num_refexp_group: the number of referring expressions that we plan to
          group as one HIT for AMT. default=5 (highly recommended, otherwise 
          need to change AMT_interface)
      flag_generate_images: if set true, will generate images for AMT
      coco_image_dir: directory that coco images can be found, e.g. 
          ./external/coco/images/train2014
      local_image_dir: directory to save the images locally.
    """
    # Load predicted results
    print 'Loading predicted result file for the generation task.'
    with open(pred_results_path) as fin:
      self.pred_results = json.load(fin)
    assert len(self.pred_results) % num_refexp_group == 0, ('The number of '
        'generated sentences should be a multiple of num of images in the'
        'AMT group (i.e. %d)' % num_refexp_group)
    
    # Generate csv file for AMT
    pred_ann_ids = self._generate_AMT_csv_file(
        AMT_csv_path, public_image_url_prefix, 
        num_refexp_group=num_refexp_group)
    
    # Generate images for AMT if necessary
    if flag_generate_images:
      assert coco_image_dir, 'Missing the directory of original coco image'
      assert local_image_dir, 'Missing the local directory for storing images'
      self._generate_images_for_AMT(pred_ann_ids, 
          coco_image_dir=coco_image_dir, local_image_dir=local_image_dir)
          
  def parse_AMT_results(self, csv_download_path, num_refexp_group=5):
    """Parse the AMT results from the downloaded csv file.
    
    Args:
      csv_download_path: the path of the downloaded csv result file from AMT.
      num_refexp_group: the number of the refexp grouped in a HIT.
      
    Return:
      A tuple with two numbers. They represent the ratio of the generated 
      referring expressions are considered to be better and similar 
      respectively.
    """
    num_better = 0
    num_similar = 0
    num_row = 0
    with open(csv_download_path) as fin:
      reader = csv.DictReader(fin)
      for row in reader:
        for ind in xrange(num_refexp_group):
          key = 'Answer.choice_%d' % ind
          if row[key] == 'GEN':
            num_better += 1
          elif row[key] == 'similar':
            num_similar += 1
        num_row += 1
    ratio_better = num_better / float(num_row * num_refexp_group)
    ratio_similar = num_similar / float(num_row * num_refexp_group)
    print ('%.4f of the generated referring expressions are considered to be '
        'better than humans (groundtruth)' % ratio_better)
    print ('%.4f of the generated referring expressions are considered to be '
        'similar to humans (groundtruth)' % ratio_similar)
    return (ratio_better, ratio_similar)
          
  def _generate_AMT_csv_file(self, AMT_csv_path, public_image_url_prefix, 
                             num_refexp_group=5):
    """Private function to generate csv file for AMT."""
    print 'Start to generate csv file to upload to AMT'
    fieldnames_template = ['image_url_o_%d', 'image_url_mask_%d',
                           'descrip_type_%d_0', 'descrip_type_%d_1',
                           'descrip_%d_0', 'descrip_%d_1']
    
    pred_ann_ids = []
    ind_cur = 0
    with open(AMT_csv_path, 'w') as fout:
      while ind_cur < len(self.pred_results):
        dct_row = {}
        fields_all = []
        for ind_group in xrange(num_refexp_group):
          # check pred_result format
          pred_elem = self.pred_results[ind_cur]
          assert 'annotation_id' in pred_elem, 'Object annotation id missing!'
          assert 'generated_refexp' in pred_elem, 'Generated refexp missing!'
          pred_ann_id = pred_elem['annotation_id']
          # load GT data
          assert pred_ann_id in self.gt_ann_ids_set, ('Cannot find object with'
              'annotation id %d' % pred_ann_id)
          gt_data = self.refexp_dataset.loadAnns(ids = [pred_ann_id])[0]  # Need to check - change
          gt_refexps = self.refexp_dataset.loadRefexps(ids = gt_data['refexp_ids'])  # Need to check - change
          pred_ann_ids.append(pred_ann_id)
          # add fieldnames
          for field_template in fieldnames_template:
            fields_all.append(field_template % ind_group)
          # add image urls
          img_name = 'coco_%d.jpg' % gt_data['image_id']
          img_mask_name = 'coco_%d_ann_%d_masked.jpg' % (gt_data['image_id'], pred_ann_id)
          dct_row['image_url_o_%d' % ind_group] = public_image_url_prefix + img_name
          dct_row['image_url_mask_%d' % ind_group] = public_image_url_prefix + img_mask_name
          # get refexp and type, shuffle them (refexp, type)
          descrip_gen = (pred_elem['generated_refexp'], 'GEN')
          descrip_gt = (' '.join(gt_refexps[0]['tokens']), 'GT')  # Need to check - change
          list_descrip = [descrip_gen, descrip_gt]
          random.shuffle(list_descrip)
          for ind in xrange(2):
            dct_row['descrip_%d_%d' % (ind_group, ind)] = list_descrip[ind][0]
            dct_row['descrip_type_%d_%d' % (ind_group, ind)] = list_descrip[ind][1]
          ind_cur += 1
          
        # write row to csv files
        assert len(dct_row) == len(fields_all)
        if ind_cur == num_refexp_group:
          writer = csv.DictWriter(fout, fieldnames=fields_all)
          writer.writeheader()
        writer.writerow(dct_row)
      print 'Finished to generate the csv file'
    return pred_ann_ids
    
  def _generate_images_for_AMT(self, pred_ann_ids, 
                               coco_image_dir=None, local_image_dir=None):
    """Private function to generated images to upload to AMT."""
    assert coco_image_dir and local_image_dir
    assert os.path.isdir(coco_image_dir)
    if not os.path.isdir(local_image_dir):
      print 'Input local image directory does not exist, create it'
      os.makedirs(local_image_dir)
    
    print 'Start to generate images for AMT in local hard disk'
    image_ids_saved = set()
    for (ind, pred_ann_id) in enumerate(pred_ann_ids):
      gt_data = self.refexp_dataset.loadAnns(ids = [pred_ann_id])[0]  # Need to check - change
      img = self._read_image(coco_image_dir, gt_data)
      mask = self._load_mask(gt_data)
      masked_img = cu.apply_mask_to_image(img, mask)
      masked_img_path = os.path.join(local_image_dir, ('coco_%d_ann_%d'
          '_masked.jpg' % (gt_data['image_id'], pred_ann_id)))
      misc.imsave(masked_img_path, masked_img)
      if not gt_data['image_id'] in image_ids_saved:
        image_ids_saved.add(gt_data['image_id'])
        img_path = os.path.join(local_image_dir, 'coco_%d.jpg' % gt_data['image_id'])
        misc.imsave(img_path, img)
    print ('Images generated in local hard disk, please make sure to make them '
           'publicly available online.')
          
  def _read_image(self, coco_image_dir, gt_data):
    """Private function to read an original coco image."""
    img_coco = self.refexp_dataset.loadImgs(ids=gt_data['image_id'])[0]
    return misc.imread(os.path.join(coco_image_dir, img_coco['file_name']))
    
  def _load_mask(self, gt_data):
    """Private function to load the mask of a coco object."""
    img_coco = self.refexp_dataset.loadImgs(ids=gt_data['image_id'])[0]
    mask = Image.new('L', (img_coco['width'], img_coco['height']), 0)
    for seg in gt_data['segmentation']:
      ImageDraw.Draw(mask).polygon(seg, outline='white', fill='white')
    return numpy.asarray(mask)
