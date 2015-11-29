"""Common util functions.

All the bounding boxes in this script should be represented by a 4-int list: 
\[x, y, w, h\]. 
(x, y) is the coordinates of the top left corner of the bounding box.
w and h are the width and height of the bounding box respectively.
"""

import numpy

def iou_bboxes(bbox1, bbox2):
  """Standard intersection over Union ratio between two bounding boxes."""
  bbox_ov_x = max(bbox1[0], bbox2[0])
  bbox_ov_y = max(bbox1[1], bbox2[1])
  bbox_ov_w = min(bbox1[0] + bbox1[2] - 1, bbox2[0] + bbox2[2] - 1) - bbox_ov_x + 1
  bbox_ov_h = min(bbox1[1] + bbox1[3] - 1, bbox2[1] + bbox2[3] - 1) - bbox_ov_y + 1
    
  area1 = area_bbox(bbox1)
  area2 = area_bbox(bbox2)
  area_o = area_bbox([bbox_ov_x, bbox_ov_y, bbox_ov_w, bbox_ov_h])
  area_u = area1 + area2 - area_o
  if area_u < 0.000001:
    return 0.0
  else:
    return area_o / area_u
      
def area_bbox(bbox):
  """Return the area of a bounding box."""
  if bbox[2] <= 0 or bbox[3] <= 0:
    return 0.0
  return float(bbox[2]) * float(bbox[3])
  
def apply_mask_to_image(img_o, mask, factor_mask = 0.5, channel_mask = 0):
  """Apply a transparent mask to an image."""
  if len(img_o.shape) == 2: # grey-scale image
    img_o_c = numpy.zeros(tuple(list(img_o.shape) + [3]), numpy.uint8)
    for i in xrange(3):
      img_o_c[:, :, i] = img_o
    img_o = img_o_c
  img_mask = numpy.zeros(img_o.shape, numpy.uint8)
  img_mask[:, :, channel_mask] = mask
  img_t = img_o * (1.0 - factor_mask) + img_mask * factor_mask
  img_t = img_t.astype(img_o.dtype)
  return img_t
  
def draw_bbox(ax, bbox, edge_color='red', line_width=3):
    """Draw one bounding box on a matplotlib axis object (ax)."""
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    
    bbox_plot = mpatches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
        fill=False, edgecolor=edge_color, linewidth=line_width)
    ax.add_patch(bbox_plot)
