Introduce the format to use the evaluation code in google_refexp_py_lib.eval 
to output the prec@k score of the bounding boxes of your method.

## format for the prec@1 evaluation of the comprehension task

The algorithm should output a json file with a list. Each element in the list
should be a dictionary with the following format:

{
"annotation_id" : int, (required, MSCOCO object annotation id)
"predicted_bounding_boxes": obj, (required, list of predicted bounding boxes)
"refexp_id" : int, (optional)
"refexp": string, (optional)
}

-  "annotation_id" denotes the original MS COCO object annotation id for an 
object in Google Refexp dataset.
-  "predicted_bounding_boxes" is list of bounding boxes. Each bounding box is 
represented as a 4-int list: \[x, y, w, h\]. (x, y) is the coordinates of the 
top left corner of the bounding box. w and h are the width and height of the
bounding box respectively. The order of the bounding box in this list 
should reflect the method's confident level. E.g. The evaluation code will treat 
the first bounding box as the most probrable bounding box output by the algorithm.
-  "refexp_id" is the id for the groundtruth referring expression in the Google 
Refexp Dataset. It will be used for the visualization of the results. If the
predicted bounding box does not cooresponding to a groundtruth referring
expression, please do NOT add this field.
-  "refexp" is the string for the referring expression that input to the comprehension
model. If you want to visualize the results (e.g. see for which referring expression
your model performs badly), you should have either "refexp_id" (if the refexp is a 
groundtruth one) or "refexp" (if the refexp is a customized or generated one).
 
