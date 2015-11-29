Introduce the format to use the evaluation code in google_refexp_py_lib.eval 
that compares the generated sentences with the groundtruth annotation.

## Format for AMT evaluation of the generation task

The algorithm should output a json file with a list. Each element in the list
should be a dictionary with the following format:

{
"annotation_id": int, (required, MSCOCO object annotation id)
"generated_refexp": string (required, generated referring expressions)
}

It is allowed that there are duplicated "annotation_id" in the list. The script
will treat the corresponding referring expressions independently.

If you want to use the *end-to-end evaluation* (evaluate the performance of the 
generation-comprehension pipeline) as described in [1], please use your generated 
sentences as the input to your comprehension model and evaluate the results under
the comprehension settings.

[1]. Junhua Mao, Jonathan Huang, Alexander Toshev, Oana Camburu, Alan Yuille, and 
Kevin Murphy. "Generation and Comprehension of Unambiguous Object Descriptions." 
arXiv preprint arXiv:1511.02283 (2015).
