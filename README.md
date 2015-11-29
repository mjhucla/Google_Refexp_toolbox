# Google Refexp (Referring Expressions) dataset toolbox

This toolbox provide **visualization** and **evaluation** tools for the 
[Google Refexp dataset](#google_refexp). 
It also provide a simple script (i.e. *setup.py*) that automatically
downloads all the necessary data and packages, and *aligns* the Google Refexp
dataset with the [MS COCO] dataset.

## Google Refexp dataset <a name="google_refexp"></a>

The Google RefExp dataset is a collection of text descriptions of objects in 
images which builds on the publicly available [MS-COCO](http://mscoco.org/) 
dataset. Where the image captions in MS-COCO apply to the entire image, this 
dataset focuses on region specific descriptions --- particularly text 
descriptions that allow one to uniquely identify a single object or region 
within an image.

See more details of the collection of the dataset in this paper: [Generation and Comprehension of Unambiguous Object Descriptions](http://arxiv.org/abs/1511.02283)

## Requirements
- python 2.7 (Need numpy, scipy, matlabplot, PIL packages. All included in 
[Anaconda](https://store.continuum.io/cshop/anaconda/))

## Setup and data downloading

### Easy setup and dataset downloading

  ```
  cd $YOUR_PATH_TO_THIS_TOOLBOX
  python setup.py
  ```
  
Running the setup.py script will do the following five things:
1. Download Google Refexp Data
2. Download and compile the COCO toolbox
3. Download COCO annotations
4. Download COCO images
5. Align the Google Refexp Data with COCO annotations

At each step you will be prompted by keyboard whether or not you would like to 
skip a step.
Note that the MS COCO images (13GB) and annotations (158MB) are very large and 
it takes some time to download all of them. 

### Manual downloading and setup (proficient users of MS COCO only)

If you have already played with MS COCO and do not want to have two copies of 
them, you can choose to create a symbolic link from external to your COCO  
toolkit. E.g. 

  ```
  cd $YOUR_PATH_TO_THIS_TOOLBOX
  ln -sf $YOUR_PATH_TO_COCO_TOOLBOX ./external/coco
  ```

Please make sure that the algorithm can find the compiled PythonAPI at 
./external/coco/PythonAPI, the annotation file at 
./external/coco/annotations/instances_train2014.json, and the COCO images at 
./external/coco/images/train2014/. You can create symbolic links if you have 
already downloaded the data and compiled the coco toolbox.

Then run *setup.py* to download the Google Refexp data and compile this toolbox. 
You can skip steps 2, 3, 4.

## Demos

For visualization and utility functions, please see 
**google_refexp_dataset_demo.ipynb**.

For automatic and Amazon Mechanical Turk (AMT) evaluation of the comprehension 
and generation tasks, please see **google_refexp_eval_demo.ipynb**; The 
appropriate output format for a comprehension/generation algorithm is described 
in ./evaluation/format_comprehension_eval.md and 
./evaluation/format_generation_eval.md

We also provide two sample outputs for reference. For the comprehension task, 
we use a naive baseline which is a random shuffle of the region candidates 
(./evaluation/sample_results/sample_results_comprehension.json). For the 
generation task, we use a naive baseline which outputs the class name of an 
object (./evaluation/sample_results/sample_results_generation.json).

If you are not familiar with AMT evaluations, please see this 
[tutorial](http://docs.aws.amazon.com/AWSMechTurk/latest/RequesterUI/amt-ui.pdf)
The interface and APIs provided by this toolbox have already grouped 5 
evaluations into one HIT. In our experiment, paying 2 cents for one HIT leads to 
reasonable results.


## Citation

If you find the dataset and toolbox useful in your research, 
please consider citing:

    @article{mao2015generation,
      title={Generation and Comprehension of Unambiguous Object Descriptions},
      author={Mao, Junhua and Huang, Jonathan and Toshev, Alexander and Camburu, Oana and Yuille, Alan and Murphy, Kevin},
      journal={arXiv preprint arXiv:1511.02283},
      year={2015}
    }
    
## Toolbox Developers

[Junhua Mao](mjhustc@ucla.edu) and [Oana Camburu](oana-maria.camburu@cs.ox.ac.uk)
