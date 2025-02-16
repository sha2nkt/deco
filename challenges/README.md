# DAMON Challenges

[Workshop website](https://rhobin-challenge.github.io/) | [Generalized 3D contact prediction]()

This folder provides the evaluation code for the Rhobin Challenges held in conjunction with the CVPR'24 workshop.

- Overview

- About the data

- Submission

- Evaluation

- Citations

## Overview

Understanding how humans use physical contact to interact with the world is a key step toward human-centric
artificial intelligence. While inferring 3D contact is crucial for modeling realistic and physically-plausible
human-object interactions, existing methods either focus on 2D, consider body joints rather than the surface, use
coarse 3D body regions, or do not generalize to in-the-wild images. In this challenge, we wish to examine how
well existing methods can infer dense vertex-level 3D contact on the full body surface from in-the-wild images. The
recently released <a href="https://deco.is.tue.mpg.de/">DAMON dataset</a> enables for the first time, an in-the-wild
benchmark for this task. Based on the DAMON dataset, this competition involves estimating full-body 3D contact from 2D
images.

The winner will be invited to give a talk in our <a href="https://rhobin-challenge.github.io/index.html">
CVPR'24 Rhobin Workshop</a>.

## About the data

The DAMON dataset is used for all the two tracks. DAMON is a collection of vertex-level 3D contact labels on the
SMPL/SMPL-X mesh paired with color images of people in unconstrained environments with a wide diversity of human-scene
and human-object interactions.

Participants are allowed to train their methods on the DAMON training and validation sets, or any other datasets EXCEPT
the DAMON test set. We have mechanisms in place to detect overfitting to the test set, and any such submissions will be
disqualified.

Both direct contact estimation and methods estimating contact by thresholding the geometric distance between
reconstruction human and scene/objects are encouraged to participate.

For convenience, we provide the following links to download the DAMON dataset:

- [DAMON trainval set (x G)](#)
- [DAMON test set images (x G)](#)

By downloading the dataset, you agree to the [DAMON dataset license](https://deco.is.tue.mpg.de/license.html).

## Submission

It is NOT mandatory to submit a report for your method. However, we DO encourage you to fill in [this](#) form about the
additional training data you used.

Each participant is allowed to submit maximum 5 times per day and 100 submissions in total.

Participants must pack their results into one pkl file named as `results.pkl` and submit it as zip file. The pkl data
should be organized as follows:

```code
{
    "image_id": { # image_id is the image name in the DAMON test set

        # results for generalized 3D contact prediction from 2D images (Track 1)
        "gen_contact_vids": [N], # binary contact prediction for each vertex, N is the number of vertices
    },
    ...
}
```

For evaluating semantic (object-wise) contact estimation, the `results.pkl` should be organized as follows:
```code
{
    "image_id": { # image_id is the image name as in the DAMON test set npz. The evaluation will fail if it doesn't match.

        # results for generalized 3D contact prediction from 2D images.
        "sem_contact_vids": {
            obj_1: [vid_1, vid_2, vid_K], # list of SMPL vertices with contact on obj1
            obj_2: [vid_1, vid_2, vid_N], # list of SMPL vertices with contact on obj2
            ...
    },
    ...
}
```
obj_1, obj_2, ... are the object names in the DAMON dataset, with spaces replaced by '_'. We ignore the "supporting" label.
The evaluation will fail if the object names do not match. Please make sure to replace all spaces in object names with underscores.

## Evaluation

We provide evaluation scripts for two tracks:
1. Generalized 3D contact prediction from 2D images: This track evaluates contact on the full body between all human-object and human-scene contacts
2. Semantic (object-wise) contact estimation: This track evaluates per-object contact on the full body, between each human-object pair

You can run evaluation code for generalized 3D contact estimation as follows, specifying the path to the ```RESULT_PKL``` file:

```bash
python evaluate_3d_contact.py --gt_pkl [GT_PKL] --pred_pkl [RESULT_PKL]
```

For example, to evaluate the baseline DECO model, you can run:

```bash 
unzip examples/semantic_contact_pkls.zip -d examples
python evaluate_3d_contact.py --pred_pkl examples/deco_pred_contacts.pkl --gt_pkl examples/deco_gt_contacts.pkl
```

You can run evaluation code for semantic 3D contact estimation as follows, specifying the path to the ```RESULT_PKL``` file:

```bash
python evaluate_semantic_contact.py --gt_pkl [GT_PKL] --pred_pkl [RESULT_PKL]
```

For example, to evaluate the baseline DECO model, you can run:

```bash 
python evaluate_semantic_contact.py --pred_pkl examples/deco_pred_semantic_contacts.pkl --gt_pkl examples/deco_gt_semantic_contacts.pkl
```

## Citations

If you use the code, please cite:

```bibtex
@InProceedings{tripathi2023deco,
    author    = {Tripathi, Shashank and Chatterjee, Agniv and Passy, Jean-Claude and Yi, Hongwei and Tzionas, Dimitrios and Black, Michael J.},
    title     = {{DECO}: Dense Estimation of {3D} Human-Scene Contact In The Wild},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {8001-8013}
}