# DECO: Dense Estimation of 3D Human-Scene COntact in the Wild
Code repository for the paper:
**DECO: Dense Estimation of 3D Human-Scene COntact in the Wild**

[Shashank Tripathi](https://sha2nkt.github.io/), [Agniv Chatterjee](https://ac5113.github.io/), [Jean-Claude Passy](https://is.mpg.de/person/jpassy), [Hongwei Yi](https://xyyhw.top/), [Dimitrios Tzionas](https://ps.is.mpg.de/person/dtzionas), [Michael J. Black](https://ps.is.mpg.de/person/black)

[![arXiv](https://img.shields.io/badge/arXiv-2305.20091-00ff00.svg)]()  [![Website shields.io](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)](https://deco.is.tue.mpg.de/)     [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]()  [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)]()

![teaser](assets/teaser.png)

## Run demo on images
The following command will run DECO on all images in the specified `--img_src`, and save rendering and colored mesh in `--out_dir`. The `--model_path` flag is used to specify the specific checkpoint being used. Additionally, the base mesh color and the color of predicted contact annotation can be specified using the `--mesh_colour` and `--annot_colour` flags respectively. 
```bash
python inference.py \
    --img_src example_images \
    --out_dir demo_out \
```