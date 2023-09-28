# DECO: Dense Estimation of 3D Human-Scene COntact in the Wild

> Code repository for the paper:  
> [**DECO: Dense Estimation of 3D Human-Scene COntact in the Wild**](https://openaccess.thecvf.com/content/ICCV2023/html/Tripathi_DECO_Dense_Estimation_of_3D_Human-Scene_Contact_In_The_Wild_ICCV_2023_paper.html)  
> [Shashank Tripathi](https://sha2nkt.github.io/), [Agniv Chatterjee](https://ac5113.github.io/), [Jean-Claude Passy](https://is.mpg.de/person/jpassy), [Hongwei Yi](https://xyyhw.top/), [Dimitrios Tzionas](https://ps.is.mpg.de/person/dtzionas), [Michael J. Black](https://ps.is.mpg.de/person/black)

> *IEEE International Conference on Computer Vision (ICCV), 2023*

[![arXiv](https://img.shields.io/badge/arXiv-2305.20091-00ff00.svg)]()  [![Website shields.io](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)](https://deco.is.tue.mpg.de/)     [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]()  [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)]()

![teaser](assets/teaser.png)

## Run demo on images
The following command will run DECO on all images in the specified `--img_src`, and save rendering and colored mesh in `--out_dir`. The `--model_path` flag is used to specify the specific checkpoint being used. Additionally, the base mesh color and the color of predicted contact annotation can be specified using the `--mesh_colour` and `--annot_colour` flags respectively. 
```bash
python inference.py \
    --img_src example_images \
    --out_dir demo_out \
```

## Training
Download the [training data]() to `./data/Datasets/` and place them as outlined in the directory structure, then start training using the following command:
```
python train.py --cfg cfg_hot.yml
```
Checkpoints and logs will be saved to `./logs/`.

## Evaluation
To run evaluation, run the following snippet

```bash
python tester.py --cfg cfg_test.yml
```

## Citing
If you find this code useful for your research, please consider citing the following paper:

```bibtex
@InProceedings{Tripathi_2023_ICCV,
    author    = {Tripathi, Shashank and Chatterjee, Agniv and Passy, Jean-Claude and Yi, Hongwei and Tzionas, Dimitrios and Black, Michael J.},
    title     = {DECO: Dense Estimation of 3D Human-Scene Contact In The Wild},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {8001-8013}
}
```

### License

See [LICENSE](LICENSE).

### Acknowledgments

We sincerely thank Alpar Cseke for his contributions to DAMON data collection and PHOSA evaluations, Sai K. Dwivedi for facilitating PROX downstream experiments, Xianghui Xie for his generous help with CHORE evaluations, Lea Muller for her help in initiating the contact annotation tool, Chun-Hao P. Huang for RICH discussions and Yixin Chen for details about the HOT paper. We are grateful to Mengqin Xue and Zhenyu Lou for their collaboration in BEHAVE evaluations, Joachim Tesch and Nikos Athanasiou for insightful visualization advice, and Tsvetelina Alexiadis for valuable data collection guidance. Their invaluable contributions enriched this research significantly. We also thank Benjamin Pellkofer for help with the website and IT support. This work was funded by the International Max Planck Research School for Intelligent Systems (IMPRS-IS).

### Contact

For technical questions, please create an issue. For other questions, please contact `deco@tue.mpg.de`.

For commercial licensing, please contact `ps-licensing@tue.mpg.de`.
