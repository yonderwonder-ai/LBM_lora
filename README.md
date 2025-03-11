# Latent Bridge Matching (LBM)

This repository is the official implementation of the paper [LBM: Latent Bridge Matching for Fast Image-to-Image Translation](http://arxiv.org/abs/2503.07535).

<p align="center">
    <a href="https://arxiv.org/abs/2503.07535">
	    <img src='https://img.shields.io/badge/Paper-2503.07535-green' />
	</a>
    <a href='https://creativecommons.org/licenses/by-nd/4.0/legalcode'>
	    <img src="https://img.shields.io/badge/Licence-CC.BY.NC-purple" />
	</a>
      <a href="https://huggingface.co/spaces/jasperai/LBM_relighting">
	    <img src='https://img.shields.io/badge/%F0%9F%A4%97%20Demo-Object%20Relighting-orange' />
	</a>
<p align="center">
  <img src="assets/relight.jpg" alt="LBM Teaser" width="800"/>
</p>


<!-- link to the demo with link big button -->
<p align="center">
    <a href="https://huggingface.co/spaces/jasperai/LBM_relighting">
	    <b style="font-size: 20px;">DEMO space</b>
	</a>
</p>


## Abstract
In this paper, we introduce Latent Bridge Matching (LBM), a new, versatile and scalable method that relies on Bridge Matching in a latent space to achieve fast image-to-image translation. We show that the method can reach state-of-the-art results for various image-to-image tasks using only a single inference step. In addition to its efficiency, we also demonstrate the versatility of the method across different image translation tasks such as object removal, normal and depth estimation, and object relighting. We also derive a conditional framework of LBM and demonstrate its effectiveness by tackling the tasks of controllable image relighting and shadow generation.

## Setup
To be up and running, you need first to create a virtual env with at least python3.10 installed and activate it

### With venv
```bash
python3.10 -m venv envs/lbm
source envs/lbm/bin/activate
```

### With conda
```bash
conda create -n lbm python=3.10
conda activate lbm
```

Then install the required dependencies and the repo in editable mode

```bash
pip install --upgrade pip
pip install -e .
```

## Inference

We are internally exploring the possibility of releasing the pre-trained models.
<!-- 
We provide in `examples` a simple script to perform image relighting using the proposed method. 

```bash
python examples/inference_relight.py --foreground path_to_your_image.jpg --background path_to_background_image.jpg --output_path output_images
``` -->

## License
This code is released under the Creative Commons BY-NC 4.0 license.

## Citation
If you find this work useful or use it in your research, please consider citing us

@misc{chadebec2025lbm,
      title={LBM: Latent Bridge Matching for Fast Image-to-Image Translation}, 
      author={Clement Chadebec and Onur Tasar and Eyal Benaroche and Benjamin Aubin},
      year={2025},
      eprint={2406.02347},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}