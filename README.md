# TDBi-RSGen
TDBi-RSGen is a text-driven generative framework designed for controllable bi-temporal remote sensing (RS) image synthesis. Our framework leverages natural language instructions to guide the evolution of RS scenes, ensuring precise semantic changes while maintaining absolute background consistency.
Official implementation of the paper: **"Assisting the Change Captioning: A Text-Driven Generative Framework for Controllable Bi-Temporal Remote Sensing Image Synthesis"** (Submitted to IEEE GRSL).

## Usage

### 1. Instruction Generation
Generate automated change instructions using:
`python auto_genetae_text.py`

### 2. Image Synthesis
Generate bi-temporal image pairs based on text prompts:
`python try_text_to_img3.py`

### 3. Or you can use Automated mode
```python
python try_text_to_img3.py \
  --images_dir      "image_dir" \
  --model_path      "models" \
  --controlnet_path "models_ControlNet" \
  --save_dir        "save_dir" \
  --api_key         "……" \
  --n_per_image     3 \
  --max_samples     600`
```
## Requirements
Install dependencies via:
`pip install -r requirements.txt`

##  Acknowledgement
This work is built upon the [HySCDG](https://github.com/yb23/HySCDG) framework. We thank the authors for their excellent work.

## Needed data 
### The provided code is adapted to FLAIR data.

* **FLAIR Dataset**: The primary remote sensing dataset is publicly available at the [FLAIR Project Page](https://ignf.github.io/FLAIR/index.html).
* **Prompts and Footprints**: The auxiliary files, including `FLAIR_Prompts.csv` and `instancesFootprints.pkl`, can be found on [Zenodo](https://zenodo.org/records/15129648).
* **Pre-trained Weights**: The generative model weights (Stable Diffusion + ControlNet) are hosted on Hugging Face at [Yanis236/HySCDG](https://huggingface.co/Yanis236/HySCDG).



