import argparse
import logging
import os
import sys

import torch
from PIL import Image

PATH = os.path.dirname(os.path.abspath(__file__))

# Add the src directory to Python path to find lbm module
src_path = os.path.join(os.path.dirname(os.path.dirname(PATH)), "src")
sys.path.insert(0, src_path)

from lbm.inference import evaluate, get_model

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--source_image", type=str, required=True)
parser.add_argument("--output_path", type=str, required=True)
parser.add_argument("--num_inference_steps", type=int, default=1)
parser.add_argument(
    "--model_name",
    type=str,
    default="normals",
    choices=["normals", "depth", "relighting", "sdxl_lbm_hybrid"],
)


args = parser.parse_args()


def main():
    # download the weights from HF hub
    if not os.path.exists(os.path.join(PATH, "ckpts", f"{args.model_name}")):
        logging.info(f"Downloading {args.model_name} LBM model from HF hub...")
        model = get_model(
            f"jasperai/LBM_{args.model_name}",
            save_dir=os.path.join(PATH, "ckpts", f"{args.model_name}"),
            torch_dtype=torch.bfloat16,
            device="cuda",
        )

    else:
        model_dir = os.path.join(PATH, "ckpts", f"{args.model_name}")
        logging.info(f"Loading {args.model_name} LBM model from local...")
        model = get_model(model_dir, torch_dtype=torch.bfloat16, device="cuda")

    source_image = Image.open(args.source_image).convert("RGB")

    output_image = evaluate(model, source_image, args.num_inference_steps)

    os.makedirs(args.output_path, exist_ok=True)

    source_image.save(os.path.join(args.output_path, "source_image.jpg"))
    output_image.save(os.path.join(args.output_path, "output_image.jpg"))


if __name__ == "__main__":
    main()
