import argparse
import os
from copy import deepcopy

import PIL
import torch
import yaml
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file
from torchvision.transforms import ToPILImage, ToTensor
from transformers import AutoModelForImageSegmentation
from utils import extract_object, get_model_from_config, resize_and_center_crop

ASPECT_RATIOS = {
    str(512 / 2048): (512, 2048),
    str(1024 / 1024): (1024, 1024),
    str(2048 / 512): (2048, 512),
    str(896 / 1152): (896, 1152),
    str(1152 / 896): (1152, 896),
    str(512 / 1920): (512, 1920),
    str(640 / 1536): (640, 1536),
    str(768 / 1280): (768, 1280),
    str(1280 / 768): (1280, 768),
    str(1536 / 640): (1536, 640),
    str(1920 / 512): (1920, 512),
}

PATH = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--foreground", type=str, required=True)
parser.add_argument("--background", type=str, required=True)
parser.add_argument("--output_path", type=str, required=True)
parser.add_argument("--num_inference_steps", type=int, default=1)
parser.add_argument(
    "--config_path", type=str, default=os.path.join(PATH, "config/relight.yaml")
)
parser.add_argument(
    "--weights_path", type=str, default=os.path.join(PATH, "config/relight.safetensors")
)

args = parser.parse_args()

birefnet = AutoModelForImageSegmentation.from_pretrained(
    "ZhengPeng7/BiRefNet", trust_remote_code=True
).cuda()
image_size = (1024, 1024)


@torch.no_grad()
def evaluate(
    model,
    fg_image: PIL.Image.Image,
    bg_image: PIL.Image.Image,
    num_sampling_steps: int = 1,
):

    ori_h_bg, ori_w_bg = fg_image.size
    ar_bg = ori_h_bg / ori_w_bg
    closest_ar_bg = min(ASPECT_RATIOS, key=lambda x: abs(float(x) - ar_bg))
    dimensions_bg = ASPECT_RATIOS[closest_ar_bg]

    _, fg_mask = extract_object(birefnet, deepcopy(fg_image))

    fg_image = resize_and_center_crop(fg_image, dimensions_bg[0], dimensions_bg[1])
    fg_mask = resize_and_center_crop(fg_mask, dimensions_bg[0], dimensions_bg[1])
    bg_image = resize_and_center_crop(bg_image, dimensions_bg[0], dimensions_bg[1])

    img_pasted = Image.composite(fg_image, bg_image, fg_mask)

    img_pasted_tensor = ToTensor()(img_pasted).unsqueeze(0) * 2 - 1
    batch = {
        "source_image": img_pasted_tensor.cuda().to(torch.bfloat16),
    }

    z_source = model.vae.encode(batch[model.source_key])

    output_image = model.sample(
        z=z_source,
        num_steps=num_sampling_steps,
        conditioner_inputs=batch,
        max_samples=1,
    ).clamp(-1, 1)

    output_image = (output_image[0].float().cpu() + 1) / 2
    output_image = ToPILImage()(output_image)
    output_image.resize((ori_h_bg, ori_w_bg))

    return fg_image, bg_image, output_image, img_pasted, fg_mask


def main():
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    model = get_model_from_config(**config)

    if os.path.exists(args.weights_path):
        sd = load_file(args.weights_path)
    else:
        # download the weights from HF hub
        sd = hf_hub_download(
            repo_id="jasperai/LBM",
            filename="model.safetensors",
            local_dir=os.path.join(PATH, "config"),
        )
        sd = load_file(sd)

    model.load_state_dict(sd, strict=True)
    model.to("cuda").to(torch.bfloat16)

    fg_image = Image.open(args.foreground).convert("RGB")
    bg_image = Image.open(args.background).convert("RGB")

    fg_image, bg_image, output_image, img_pasted, mask = evaluate(
        model, fg_image, bg_image, args.num_inference_steps
    )

    os.makedirs(args.output_path, exist_ok=True)

    output_image.save(os.path.join(args.output_path, "output_image.jpg"))
    img_pasted.save(os.path.join(args.output_path, "composite.jpg"))


if __name__ == "__main__":
    main()
