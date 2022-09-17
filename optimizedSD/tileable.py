# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: © 2022 lox9973

import torch
from torch import autocast
from diffusers import LDMTextToImagePipeline, StableDiffusionPipeline


def patch_conv(klass):
    init = klass.__init__

    def __init__(self, *args, **kwargs):
        return init(self, *args, **kwargs, padding_mode='circular')
    klass.__init__ = __init__


for klass in [torch.nn.Conv2d, torch.nn.ConvTranspose2d]:
    patch_conv(klass)

prompt = "Normal map of polyester fiber"

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16, use_auth_token="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")
pipe.to("cuda")
with autocast("cuda"):
    images = pipe(prompt)["sample"]

prompt = prompt.replace(" ", "_").replace(".", "_")

# save images
for idx, image in enumerate(images):
    image.save(f"./outputs/tiles/{prompt}/image-{idx}.png")
