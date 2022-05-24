from vit import vit_large_patch32_384, vit_base_patch16_224,vit_lite_depth7_patch4_32,vit_large_patch32_224
from colossalai import launch_from_torch
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
import torch
from typing import Any

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image

config = dict(parallel=dict(pipeline=dict(size=2), tensor=dict(size=1, mode='1d')))

launch_from_torch(config)

def accimage_loader(path: str) -> Any:
    import accimage

    try:
        return accimage.Image(path)
    except OSError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

img = pil_loader('/home/lcdjs/ColossalAI-Inference/examples/vit/dataset/n01667114_9985.JPEG')

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
img = transform(img)



img = torch.unsqueeze(img, 0).half().cuda()

model = vit_large_patch32_224(dtype=torch.half).cuda()

# print(model)

if gpc.is_first_rank(ParallelMode.PIPELINE):
    output = model(img)
    print(type(output))

# print(model)
# print(torch.cuda.memory_allocated())