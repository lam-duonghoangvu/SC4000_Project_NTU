"""Frozen DINOv3 global descriptor, computed for the 4 rot90 versions of every image."""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from sc4000.config import ROTATION_COUNT

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Dinov3GlobalDescriptor:
    """Descriptor = [L2(CLS token), L2(mean of patch tokens)] / sqrt(2), so it is unit length.

    The backbone is frozen. Images keep their aspect ratio: the long side is resized to
    `resize_long_side` and both sides are rounded to a multiple of the patch size.
    """

    def __init__(self, model_dir_or_hub_id: str, device: str, resize_long_side: int = 448):
        from transformers import AutoModel  # DINOv3 needs transformers >= 4.56

        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.model = AutoModel.from_pretrained(model_dir_or_hub_id).eval().to(self.device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.patch_size = int(getattr(self.model.config, "patch_size", 16))
        self.num_register_tokens = int(getattr(self.model.config, "num_register_tokens", 0))
        self.resize_long_side = resize_long_side
        self.descriptor_dimension = 2 * int(self.model.config.hidden_size)

    def _load_normalized_image_tensor(self, image_path: Path) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        scale = self.resize_long_side / max(image.size)
        new_width = max(self.patch_size, round(image.width * scale / self.patch_size) * self.patch_size)
        new_height = max(self.patch_size, round(image.height * scale / self.patch_size) * self.patch_size)
        image = image.resize((new_width, new_height), Image.BICUBIC)
        tensor = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1)
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        return (tensor - mean) / std

    @torch.no_grad()
    def _describe_batch(self, image_batch: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
            tokens = self.model(pixel_values=image_batch.to(self.device)).last_hidden_state.float()
        cls_token = F.normalize(tokens[:, 0], dim=-1)
        patch_tokens = tokens[:, 1 + self.num_register_tokens:]
        mean_patch_token = F.normalize(patch_tokens.mean(dim=1), dim=-1)
        return torch.cat([cls_token, mean_patch_token], dim=-1) / np.sqrt(2.0)

    @torch.no_grad()
    def describe_image_in_four_rotations(self, image_path: Path) -> np.ndarray:
        """Returns float32 [ROTATION_COUNT, descriptor_dimension]; row k describes np.rot90(image, k)."""
        image_tensor = self._load_normalized_image_tensor(image_path)
        rotated = [torch.rot90(image_tensor, k, dims=(1, 2)) for k in range(ROTATION_COUNT)]
        # rotations 0/2 and 1/3 share a shape, so they can be batched together
        descriptors_even = self._describe_batch(torch.stack([rotated[0], rotated[2]]))
        descriptors_odd = self._describe_batch(torch.stack([rotated[1], rotated[3]]))
        descriptors = torch.stack([descriptors_even[0], descriptors_odd[0], descriptors_even[1], descriptors_odd[1]])
        return descriptors.cpu().numpy().astype(np.float32)
