import torch
import numpy as np

import __main__

from .third_party import decode_json_as_poses, draw_poses, HWC3


class OpenPoseEditor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "openpose_json": ("OPENPOSE_JSON",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "ControlNet Preprocessors"

    def execute(self, openpose_json: str) -> np.ndarray:
        poses, canvas_height, canvas_width = decode_json_as_poses(openpose_json)
        image = draw_poses(poses, canvas_height, canvas_width)
        torch_image = torch.from_numpy(HWC3(image).astype(np.float32) / 255.0)
        return (torch_image.unsqueeze(0),)


NODE_CLASS_MAPPINGS = {"huchenlei.OpenPoseEditor": OpenPoseEditor}

NODE_DISPLAY_NAME_MAPPINGS = {
    "huchenlei.OpenPoseEditor": "sd-webui-openpose-editor",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
