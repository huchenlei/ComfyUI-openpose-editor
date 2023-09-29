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


import os
import filecmp
import shutil


def setup_js():
    extensions_folder = os.path.join(
        os.path.dirname(os.path.realpath(__main__.__file__)),
        "web",
        "extensions",
        "ComfyUI-openpose-editor",
    )
    javascript_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "js")

    if not os.path.exists(extensions_folder):
        print("Creating frontend extension folder: " + extensions_folder)
        os.mkdir(extensions_folder)

    result = filecmp.dircmp(javascript_folder, extensions_folder)

    if result.left_only or result.diff_files:
        print("Update to javascripts files detected")
        file_list = list(result.left_only)
        file_list.extend(x for x in result.diff_files if x not in file_list)

        for file in file_list:
            print(f"Copying {file} to extensions folder")
            src_file = os.path.join(javascript_folder, file)
            dst_file = os.path.join(extensions_folder, file)
            if os.path.exists(dst_file):
                os.remove(dst_file)
            shutil.copy(src_file, dst_file)


setup_js()
