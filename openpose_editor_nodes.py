import json


OpenposeJSON = dict


class LoadOpenposeJSONNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "json_str": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("POSE_KEYPOINT",)
    FUNCTION = "load_json"
    CATEGORY = "openpose"

    def load_json(self, json_str: str) -> tuple[OpenposeJSON]:
        return (json.loads(json_str),)
