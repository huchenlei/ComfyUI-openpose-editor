from .openpose_editor_nodes import LoadOpenposeJSONNode


WEB_DIRECTORY = "js"

NODE_CLASS_MAPPINGS = {
    "huchenlei.LoadOpenposeJSON": LoadOpenposeJSONNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "huchenlei.LoadOpenposeJSON": "Load Openpose JSON",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
