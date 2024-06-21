# ComfyUI-openpose-editor
[sd-webui-openpose-editor](https://github.com/huchenlei/sd-webui-openpose-editor) in ComfyUI

## Requirement
- https://github.com/Fannovel16/comfyui_controlnet_aux

## How to use
### Step 1: Use pose estimator node to obtain `POSE_KEYPOINT` JSON
![image](https://github.com/huchenlei/ComfyUI-openpose-editor/assets/20929282/5ed82dc9-4804-4263-9f05-e58a3202bf17)

### Step 2: Use `Load Openpose JSON` node to load JSON
![image](https://github.com/huchenlei/ComfyUI-openpose-editor/assets/20929282/2ca2021f-c618-48d6-bdfb-e132ffc95167)

### Step 3: Perform necessary edits
![image](https://github.com/huchenlei/ComfyUI-openpose-editor/assets/20929282/c37af495-6b3e-4e11-b801-ae342f97760a)

Click `Send pose to ControlNet` will send the pose back to ComfyUI and close the modal.
![image](https://github.com/huchenlei/ComfyUI-openpose-editor/assets/20929282/5178044f-1577-4174-97ca-63b7404b317c)
![image](https://github.com/huchenlei/ComfyUI-openpose-editor/assets/20929282/c21bc5d0-4488-4b6b-8857-398cfbeebd53)
