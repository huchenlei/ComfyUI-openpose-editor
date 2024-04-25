# ComfyUI-openpose-editor
Port of https://github.com/huchenlei/sd-webui-openpose-editor in ComfyUI

WIP

## Architecture
### Front-End node
- Stores the json pose object
- Call `/openpose/render` API endpoint to convert json to image
- Preview the pose image
- Display an edit button to trigger the editor modal
  -  On click: send the json pose to editor
  - On editor close: receive the updated json and update preview
- Input: [Optional] Background image
- Output: Pose Image

### Back-End node
- Output the rendered openpose image

Dev List:
- [ ] Figure out how backend can read change made by front-end on rendered openpose image.
