import io
from aiohttp import web

from .third_party import decode_json_as_poses, draw_poses
from server import PromptServer


@PromptServer.instance.routes.post("/openpose/render")
async def render_pose_json(request):
    openpose_json = await request.json()
    poses, canvas_height, canvas_width = decode_json_as_poses(openpose_json)
    image = draw_poses(poses, canvas_height, canvas_width)
    img_buffer = io.BytesIO()
    image.save(img_buffer, format="png")
    headers = {"Content-Type": "image/png"}
    return web.Response(body=img_buffer.getvalue(), headers=headers)
