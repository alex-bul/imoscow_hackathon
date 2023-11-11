import os.path

import uvicorn
import aiofiles

from fastapi import FastAPI, Request, UploadFile, File

from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from models import AnalyzeResult
import cv2
import time

from uuid import uuid4
from yolo_detector import YoloDetector

app = FastAPI()
yolo_detect = YoloDetector('./best.pt')

TMP_UPLOADS_DIRECTORY = 'uploads'
TMP_DETECTED_FRAMES = 'frames'
TMP_DETECTED_VIDEOS = 'detected_videos'

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "id": id})


@app.get("/download")
async def read_item(filename: str) -> FileResponse:
    # Путь к файлу, который нужно отправить для скачивания
    file_path = os.path.join(TMP_DETECTED_VIDEOS, filename)
    # Отправить файл для скачивания
    return FileResponse(file_path, filename=filename)


@app.post("/upload")
async def upload_video(video: UploadFile = File(...)) -> AnalyzeResult:
    filename = str(uuid4()) + '.mp4'
    if TMP_UPLOADS_DIRECTORY not in os.listdir():
        os.mkdir(TMP_UPLOADS_DIRECTORY)

    origin_video_path = os.path.join(TMP_UPLOADS_DIRECTORY, filename)
    async with aiofiles.open(origin_video_path, 'wb') as out_file:
        content = await video.read()  # async read
        await out_file.write(content)  # async write

    cap = cv2.VideoCapture(origin_video_path)
    if cap.isOpened():
        v_width = int(cap.get(3))
        v_height = int(cap.get(4))
        v_fps = int(cap.get(5))

    detected_video_name = str(uuid4()) + '.mp4'
    detected_video_path = os.path.join(TMP_DETECTED_VIDEOS, detected_video_name)
    counter = 0
    vid_writer = cv2.VideoWriter(detected_video_path,
                                 cv2.VideoWriter_fourcc(*'mp4v'),
                                 v_fps,
                                 (v_width, v_height)
                                 )

    while cap.isOpened():
        success, img = cap.read()
        start = time.perf_counter()
        if not success:
            break

        if counter % 3 == 0:
            result, labels, cords, confs = yolo_detect.score_frame(img)
            vid_writer.write(result.plot()[:, :, :])
        else:
            vid_writer.write(img)

        counter += 1
    vid_writer.release()
    cap.release()



    # Возвращаем результаты проверки в формате JSON
    results = {
        'filename': detected_video_name,
        'objects': [{
            'object_name': 'example_object',
            'count': 5,
            'time': 10
        }]}
    return AnalyzeResult.model_validate(results)


if __name__ == '__main__':
    uvicorn.run(app, port=3000)
