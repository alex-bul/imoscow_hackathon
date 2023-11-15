import os.path

import datetime
import uvicorn
import aiofiles

from fastapi import FastAPI, Request, UploadFile, File

from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from models import AnalyzeResult
import cv2
import numpy as np

from uuid import uuid4
from yolo_detector import YoloDetector

app = FastAPI()
yolo_detect = YoloDetector('./best.pt')

TMP_UPLOADS_DIRECTORY = 'uploads'
TMP_DETECTED_FRAMES = 'frames'
TMP_DETECTED_VIDEOS = 'detected_videos'

CLASS_NAMING = {
    'balloons': "Воздушные шары",
    'box of products': "Продукты",
    'cotton candy': "Сладкая вата",
    'cotton candy tent': "Оборудование для сладкой ваты",
    'different items': "Продажа вещей и предметов",
    'flowers': "Цветы",
    'kvas bottle': "Квас в бутылях",
    'kvas': "Квас в цистерне",
    'stall': "Продовольственный контейнер",
    'stall umbr': "Торговая палатка/навес",
    'trunk': "Продажа из авто"
}

STATIST_BORDER = {
    0: 1,  # Balloons
    1: 10,  # Box of Products
    2: 5,  # Cotton Candy
    3: 1,  # Cotton Candy Tent
    4: 2,  # Different items
    5: 7,  # Flowers
    6: 1,  # Kvas
    7: 2,  # Kvas bottle
    8: 1,  # Stall
    9: 1,  # Stall Umbr
    10: 1  # Trunk
}

for folder in [TMP_UPLOADS_DIRECTORY, TMP_DETECTED_FRAMES, TMP_DETECTED_VIDEOS]:
    if folder not in os.listdir():
        os.mkdir(folder)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/frames", StaticFiles(directory="frames"), name="frames")

templates = Jinja2Templates(directory="templates")


async def is_point_of_sale(classes):
    for i in range(11):
        if len(classes) > 0 and sum(classes == i) >= STATIST_BORDER[i]:
            return True
    return False


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
async def upload_video(request: Request, video: UploadFile = File(...)) -> AnalyzeResult:
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
    frames_objects_dict = {}
    frame_cut_time = {}
    previous_annot = None
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break

        if counter % 3 == 0:
            result, labels, cords, confs = yolo_detect.score_frame(img)
            if await is_point_of_sale(result.boxes.cls):
                frame_name = str(uuid4()) + '.jpg'
                frame_path = os.path.join(TMP_DETECTED_FRAMES, frame_name)
                frame_cut_time[frame_name] = counter / v_fps

                det_classes = [result.names.get(value, value) for value in result.boxes.cls]
                num_classes = {name: 0 for name in result.names.values()}
                for cl in det_classes:
                    num_classes[result.names[cl.item()]] += 1
                dict_count_obj = {key: value for key, value in num_classes.items() if value > 0}

                if previous_annot != dict_count_obj:
                    cv2.imwrite(frame_path, result.plot()[:, :, :])
                    previous_annot = dict_count_obj
                    frames_objects_dict[frame_name] = dict_count_obj

            vid_writer.write(result.plot()[:, :, :])
        else:
            vid_writer.write(img)

        counter += 1
    vid_writer.release()
    cap.release()

    detected_objects_frame = []
    # Возвращаем результаты проверки в формате JSON
    for key, value in frames_objects_dict.items():
        for key_obj, value_obj in value.items():
            detected_objects_frame.append(
                {'frame_name': key, 'object_name': CLASS_NAMING.get(key_obj.lower(), key_obj), 'count': value_obj,
                 'time': str(datetime.timedelta(seconds=int(frame_cut_time[key])))[-9:],
                 'frame_url': str(request.url_for('frames', path=key))})
    results = {
        'filename': detected_video_name,
        'objects': detected_objects_frame}
    print(results)
    return AnalyzeResult.model_validate(results)


if __name__ == '__main__':
    uvicorn.run(app, port=3000)
