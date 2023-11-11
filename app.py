import os.path

import uvicorn
import aiofiles

from fastapi import FastAPI, Request, UploadFile, File

from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from typing import List, Dict
from models import AnalyzeResult

from uuid import uuid4

app = FastAPI()
TMP_UPLOADS_DIRECTORY = 'uploads'

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "id": id})


@app.get("/download")
async def read_item(filename: str) -> FileResponse:
    # Путь к файлу, который нужно отправить для скачивания
    file_path = os.path.join(TMP_UPLOADS_DIRECTORY, filename)

    # Отправить файл для скачивания
    return FileResponse(file_path, filename=filename)


@app.post("/upload")
async def upload_video(video: UploadFile = File(...)) -> AnalyzeResult:
    # Здесь можно выполнить проверку видео на бекенде
    filename = str(uuid4()) + '.mp4'
    if TMP_UPLOADS_DIRECTORY not in os.listdir():
        os.mkdir(TMP_UPLOADS_DIRECTORY)

    async with aiofiles.open(os.path.join(TMP_UPLOADS_DIRECTORY, filename), 'wb') as out_file:
        content = await video.read()  # async read
        await out_file.write(content)  # async write

    # Возвращаем результаты проверки в формате JSON
    results = {
        'filename': filename,
        'objects': [{
            'object_name': 'example_object',
            'count': 5,
            'time': 10
        }]}
    return AnalyzeResult.model_validate(results)


if __name__ == '__main__':
    uvicorn.run(app, port=3000)
