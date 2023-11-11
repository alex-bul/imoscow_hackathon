from pydantic import BaseModel

from typing import List


class ObjectInfo(BaseModel):
    frame_name: str
    frame_url: str
    object_name: str
    count: int
    time: str


class AnalyzeResult(BaseModel):
    filename: str
    objects: List[ObjectInfo]
