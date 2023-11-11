from pydantic import BaseModel

from typing import List


class ObjectInfo(BaseModel):
    frame_name: str
    object_name: str
    count: int
    time: int


class AnalyzeResult(BaseModel):
    filename: str
    objects: List[ObjectInfo]
