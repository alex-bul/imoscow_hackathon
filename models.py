from pydantic import BaseModel

from typing import List


class ObjectInfo(BaseModel):
    object_name: str
    count: int
    time: int


class AnalyzeResult(BaseModel):
    filename: str
    objects: List[ObjectInfo]
