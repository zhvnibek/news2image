from typing_extensions import TypedDict
from typing import Optional, List


class Params:
    n_images: int
    show_text_summary: bool
    show_text_keywords: bool
    show_image_captions: bool
    show_image_keywords: bool
    # database_size


class Request:
    text: str
    params: Optional[Params]


class Response:
    collage_loc: str
    text_summary: Optional[str]
    text_keywords: Optional[List[str]]
    image_captions: Optional[List[str]]
    image_keywords: Optional[List[str]]
