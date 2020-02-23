import os
import cv2
import math
from typing import Optional
from src.config import ImageConfig


# Code from https://note.nkmk.me/en/python-opencv-hconcat-vconcat-np-tile/
def stack_horizontal(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)


def stack_vertical(im_list, interpolation=cv2.INTER_CUBIC):
    w_min = min(im.shape[1] for im in im_list)
    im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                      for im in im_list]
    return cv2.vconcat(im_list_resize)


def stack_grid(image_ids: list) -> Optional[str]:

    def _get_dims(count: int) -> tuple:
        _cols = math.ceil(math.sqrt(count))
        _rows = math.ceil(count / _cols)
        return _cols, _rows

    if not image_ids:
        return None

    image_full_filenames = [os.path.join(ImageConfig.get_images_folder(), f'{_id}.jpg') for _id in image_ids]
    cols, rows = _get_dims(len(image_full_filenames))
    v_list = []
    for row in range(rows):
        row_im_list = [cv2.imread(im) for im in image_full_filenames[row*cols:(row+1)*cols]]
        im_h_resize = stack_horizontal(im_list=row_im_list)
        v_list.append(im_h_resize)
    im_v_resize = stack_vertical(im_list=v_list)
    img_loc = f'temp/collage/{"&".join(image_ids)}.jpg'
    cv2.imwrite(img_loc, im_v_resize)
    return img_loc


def make_collage(image_ids: list) -> str:
    default = 'temp/collage/default.png'
    collage = stack_grid(image_ids=image_ids)
    return collage if collage is not None else default


if __name__ == '__main__':
    img_ids = [img.split('.')[0] for img in os.listdir(ImageConfig.get_images_folder())]
    test_img_ids = img_ids[11:15]
    collage_loc: str = make_collage(image_ids=test_img_ids)
