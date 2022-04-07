import argparse
import glob
import os.path as Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from detection.object_detection import detect_object

parser = argparse.ArgumentParser(description="YOLOv4 COCO detection with PyTorch")
parser.add_argument("--cfg", type=str, default="yolo/yolov4.cfg", help="Config path")
parser.add_argument(
    "--weights", type=str, default="yolo/yolov4.weights", help="Weights path"
)
parser.add_argument("--names", type=str, default="yolo/coco.names", help="Names path")
parser.add_argument(
    "--car_images", type=str, default="data/test", help="Car images path"
)
parser.add_argument(
    "--random_images", type=str, default="data/test2", help="Random images path"
)
args = parser.parse_args()

with open(args.names) as file:
    lines = [line.rstrip() for line in file]
lines.insert(0, "anything")


def main():
    readme_text: str = st.markdown(open("info").read())
    st.sidebar.title("Меню")
    app_mode = st.sidebar.selectbox(
        "Выберите что сделать", ["О программе", "Запуск приложения", "Исходный код"]
    )
    if app_mode == "О программе":
        st.sidebar.success('Чтобы продолжить, выберите "Запуск приложения"')
        st.image("readme.png")
    elif app_mode == "Исходный код":
        readme_text.empty()
        st.title("Код главного файла")
        st.code(open("myselfdrive.py").read())
        st.title("Код файла предсказаний")
        st.code(open("detection/object_detection.py").read())
    elif app_mode == "Запуск приложения":
        readme_text.empty()
        image_pack = st.sidebar.selectbox(
            "Выберите набор картинок", ["Транспорты и пешеходы", "Случайные картинки"]
        )
        if image_pack == "Транспорты и пешеходы":
            DATA_ROOT: str = args.car_images
            run_the_app(DATA_ROOT)
        elif image_pack == "Случайные картинки":
            DATA_ROOT: str = args.random_images
            run_the_app(DATA_ROOT)


def run_the_app(data_root: str):
    DATA_ROOT = data_root
    selected_frame_index, selected_frame = frame_selector_ui(DATA_ROOT)
    orig_img = load_image(selected_frame_index, DATA_ROOT)
    orig_img = cv2.resize(orig_img, dsize=(704, 704), interpolation=cv2.INTER_AREA)
    image = st.image(orig_img)
    image.empty()
    confidence = confidence_value()
    object_selector(orig_img, confidence)


def object_selector(img, confidence: int):
    req_class = st.sidebar.selectbox("Выберите объект", lines)
    image = detect_object(img, req_class, confidence, args)
    return st.image(image)


def frame_selector_ui(data_root: str):
    images: str = glob.glob(Path.join(data_root, "*.jpg"))
    selected_frame_index = st.sidebar.slider("Выберите картинку", 0, 256)
    selected_frame = images[selected_frame_index]
    return selected_frame_index, selected_frame


def confidence_value():
    confidence: float = st.sidebar.slider("Уверенность", 0.0, 1.0, 0.3)
    return confidence


@st.cache(show_spinner=False)
def load_image(img_idx: int, data_root: str):
    images: str = glob.glob(Path.join(data_root, "*.jpg"))
    image = images[img_idx]
    if image is not None:
        try:
            image = Image.open(image)
        except Exception:
            st.error("Ошибка, выбрана не картинка")
        else:
            img_array = np.array(image)
            return img_array
    return image


if __name__ == "__main__":
    main()
