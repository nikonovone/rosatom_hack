import io

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

import numpy as np
import torch
from model import load_model, FeatureExtractor
import config as c
from utils import *
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
from torchvision import transforms
import matplotlib.pyplot as plt


SIZE_FRAME = 160
NMS_THRESHOLD = 0.5
device = "cuda"
model_autoenc = load_model("detection_model")


def read_image(uploaded_image):
    image_bytes = uploaded_image.getvalue()
    buf = io.BytesIO(image_bytes)
    image = Image.open(buf)
    image = np.array(image)
    # image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def read_image_autoencoder(uploaded_image):
    image = Image.open(uploaded_image)
    image = image.convert("RGB")
    return image


def preprocess(inputs):
    """move data to device and reshape image"""
    tfs = [
        transforms.Resize(c.img_size),
        transforms.ToTensor(),
        transforms.Normalize(c.norm_mean, c.norm_std),
    ]
    transform = transforms.Compose(tfs)
    inputs = transform(inputs)
    inputs = inputs.view(-1, *inputs.shape[-3:])
    inputs = inputs.to(c.device)
    return inputs


def viz_maps(maps, image):
    image = np.array(image)

    map_to_viz = t2np(
        F.interpolate(
            maps[0][None, None],
            size=image.shape[:2],
            mode="bilinear",
            align_corners=False,
        )
    )[0, 0]
    print(map_to_viz.shape)

    cm = plt.get_cmap("gist_rainbow")
    map_to_viz = map_to_viz / map_to_viz.max()
    colored_image = cm(map_to_viz)
    map_to_viz = 0.7 * image/255 + 0.3 * colored_image[:, :, :3]
    return map_to_viz


def predict_autoencoder(image):
    image = read_image_autoencoder(image)
    device = "cuda"

    model_autoenc.to(device)
    model_autoenc.eval()
    fe = FeatureExtractor()
    fe.eval()
    fe.to(c.device)
    for param in fe.parameters():
        param.requires_grad = False

    print("\nCompute maps, loss and scores on test set:")
    anomaly_score = list()
    c.viz_sample_count = 0
    all_maps = list()
    with torch.no_grad():
        inputs_preproc = preprocess(image)
        inputs = fe(inputs_preproc)
        z = model_autoenc(inputs)

        z_concat = t2np(concat_maps(z))
        nll_score = np.mean(z_concat**2 / 2, axis=(1, 2))
        anomaly_score.append(nll_score)

        z_grouped = list()
        likelihood_grouped = list()
        for i in range(len(z)):
            z_grouped.append(z[i].view(-1, *z[i].shape[1:]))
            likelihood_grouped.append(torch.mean(z_grouped[-1] ** 2, dim=(1,)))
        all_maps.extend(likelihood_grouped[0])
        img = viz_maps([lg[0] for lg in likelihood_grouped], image)
    return img


def autoAdjustments(img):
    # create new image with the same size and type as the original image
    new_img = np.zeros(img.shape, img.dtype)

    # calculate stats
    alow = img.min()
    ahigh = img.max()
    amax = 255
    amin = 0

    # auto adjust each pixel using numpy operations
    new_img = amin + (img - alow) * ((amax - amin) / (ahigh - alow))

    new_img = ((new_img / new_img.max()) * 255).astype(np.uint8)
    return new_img


def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1, y1 = pt1
    x2, y2 = pt2
    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)
    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    return img


def draw_text(
    image,
    text,
    x,
    y,
    color=(0, 255, 0),
    scale=1,
    thikness=1,
):
    image = cv2.putText(
        image.copy(),
        text,
        (x, y),
        cv2.FONT_HERSHEY_COMPLEX,
        scale,
        color,
        thikness,
    )
    return image


def draw_annotations(image, boxes, labels, size_frame, color=(0, 255, 0)):
    _image = image.copy()

    for box, label in zip(boxes, labels):
        p1 = box[:2]
        p2 = box[2:]
        _image = draw_border(_image, p1, p2, color, 4, 5, 10)

        _image = draw_text(
            _image,
            label,
            int(p1[0]) + size_frame,
            int(p1[1]),
            color=color,
            thikness=2,
        )
    return _image


def get_sobel(image, ksize=27):
    # img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(image, (3, 3), 0)

    sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=ksize)
    sobelxy = sobelxy / sobelxy.max()
    # sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=ksize)

    return sobelxy


def predict(image, model):
    # Run batched inference on a list of images
    results = model(image)  # return a list of Results objects

    boxes = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    labels = [results[0].names[x] for x in results[0].boxes.cls.cpu().numpy()]

    return boxes, labels, confs


def nms(bounding_boxes, confidence_score, cls, threshold):
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [], []

    # Bounding boxes
    boxes = np.array(bounding_boxes)

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Confidence scores of bounding boxes
    score = np.array(confidence_score)

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []
    picked_cls = []
    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])
        picked_cls.append(cls[index])
        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]

    return picked_boxes, picked_cls


def main():
    st.set_page_config(layout="wide")
    st.title("Детекция дефектов")

    header = st.container()
    body = st.container()
    with header:
        left_h, right_h = st.columns([1, 1])
        with left_h:
            uploaded_image = st.file_uploader(
                "Выберите изображение",
                type=["jpg", "png", "bmp"],
            )
    with body:
        left, right = st.columns([3, 1])
        with left:
            col_orig, col_sobel = st.columns(2)
            col_autoenoder, col_yolo = st.columns(2)

    model = YOLO(os.path.abspath("./weights/model.pt"))

    if uploaded_image is not None:
        image_orig = read_image(uploaded_image)
        autoencoder_image = predict_autoencoder(uploaded_image)
        sobel_image = get_sobel(image_orig)
        sobel_image = autoAdjustments(sobel_image)
        image = np.stack([image_orig, sobel_image, np.zeros_like(image_orig)], axis=2)

        boxes, labels, confs = predict(image, model)
        boxes, labels = nms(boxes, confs, labels, threshold=NMS_THRESHOLD)

        yolo_image = draw_annotations(
            image_orig,
            boxes,
            labels,
            SIZE_FRAME,
        )
        with left:
            # Display the original image on the left
            col_orig.image(
                image_orig,
                caption="Оригинал",
                use_column_width=True,
                clamp=True,
            )
            col_sobel.image(
                sobel_image,
                caption="Фильтр Собеля",
                use_column_width=True,
                clamp=True,
            )
            col_autoenoder.image(
                autoencoder_image,
                caption="Автоэнкодер",
                use_column_width=True,
                clamp=True,
            )
            col_yolo.image(
                yolo_image,
                caption="Предсказанные дефект",
                use_column_width=True,
                clamp=True,
            )

        x = [int(b[0]) + SIZE_FRAME for b in boxes]
        y = [int(b[1]) + SIZE_FRAME for b in boxes]
        df = pd.DataFrame({"Класс": labels, "x": x, "y": y})
        with body:
            with right:
                st.table(data=df)


if __name__ == "__main__":
    main()
