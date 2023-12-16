from ultralytics import YOLO

model = YOLO("yolov8n.pt")

augs = {
    #'hsv_h': 0.015  , #image HSV-Hue augmentation (fraction)
    #'hsv_s': 0.7      , #image HSV-Saturation augmentation (fraction)
    #'hsv_v': 0.4      , #image HSV-Value augmentation (fraction)
    #'degrees': 30.0    , #image rotation (+/- deg)
    "translate": 0.1,  # image translation (+/- fraction)
    "scale": 0.5,  # image scale (+/- gain)
    "shear": 0.0,  # image shear (+/- deg)
    #'perspective': 0.0001, #image perspective (+/- fraction), range 0-0.001
    "flipud": 0.5,  # image flip up-down (probability)
    "fliplr": 0.5,  # image flip left-right (probability)
    "mosaic": 0.3,  # image mosaic (probability)
    "mixup": 0.3,  # image mixup (probability)
    "copy_paste": 0.1,  # segment copy-paste (probability)
}

model.train(
    data="config.yaml", epochs=400, save_period=10, batch=32, **augs
)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
