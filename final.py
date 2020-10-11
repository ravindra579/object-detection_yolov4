from keras.models import Model
from yolov4_model import yolov4_model
model=yolov4_model()
from yolov4_ import load_image_pixels,bounding_boxes
file="test1.jpg"
image,imag_h,image_w=load_image_pixels(file,(416,416))
model.load_weights("check.h5")
yhat=model.predict(image)
bounding_boxes(yhat,file)
