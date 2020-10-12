import numpy as np 
from matplotlib import pyplot
import tensorflow as tf
from PIL import Image
from keras.preprocessing.image import load_img,img_to_array
threshold=0.2
nms_threshold=0.7
input_h,input_w=416,416
anchors = [ [12, 16, 19, 36, 40, 28],[36, 75, 76, 55, 72, 146],[142, 110, 192, 243, 459, 401]]
input_h,input_w=416,416
#-----to elminate grid sensitivity scale up -----------
scales_x_y = [1.2, 1.1, 1.05]
boxes = list()
#-----intersection over union-------------
def iou(box1,box2):
    x11,x21=box1.xmin,box1.xmax
    x31,x41=box1.ymin,box1.ymax
    x12,x22=box2.xmin,box2.xmax
    x32,x42=box2.ymin,box2.ymax
    b1_l=x21-x11
    b1_w=x41-x31
    b2_l=x22-x12
    b2_w=x42-x32
    #intersection
    inter=(min(x21,x22)-max(x11,x12))*(min(x41,x42)-max(x32,x31))
    #union
    union=b1_l*b1_w+b2_l*b2_w-inter
    return float(inter)/union #float helps to convert division part to float
#---------non max supression---------------
def nms(a,nms_tresh=0.5):
        if(len(a)>0):
            num_classes=len(a[0].classes)
        else:
            return
        for i in range(num_classes):
            sort_indexes=np.argsort([-box.classes[i] for box in a])# sort in descending order such that box with max prob is selected
            for j in range(len(sort_indexes)):
                index=sort_indexes[j]
                if a[index].classes[i]==0:
                    continue
                for k in range(j+1,len(sort_indexes)):
                    index_=sort_indexes[k]
                    if iou(a[index],a[index_])>=nms_tresh:
                        a[index_].classes[i]=0#supress the overlapping boxes
def _sigmoid(x):
    return 1./(1.+np.exp(-x))
#-----------to save box parameters in order------------------------
class save_box_param:
    def __init__(self,xmin,ymin,xmax,ymax,prob=None,classes=None):
        self.xmin=xmin
        self.xmax=xmax
        self.ymin=ymin
        self.ymax=ymax
        self.prob=prob
        self.classes=classes
        self.label=-1
        self.score=-1
    def get_label(self):
        if self.label==-1:
            self.label=np.argmax(self.classes)#class num which has high probability
        return self.label
    def get_score(self):
        if self.score==-1:
            self.score=self.classes[self.get_label()]#high probabilty of class
        return self.score
#----------------grid sensitivity-------------------------
def decode(y,anchor,tresh,x_1,x_2,n,scale):
    grid_h,grid_w=y.shape[:2]# 1-52 x 52 ,2- 26 x 26 ,3- 13 x 13
    num_boxes=n#3
    y=y.reshape((grid_h,grid_w,num_boxes,-1))# grid_h,grid_w,3,85
    boxes=[]
    nb_class = y.shape[-1] - 5 #no. of classes 80
    #-------- in order to eliminate grid sensitivity bx=sigmoid(tx)+cx
    y[...,:2]=_sigmoid(y[...,:2])# sigmoid of centers
    y[...,:2]=y[...,:2]*scale- 0.5*(scale - 1.0)
    y[..., 4:] = _sigmoid(y[..., 4:]) # probablity of  80 classes
    for i in range(grid_w*grid_h):
        row=i/grid_h#1,2,3
        col=i%grid_w#1,1+no.of rows
        for j in range(num_boxes):
            prob=y[int(row)][int(col)][j][4]
            if(prob>tresh):# elimnate the objects with probablity less than tresh
                x, y, w, h = y[int(row)][int(col)][j][:4]
                x = (col + x) / grid_w # x center
                y = (row + y) / grid_h # y center 
                w = anchor[2 * j + 0] * np.exp(w) / x_2
                h = anchor[2 * j + 1] * np.exp(h) / x_1
                clas = prob*y[int(row)][col][j][5:]
                clas *= clas > tresh
                box=save_box_param(x-w/2,y-h/2,x+w/2,y+h/2,prob,clas)
                boxes.append(box)
    return boxes
def get_boudings(boxes,labels):
    out_boxes,out_labels,out_scores=list(),list(),list()
    for box in boxes:
        for i in range(len(labels)):
             if box.classes[i] > threshold:
                out_boxes.append(box)
                out_labels.append(labels[i])
                out_scores.append(box.classes[i]*100)
    return out_boxes,out_labels,out_scores
def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
 new_w, new_h = net_w, net_h
 for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
        y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h
        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)
def draw_boxes(filename, out_boxes,out_labels, out_scores):
    #data = pyplot.imread(filename) #load the image
    data=filename
    pyplot.imshow(data)# plot the image
    ax = pyplot.gca()# get the context for drawing boxes
    # plot each box
    for i in range(len(out_boxes)):
        box = out_boxes[i]
        y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax# get coordinates
        width, height = x2 - x1, y2 - y1# calculate width and height of the box
        rect = pyplot.Rectangle((x1, y1), width, height, fill=False, color="red")# create the shape
        ax.add_patch(rect)# draw the box
        label = "%s (%.3f)" % (out_labels[i], out_scores[i])# draw text and score in top left corner
        pyplot.text(x1, y1, label, color='white')
    pyplot.show()# show the plot
def load_image_pixels(filename, shape):
    #image = load_img(filename)# load the image to get its shape
    image=filename
    width, height = image.size
    #image = load_img(filename, interpolation = 'bilinear', target_size=shape)# load the image with the required size
    image = img_to_array(image)# convert to numpy array
    image = image.astype('float32')# scale pixel values to [0, 1]
    image /= 255.0
    image = tf.expand_dims(image, 0)# add a dimension so that we have one sample
    return image, width, height
def decode_netout(netout, anchors, obj_thresh, net_h, net_w, anchors_nb, scales_x_y):
    grid_h, grid_w = netout.shape[:2]  
    nb_box = anchors_nb
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))
    nb_class = netout.shape[-1] - 5 # 5 = bx,by,bh,bw,pc
    boxes = []
    netout[..., :2] = _sigmoid(netout[..., :2]) # x, y
    netout[..., :2] = netout[..., :2]*scales_x_y - 0.5*(scales_x_y - 1.0) # scale x, y
    netout[..., 4:] = _sigmoid(netout[..., 4:]) # objectness + classes probabilities
    for i in range(grid_h*grid_w):
        row = i / grid_w
        col = i % grid_w
        for b in range(nb_box):
            # 4th element is objectness
            objectness = netout[int(row)][int(col)][b][4]
            if(objectness > obj_thresh):
                # first 4 elements are x, y, w, and h
                x, y, w, h = netout[int(row)][int(col)][b][:4]
                x = (col + x) / grid_w # center position, unit: image width
                y = (row + y) / grid_h # center position, unit: image height
                w = anchors[2 * b + 0] * np.exp(w) / net_w # unit: image width
                h = anchors[2 * b + 1] * np.exp(h) / net_h # unit: image height            
                # last elements are class probabilities
                classes = objectness*netout[int(row)][col][b][5:]
                classes *= classes > obj_thresh
                box = save_box_param(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)           
                boxes.append(box)
    return boxes
def read_labels(labels_path):
    with open(labels_path) as f:
        labels = f.readlines()
    labels = [c.strip() for c in labels]
    return labels
def bounding_boxes(yhat,file,labels):
    #image, image_w, image_h = load_image_pixels(file, (input_w, input_h))
    boxes=list()
    image_h,image_w=416,416
    for i in range(len(anchors)):
        boxes+=decode_netout(yhat[i][0], anchors[i],threshold, input_h, input_w, len(anchors), scales_x_y[i])
    correct_yolo_boxes(boxes, image_h, image_w, 416,416)
    nms(boxes,nms_threshold)
    out_boxes,out_labels,out_scores=get_boudings(boxes,labels)
    draw_boxes(file, out_boxes,out_labels, out_scores)
