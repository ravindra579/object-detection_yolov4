# object-detection_yolov4
YOLOV4(you only look at once) it is the fourt version 

First we see what's new in yolov4

 YOLOV4's backbone arcitecture can be vgg16,resnet-50,resnet16-101,darknet53.... as said in official paper it is better to use "CSPdarknet53" you can check the architecture the flow chart is shown in "yolov4_model.pdf" and the code is shown in "yolov4_model.py"
 
 "cspdarknet53" is a novel backbone that can enhance the learning capability of cnn 
 
 You can download offical paper from this link https://arxiv.org/abs/2004.10934 and go through the paper to unsterstood better
 
 The neck part of yolov4 will be fpn,spp,panet..... if we use "SPP(spatial pyramid pooling)" it gives more accuracy the spp block is added over "cspdarknet53" to increase the receptive field and seperate out most signitficant features 
 
 YOLOV4 is twice as fast as efficiendet with comparable performance and fps increased by 10% to 12% compared to YOLOV3
 
 Higher input network size (resolution) – for detecting multiple small-sized objects ,More layers – for a higher receptive field to cover the increased size of input network , More parameters – for greater capacity of a model to detect multiple objects of different sizes in a single image
 
 I have used "412 x 412" as input image shape for model
 
 "convert.py" is usedto convert the outputs of official "yolov4.weights" to "yolov4.h5" becuase load_model can recognize ".h5" or ".hdf5" format but the official weights are in ".weights" format if you didn't understood how the "convert.py" works check this blog https://medium.com/@ravindrareddysiddam/how-to-convert-yolov4-from-weights-to-h5-format-b50b244b3298 you will get an idea the blog was written by me it my or may not be a good blog but you will get an idea if you read that whole convert.py is explained in that blog
 
 "model.txt" consist all labels  as it is a cocodataset it contains 0 classes or labels 
 
 If you want to download weights the link in "weights.txt" file you can check that otherwise if you want to train on your own coustom dataset you can use man losses like ciou losses giou loss if you want reference you can check it in "losses.py" file
 
 "yolov4_.py" is just like a dense prediction contains "NMS(non max supression)" ,IOU(intersection over union)" and many functions used for images and for videos we can use "yolov4_video.py" 
 
 Finally "final.py" is used for combining all theses files and to get the output for images whereas for videos we can use "video.ipynb" 
