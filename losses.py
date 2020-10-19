def loss(pred,target):
    pred_xywh=pred[:,:,:,:,0:4]
    pred_conf=pred[:,:,:,:,4:5]
    pred_prob=pred[:,:,:,:,5:]
    label_xywh=pred[:,:,:,:,0:4]
    label_conf=pred[:,:,:,:,4:5]
    label_prob=pred[:,:,:,:,5:]
    ciou = tf.expand_dims(bbox_ciou(pred_xywh, label_xywh), axis=-1)
    input_size = tf.cast(input_size, tf.float32)
    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
    ciou_loss = respond_bbox * bbox_loss_scale * (1 - ciou)
    iou = bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
    max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)
    respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < IOU_LOSS_THRESH, tf.float32)
    conf_focal = tf.pow(respond_bbox - pred_conf, 2)
    conf_loss = conf_focal * (respond_bbox * -(respond_bbox * tf.math.log(tf.clip_by_value(pred_conf, eps, 1.0)))+respond_bgd * -(respond_bgd * tf.math.log(tf.clip_by_value((1- pred_conf), eps, 1.0))))
    prob_loss = respond_bbox * -(label_prob * tf.math.log(tf.clip_by_value(pred_prob, eps, 1.0))+(1 - label_prob) * tf.math.log(tf.clip_by_value((1 - pred_prob), eps, 1.0)))
    ciou_loss = tf.reduce_mean(tf.reduce_sum(ciou_loss, axis=[1, 2, 3, 4]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))
    return ciou_loss, conf_loss, prob_loss
def bbox_ciou(boxes1, boxes2):
    boxes1_coor = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                             boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2_coor = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                             boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]
    enclose_left_up    = tf.maximum(boxes1_coor[..., :2], boxes2_coor[..., :2])
    enclose_right_down = tf.minimum(boxes1_coor[..., 2:], boxes2_coor[..., 2:])
    inter_section = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]#intersection
    union_area = boxes1_area + boxes2_area - inter_area#union
    iou = tf.math.divide_no_nan(inter_area, union_area)#intersection/union
    left_up     = tf.minimum(boxes1_coor[..., :2], boxes2_coor[..., :2])
    right_down  = tf.maximum(boxes1_coor[..., 2:], boxes2_coor[..., 2:])
    c = tf.maximum(right_down - left_up, 0.0)
    c = tf.pow(c[..., 0], 2) + tf.pow(c[..., 1], 2)
    u = (boxes1[..., 0] - boxes2[..., 0]) * (boxes1[..., 0] - boxes2[..., 0]) + \
        (boxes1[..., 1] - boxes2[..., 1]) * (boxes1[..., 1] - boxes2[..., 1])
    d = tf.math.divide_no_nan(u, c)
    ar_gt = tf.math.divide_no_nan(boxes2[..., 2] , boxes2[..., 3])
    ar_pred = tf.math.divide_no_nan(boxes1[..., 2], boxes1[..., 3])
    pi = tf.convert_to_tensor(np.pi)
    ar_loss = tf.math.divide_no_nan(4.0, pi * pi ) * tf.pow((tf.atan(ar_gt) - tf.atan(ar_pred)), 2)
    alpha = tf.math.divide_no_nan(ar_loss ,(1 - iou + ar_loss))
    ciou_term = d + alpha * ar_loss
    ciou = iou - ciou_term
    ciou = tf.clip_by_value(ciou, clip_value_min=-1.0, clip_value_max=0.99)
    return ciou
