import os
import cv2
import numpy as np
import tensorflow as tf
from utils import label_map_util
from tracker import *

NUMBER_OF_LEFT_MOTORS = 0
NUMBER_OF_LEFT_CARS = 0
NUMBER_OF_LEFT_BUSES = 0
NUMBER_OF_RIGHT_MOTORS = 0
NUMBER_OF_RIGHT_CARS = 0
NUMBER_OF_RIGHT_BUSES = 0
LIST_COUNTED = []
NUM_CLASSES = 3

labels_path = 'training/labelmap.pbtxt'
ckpt_path = 'inference_graph/frozen_inference_graph.pb'
video_path = 'test2.mp4'
# Load map & categories
label_map = label_map_util.load_labelmap(labels_path)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(ckpt_path, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier
# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')
video = cv2.VideoCapture(video_path)
out = cv2.VideoWriter('output3.avi', -1, 20.0, (960, 540))
# initialize our centroid tracker and frame dimensions
ct = CentroidTracker()
height, width = None, None

while video.isOpened():

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    ret, frame = video.read()
    # Get height & weight of video at the first time
    if height is None or width is None:
        height, width = frame.shape[:2]
        # out = cv2.VideoWriter('output.avi', -1, 20.0, (int(width*0.5), int(height*0.5)))
    frame_expanded = np.expand_dims(frame, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})
    # print(boxes[0][1])
    # Draw the results of the detection (aka 'visulaize the results')
    # vis_util.visualize_boxes_and_labels_on_image_array(
    #     frame,
    #     np.squeeze(boxes),
    #     np.squeeze(classes).astype(np.int32),
    #     np.squeeze(scores),
    #     category_index,
    #     use_normalized_coordinates=True,
    #     line_thickness=8,
    #     min_score_thresh=0.80)

    # print(str(len(np.squeeze(scores))) + ',' + str(len(np.squeeze(scores))))
    rects = []
    track_categories = []
    list_boxes = np.squeeze(boxes)
    list_scores = np.squeeze(scores)
    list_classes = np.squeeze(classes)
    for i in range(0, len(list_boxes)):
        if list_scores[i] > 0.9:
            box = list_scores[i]
            ymin, xmin, ymax, xmax = int(list_boxes[i][0] * height), int(list_boxes[i][1] * width), \
                                     int(list_boxes[i][2] * height), int(list_boxes[i][3] * width)
            # print(ymin, xmin, ymax, xmax)
            rects.append([xmin, ymax, xmax, ymin])
            track_categories.append(list_classes[i])
            cv2.rectangle(frame, (xmin, ymax), (xmax, ymin), (100, 255, 0), 2)
    objects = ct.update(rects, track_categories)
    list_deregister = ct.get_deregister()
    # print(list_deregister)
    for x in list_deregister:
        if x in LIST_COUNTED:
            LIST_COUNTED.remove(x)
    # print('counted: ' + str(LIST_COUNTED))
    for(objectID, centroid) in objects.items():
        # print(objectID)
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = str(ct.categories[objectID]) + "-{}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
        if objectID not in LIST_COUNTED:
            # print('check ' + str(centroid))
            result = check_and_count(centroid, 535, 290, 250)
            if result == 'Left':
                LIST_COUNTED.append(objectID)
                if ct.categories[objectID] == 1:
                    NUMBER_OF_LEFT_MOTORS += 1
                elif ct.categories[objectID] == 2:
                    NUMBER_OF_LEFT_CARS += 1
                elif ct.categories[objectID] == 3:
                    NUMBER_OF_LEFT_BUSES += 1
            elif result == 'Right':
                LIST_COUNTED.append(objectID)
                if ct.categories[objectID] == 1:
                    NUMBER_OF_RIGHT_MOTORS += 1
                elif ct.categories[objectID] == 2:
                    NUMBER_OF_RIGHT_CARS += 1
                elif ct.categories[objectID] == 3:
                    NUMBER_OF_RIGHT_BUSES += 1

    # Show number of vehicles on the left lane
    cv2.putText(frame, 'Motors:' + str(NUMBER_OF_LEFT_MOTORS), (3, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, 'Cars:' + str(NUMBER_OF_LEFT_CARS), (3, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # Show number of vehicles on the right lane
    cv2.putText(frame, 'Motors:' + str(NUMBER_OF_RIGHT_MOTORS), (800, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, 'Cars:' + str(NUMBER_OF_RIGHT_CARS), (800, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # All the results have been drawn on the frame, so it's time to display it.
    cv2.line(frame, (535, 1), (535, height), (255, 0, 0), 2)
    cv2.line(frame, (1, 290), (535, 290), (0, 0, 255), 2)
    cv2.line(frame, (535, 250), (width, 250), (0, 0, 255), 2)
    cv2.imshow('Object detector', frame)
    # small_f = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    out.write(frame)
    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
video.release()
# out.release()
cv2.destroyAllWindows()
