
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pose_engine import PoseEngine
from PIL import Image
from PIL import ImageDraw
import argparse
import cv2
import os

from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference
import numpy as np
import os

#os.system('wget https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/'
#          'Hindu_marriage_ceremony_offering.jpg/'
#          '640px-Hindu_marriage_ceremony_offering.jpg -O /tmp/couple.jpg')
#pil_image = Image.open('/tmp/couple.jpg').convert('RGB')
#engine = PoseEngine(
 #   'models/mobilenet/posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite')

#print('Inference time: %.f ms' % (inference_time * 1000))

#for pose in poses:
#    if pose.score < 0.4: continue
#    print('\nPose Score: ', pose.score)
#    for label, keypoint in pose.keypoints.items():
#        print('  %-20s x=%-4d y=%-4d score=%.1f' %
#              (label.name, keypoint.point[0], keypoint.point[1], keypoint.score))







def main():
    default_model_dir = '../all_models'
    default_model = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
    default_labels = 'coco_labels.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir,default_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--top_k', type=int, default=3,
                        help='number of categories with highest score to display')
    parser.add_argument('--camera_idx', type=int, help='Index of which video source to use. ', default = 0)
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='classifier score threshold')
    parser.add_argument('--stations', type=float, default=2,
                        help='number of stations that need boxes')
    args = parser.parse_args()

    print('Loading {} with {} labels.'.format(args.model, args.labels))
    engine = PoseEngine(
    'models/mobilenet/posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite')
    inference_size = (640, 480)
#    interpreter = make_interpreter(args.model)
#    interpreter.allocate_tensors()
#    labels = read_label_file(args.labels)
#    inference_size = input_size(interpreter)

    cap = cv2.VideoCapture(args.camera_idx)
    coordinates = imageinteract(cap, args.stations)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2_im = frame
        statement = []
#        print(frame)
#        poses, inference_time = engine.DetectPosesInImage(pil_image)
        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)
        pil_image = Image.fromarray(cv2_im_rgb)
        poses, inference_time = engine.DetectPosesInImage(pil_image)
 #       run_inference(interpreter, cv2_im_rgb.tobytes())
  #      objs = get_objects(interpreter, args.threshold)[:args.top_k]
        for stat in range(args.stations):
            cv2.polylines(cv2_im, [coordinates[stat]],True,(0,0,255),2)
        cv2_im = append_objs_to_img(cv2_im, inference_size, poses, inference_time)

        cv2.imshow('frame', cv2_im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def append_objs_to_img(cv2_im, inference_size, poses, labels):
    height, width, channels = cv2_im.shape
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]
    statement = []

    for pose in poses:
        if pose.score < 0.1: continue
#        print('\nPose Score: ', pose.score)
        for label, keypoint in pose.keypoints.items():
            print(str(int(keypoint.point[0]))+str(int(keypoint.point[1])))
            if label.name == 'RIGHT_ANKLE' or label.name == 'LEFT_ANKLE':
                cv2_im = cv2.circle(cv2_im, (int(keypoint.point[0]),int(keypoint.point[1])), radius=8, color=(0, 255, 0), thickness=-1)
                
                statement.append(inRestrictedSection(cv2_im, inference_size, track, args.stations,restricted_region=coordinates))
            else:
                cv2_im = cv2.circle(cv2_im, (int(keypoint.point[0]),int(keypoint.point[1])), radius=8, color=(0, 0, 255), thickness=-1)
            
#            print('  %-20s x=%-4d y=%-4d score=%.1f' %
#                (label.name, keypoint.point[0], keypoint.point[1], keypoint.score))
        if len(statement)>1:
            result = 'person fully within machine'
        elif result not == 'person fully within machine' or len(statement)<2:
            result = 'person not in machine'

    cv2_im = cv2.putText(cv2_im, result, (20, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

#    for pose in poses:
#        bbox = obj.bbox.scale(scale_x, scale_y)
#        x0, y0 = int(bbox.xmin), int(bbox.ymin)
#        x1, y1 = int(bbox.xmax), int(bbox.ymax\0


#        percent = int(100 * obj.score)
#        label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))##

#        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
#        cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),
#                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    return cv2_im, result

def inRestrictedSection(cv2_im, inference_size, track, stations, restricted_region):  
    R1 = None
    statement = []
    height, width, channels = cv2_im.shape
#    scale_x, scale_y = width / inference_size[0], height / inference_size[1]
    statement = []
#    print(str(scale_x)+str(scale_y))
    x1=int(0 if track.box[0]<0 else track.box[0])
    y1=int(0 if track.box[1]<0 else track.box[1])
    x2=int(track.box[2])
    y2=int(track.box[3])
#    print('x1= '+ str(x1)+'   x2= '+str(x2)+'   y1= '+str(y1)+'   y2= '+str(y2))
#    print(str(restricted_region[0]))
#    print(str(restricted_region[1]))
#    R1=np.array([[track.box[0].astype(int)],[track.box[0].astype(int)],[track.box[0].astype(int)],[track.box[0].astype(int)]],np.int32)
#    R1 = np.array([R1],np.int32)
    R1 = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], np.int32)
#    Im1 = np.zeros((inference_size[0],inference_size[1],1), np.int32)
    Im1 = np.zeros((width,height,1), np.int32)



 #   print('R1 type' + str(type(R1)))
#    print('im type' + str(type(Im1)))
    cv2.fillPoly(Im1, [R1], 255)
#    cv2.imshow('restrictedrs', Im1)
    for stat in range(stations):
        Im2 = np.zeros((width,height,1), np.int32)
#        print(str(Im2.cols)+'  ' + str(Im2.rows))
#        if restricted_region is None:  
#            restricted_region = np.array([[0,ImShape[0]],[ImShape[1],ImShape[0]],[ImShape[1],0], [0,0]], np.int32)  
        cv2.fillPoly(Im2, [restricted_region[stat]], 255)
#        print(str(Im2.shape))
#        cv2.imshow('restrictedrs',Im2 )
 #       print('searching station'+str(stat))
        Im = Im1 * Im2  
        if np.sum(np.greater(Im, 0))>0:
#            print('in a station' + str(stat))
            state = str(track.id) + ' in station ' + str(stat)
            statement.append(state)
        else:
            pass

#    statement.append(state)  
    return statement


def mousepoints(event,x,y,flags,params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,y)
        newpoint = [x,y]
        global points
        points = np.vstack([points, newpoint])

def imageinteract(cap, stations):
#    image = cv2.imread(args["image"])
    coordinates = []

    for i in range(stations):
        global points
        points = np.empty((1,2),dtype=int)
        ret, image = cap.read()
        clone = image
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", mousepoints)
        # keep looping until the 'q' key is pressed
        while True:
    #        ret, image = cap.read()
    #        if not ret:
    #            break
            # display the image and wait for a keypress
            image = clone
            cv2.polylines(image,[points[1:,:]],True, (255,0,0), thickness = 1)  
            cv2.imshow("image", image)
            key = cv2.waitKey(1) & 0xFF
    #        key = cv2.waitKey(33)
            # if the 'r' key is pressed, reset the cropping region
            if key == ord("r"):
                image = clone
            # if thestat 'c' key is pressed, break from the loop
            elif key == ord("c"):
                break
            elif key == ord("n"):
                ret, image = cap.read()
        # if there are two reference points, then crop the region of interest
        # from teh image and display it
    #    if len(refPt) == 2:
    #       roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
    #       cv2.imshow("ROI", roi)
        cv2.waitKey(0)    # close all open windows
        cv2.destroyAllWindows()
        print(points)
        points = np.delete(points, (0), axis=0)
        
        coordinates.append(points)
        
        
        
    return coordinates

if __name__ == '__main__':
    main()
