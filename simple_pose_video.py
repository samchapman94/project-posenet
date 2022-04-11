
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
engine = PoseEngine(
    'models/mobilenet/posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite')

print('Inference time: %.f ms' % (inference_time * 1000))

for pose in poses:
    if pose.score < 0.4: continue
    print('\nPose Score: ', pose.score)
    for label, keypoint in pose.keypoints.items():
        print('  %-20s x=%-4d y=%-4d score=%.1f' %
              (label.name, keypoint.point[0], keypoint.point[1], keypoint.score))







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
    args = parser.parse_args()

    print('Loading {} with {} labels.'.format(args.model, args.labels))
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = read_label_file(args.labels)
    inference_size = input_size(interpreter)

    cap = cv2.VideoCapture(args.camera_idx)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2_im = frame
#        poses, inference_time = engine.DetectPosesInImage(pil_image)
        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)
        pil_image = Image.fromarray(cv2_im_rgb)
        poses, inference_time = engine.DetectPosesInImage(pil_image)
 #       run_inference(interpreter, cv2_im_rgb.tobytes())
  #      objs = get_objects(interpreter, args.threshold)[:args.top_k]
        cv2_im = append_objs_to_img(cv2_im, inference_size, poses, inference_time)

        cv2.imshow('frame', cv2_im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def append_objs_to_img(cv2_im, inference_size, poses, labels):
    height, width, channels = cv2_im.shape
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]


    for pose in poses:
        if pose.score < 0.4: continue
#        print('\nPose Score: ', pose.score)
        for label, keypoint in pose.keypoints.items():
            if label.name == RIGHT_ANKLE or label.name == LEFT_ANKLE:
                image = cv2.circle(image, (keypoint.point[0]*scale_x,keypoint.point[1]*scale_y), radius=3, color=(0, 255, 0), thickness=-1)
            else:
                image = cv2.circle(image, (keypoint.point[0]*scale_x,keypoint.point[1]*scale_y), radius=3, color=(0, 0, 255), thickness=-1)
            
#            print('  %-20s x=%-4d y=%-4d score=%.1f' %
#                (label.name, keypoint.point[0], keypoint.point[1], keypoint.score))



#    for pose in poses:
#        bbox = obj.bbox.scale(scale_x, scale_y)
#        x0, y0 = int(bbox.xmin), int(bbox.ymin)
#        x1, y1 = int(bbox.xmax), int(bbox.ymax\0


#        percent = int(100 * obj.score)
#        label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))##

#        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
#        cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),
#                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
#    return cv2_im

if __name__ == '__main__':
    main()
