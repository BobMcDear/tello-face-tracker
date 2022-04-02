# Tello-Face-Tracker

## Description
This is an implementation of a face tracker for the Ryze Tello drone. You can find the accompanying blog [here](https://borna-ahz.medium.com/face-tracking-with-the-ryze-tello-part-1-face-detection-75ec97b1d8d2).

## Usage
To run the face tracker, you must first download [the Caffe model](https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel) for face detection and [its associated text description](https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy.prototxt).
Next, ```main.py``` needs to be run, and the paths to the model and architecture description are to be provided. Assuming your device's Wi-Fi is linked to that
of the Tello, this would connect to the drone, take off, and track the main face within the camera's field of view so it is in front of the drone. The drone's video stream is displayed
and identified faces are shown with bounding boxes.

Example:

```
python main.py --model_file res10_300x300_ssd_iter_140000.caffemodel --proto_file deploy.prototxt.txt
```

The implemented modules may also be used out-of-the-box. They include,

* ```face_detection.py```: 
  * ```get_caffe_net```: Reads a Caffe network with OpenCV
    * Args:
      * ```proto_file (str)```: Path to the .prototxt file that contains the model's textual description. Default is  ```deploy.prototxt.txt```
      * ```model_file (str)```: Path to the model's parameters. Default is ```res10_300x300_ssd_iter_140000.caffemodel```
    * Returns (```dnn_Net```): The Caffe network
  * ```get_faces```: Identifies faces in an image
    * Args:
      * ```net (dnn_Net)```: ```dnn_Net``` for face identification (e.g., the one linked above)
      * ```image (array)```: Image, represented as a NumPy array 
    * Returns (```List[Tuple[int, int, int, int, float]]```): List of detected faces, where each element is a Tuple in the form of [top-left X-coordinate, top-left Y-coordinate, bottom-right X-coordinate, bottom-right Y-coordinate, confidence] associated with a face. Only faces with scores of over 0.75 are included.
  * ```get_most_confident_face```: Identifies the face in an image with the highest confidence
    * Args:
      * ```net (dnn_Net)```: ```dnn_Net``` for face identification (e.g., the one linked above)
      * ```image (array)```: Image, represented as a NumPy array 
    * Returns (```Tuple[int, int, int, int]```): Tuple in the form of [top-left X-coordinate, top-left Y-coordinate, bottom-right X-coordinate, bottom-right Y-coordinate] for the face with the highest confidence
* ```face_tracking.py```: 
  * ```Controls```: A control system for tracking objects with the Tello. All methods are ```@staticmethod```.
    * ```get_forward_backward_velocity```: Gets the best forward/backward velocity for tracking an object
      * Args:
        * ```x1 (int)```: X-coordinate of the top-left corner of the object
        * ```y1 (int)```: Y-coordinate of the top-left corner of the object
        * ```x2 (int)```: X-coordinate of the bottom-right corner of the object
        * ```y2 (int)```: Y-coordinate of the bottom-right corner of the object
      * Returns (```int```): Forward/backward velocity for tracking the specified object
    * ```get_up_down_velocity```: Gets the best up/down velocity for tracking an object
      * Args:
        * ```y1 (int)```: Y-coordinate of the top-left corner of the object
        * ```y2 (int)```: Y-coordinate of the bottom-right corner of the object
      * Returns (```int```): Up/down velocity for tracking the specified object
    * ```get_yaw_velocity```: Gets the best yaw velocity for tracking an object
      * Args:
        * ```x1 (int)```: X-coordinate of the top-left corner of the object
        * ```x2 (int)```: X-coordinate of the bottom-right corner of the object
      * Returns (```int```): Yaw velocity for tracking the specified object
    * ```get_rc_controls```: Gets the best forward/backward, up/down, and yaw velocities for tracking an object
      * Args:
        * ```x1 (int)```: X-coordinate of the top-left corner of the object
        * ```y1 (int)```: Y-coordinate of the top-left corner of the object
        * ```x2 (int)```: X-coordinate of the bottom-right corner of the object
        * ```y2 (int)```: Y-coordinate of the bottom-right corner of the object
      * Returns (```Tuple[int, int, int, int]```): The first element is always zero, and after that, it's forward/backward, up/down, and yaw velocities for tracking the specified object
  * ```FaceTracker```: Communicates with the Tello and tracks faces
    * ```__init__```: Connects to the drone and sets up the face detector
      * Args:
        * ```proto_file (str)```: Path to the .prototxt file that contains the model's textual description. Default is  ```deploy.prototxt.txt```
        * ```model_file (str)```: Path to the model's parameters. Default is ```res10_300x300_ssd_iter_140000.caffemodel```
    * ```get_frame```: Gets the camera's current frame, resized to 300 X 300
      * Returns (```array```): The current frame as a NumPy ```array```, resized to 300 X 300
    * ```track_face```: Tracks faces in the current video frame

