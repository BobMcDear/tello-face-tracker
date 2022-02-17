from typing import Tuple

from cv2 import imshow, rectangle, resize
from djitellopy import Tello
from numpy import array

from face_detection import get_caffe_net, get_most_confident_face


class Controls:
    """
    Evaluates the best set of controls given face coordinates
    """
    @staticmethod
    def get_forward_backward_velocity(
        x1: int, 
        y1: int, 
        x2: int, 
        y2: int,
        ) -> int:
        """
        Evaluates the best forward-backward velocity value for tracking a face

        Args:
            x1 (int): X-coordinate of the top-left corner of the face
            y1 (int): Y-coordinate of the top-left corner of the face
            x2 (int): X-coordinate of the bottom-right corner of the face
            y2 (int): Y-coordinate of the bottom-right corner of the face
        
        Returns (int): Best forward-backward velocity value for tracking the face
        """
        height = y2-y1
        width = x2-x1
        area = height*width

        if (area == 0) or (6000 < area < 17000):
            forward_backward_velocity = 0

        elif area <= 6000:
            forward_backward_velocity = 20
        
        else:
            forward_backward_velocity = -20

        return forward_backward_velocity
    
    @staticmethod
    def get_up_down_velocity(
        y1: int, 
        y2: int,
        ) -> int:
        """
        Evaluates the best up-down velocity value for tracking a face

        Args:
            y1 (int): Y-coordinate of the top-left corner of the face
            y2 (int): Y-coordinate of the bottom-right corner of the face
        
        Returns (int): Best up-down velocity value for tracking the face
        """
        y_mid = (y1+y2)//2

        if (y_mid == 0) or (70 < y_mid < 135):
            up_down_velocity = 0

        elif y_mid <= 70:
            up_down_velocity = 15
        
        else:
            up_down_velocity = -15

        return up_down_velocity
    
    @staticmethod
    def get_yaw_velocity(
        x1: int, 
        x2: int, 
        ) -> int:
        """
        Evaluates the best yaw velocity value for tracking a face

        Args:
            x1 (int): X-coordinate of the top-left corner of the face
            x2 (int): X-coordinate of the bottom-right corner of the face
        
        Returns (int): Best yaw velocity value for tracking the face
        """
        x_mid = (x1+x2)//2

        if (x_mid == 0) or (120 < x_mid < 180):
            yaw_velocity = 0

        elif x_mid <= 120:
            yaw_velocity = -25
        
        else:
            yaw_velocity = 25

        return yaw_velocity
    
    @staticmethod
    def get_rc_controls(
        x1: int, 
        y1: int, 
        x2: int, 
        y2: int,
        ) -> Tuple[int, int, int, int]:
        """
        Evaluates the best set of controls given face coordinates

        Args:
            x1 (int): X-coordinate of the top-left corner of the face
            y1 (int): Y-coordinate of the top-left corner of the face
            x2 (int): X-coordinate of the bottom-right corner of the face
            y2 (int): Y-coordinate of the bottom-right corner of the face
        """
        left_right_velocity = 0
        
        forward_backward_velocity = Controls.get_forward_backward_velocity(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            )
        
        if forward_backward_velocity == 0:
            up_down_velocity = Controls.get_up_down_velocity(
                y1=y1,
                y2=y2,
                )
        
        else:
            up_down_velocity = 0

        yaw_velocity = Controls.get_yaw_velocity(
            x1=x1,
            x2=x2,
            )

        return (
            left_right_velocity,
            forward_backward_velocity, 
            up_down_velocity, 
            yaw_velocity,
            )
    

class FaceTracker:
    """
    A Tello controller that tracks faces
    """
    def __init__(
        self,
        proto_file: str = 'deploy.prototxt.txt',
        model_file: str = 'res10_300x300_ssd_iter_140000.caffemodel',
        ) -> None:
        """
        Sets up the drone and face detection network

        Args:
            net (dnn_Net): A dnn_Net that detects faces
            image (array): An image to find faces in
        """
        self.tello = Tello()

        self.tello.connect()
        self.tello.streamon()

        print('Battery: ', self.tello.get_battery())

        self.tello.takeoff()
        self.tello.send_rc_control(0, 0, 0, 0)

        self.net = get_caffe_net(
            proto_file=proto_file,
            model_file=model_file,
            )
        
    def get_frame(
        self,
        ) -> array:
        """
        Current frame, resized to 300 X 300

        Returns (array): Current frame, resized to 300 X 300
        """
        image = self.tello.get_frame_read().frame
        image = resize(image, (300, 300))
        return image


    def track_face(
        self,
        ) -> None:
        """
        Tracks the main face in the current frame
        """
        image = self.get_frame()

        x1, y1, x2, y2 = get_most_confident_face(
            net=self.net,
            image=image,
            )

        rectangle(image, (x1, y1), (x2, y2), (0, 0, 255))
        imshow('Image', image)

        controls = Controls.get_rc_controls(
            x1=x1, 
            x2=x2, 
            y1=y1, 
            y2=y2,
            )

        self.tello.send_rc_control(*controls)
