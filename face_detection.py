from typing import List, Tuple

from cv2 import dnn_Net
from cv2.dnn import blobFromImage, readNetFromCaffe
from numpy import array, squeeze


def get_caffe_net(
    proto_file: str = 'deploy.prototxt.txt',
    model_file: str = 'res10_300x300_ssd_iter_140000.caffemodel',
    ) -> dnn_Net:
    """
    Reads in a Caffe network with CV2 

    Args:
        proto_file (str): Path to the model proto file.
        Default is 'deploy.prototxt.txt'
        model_file (str): Path to the model file.
        Default is 'res10_300x300_ssd_iter_140000.caffemodel'
    
    Returns (dnn_Net): A dnn_Net from the provided proto and model files
    """
    caffe_net = readNetFromCaffe(proto_file, model_file)
    return caffe_net


def get_faces(
    net: dnn_Net,
    image: array,
    ) -> List[Tuple[int, int, int, int, float]]:
    """
    Detects faces in a picture with a dnn_Net

    Args:
        net (dnn_Net): A dnn_Net that detects faces
        image (array): An image to find faces in
    
    Returns (List[Tuple[int, int, int, int, float]]): A List containing Tuples
    of face coordinates (top-left X, top-left Y, bottom-right X, bottom-right 
    Y). The results always starts with 
    (0, 0, 0, 0, 0)
    """
    height, width = image.shape[:2]
    norm_stats = (104.0, 117.0, 123.0)

    blob = blobFromImage(
        image, 
        1.0,
        (height, width),
        norm_stats,
        )

    net.setInput(blob)
    faces = net.forward()

    faces = squeeze(faces, axis=(0, 1))
    faces = faces[:, 2:]

    results = [(0, 0, 0, 0, 0)]
    for i in range(len(faces)):
        curr_face = faces[i]
        confidence = curr_face[0]
        if 0.75 <= confidence:
            face = curr_face[1:] * array([width, height, width, height])
            face = face.astype('int')
            
            result = (*face, confidence)
            results.append(result)

    return results


def get_most_confident_face(
    net: dnn_Net,
    image: array,
    ) -> Tuple[int, int, int, int]:
    """
    Finds the face with the highest associated confidence in an image

    Args:
        net (dnn_Net): A dnn_Net that detects faces
        image (array): An image to find faces in
    
    Returns (Tuple[int, int, int, int]): A Tuple containing the coordinates of
    the face (top-left X, top-left Y, bottom-right X, bottom-right Y) with the
    highest associated confidence
    """
    faces = get_faces(
        net=net, 
        image=image,
        )
    
    key = lambda face: face[-1]
    most_confident_face = sorted(faces, key=key)[-1][:4]
    return most_confident_face
