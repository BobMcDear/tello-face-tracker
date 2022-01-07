from argparse import ArgumentParser

from cv2 import waitKey

from face_tracking import FaceTracker


def main(
    proto_file: str = 'deploy.prototxt.txt',
    model_file: str = 'res10_300x300_ssd_iter_140000.caffemodel',
    ) -> None:
    face_tracker = FaceTracker(
        proto_file=proto_file,
        model_file=model_file,
        )

    while True:
        face_tracker.track_face()
        waitKey(1)


if __name__ == '__main__':
    parser = ArgumentParser(description='Tracks faces with Tello drone')

    parser.add_argument(
        '--proto_file',
        type=str,
        help='Path to the model proto file',
        default='deploy.prototxt.txt',
        )
    parser.add_argument(
        '--model_file',
        type=str,
        help='Path to the model file',
        default='res10_300x300_ssd_iter_140000.caffemodel',
        )
    
    args = parser.parse_args()

    main(
        proto_file=args.proto_file,
        model_file=args.model_file,
        )
