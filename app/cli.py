import argparse,json
from .detector import ObjectDetector
from .processor import process_video
def main():
    p=argparse.ArgumentParser(description="YOLO object detection and tracking on a video")
    p.add_argument("video");p.add_argument("--output",default="annotated_detection.mp4")
    a=p.parse_args()
    print(json.dumps(process_video(a.video,ObjectDetector(),a.output),indent=2))
if __name__=="__main__":main()
