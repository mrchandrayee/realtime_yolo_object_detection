from ultralytics import YOLO
from .config import CONFIG

class ObjectDetector:
    def __init__(self):
        self.model=YOLO(CONFIG["detector"]["model"])

    def track(self, frame):
        opts={"source":frame,"conf":float(CONFIG["detector"]["confidence"]),
              "iou":float(CONFIG["detector"]["iou"]),
              "imgsz":int(CONFIG["detector"]["image_size"]),
              "persist":True,"tracker":CONFIG["detector"]["tracker"],"verbose":False}
        device=CONFIG["detector"].get("device","")
        if device: opts["device"]=device
        result=self.model.track(**opts)[0]
        names=result.names
        detections=[]
        for box in result.boxes:
            cls_id=int(box.cls[0]); label=names.get(cls_id,str(cls_id)) if isinstance(names,dict) else names[cls_id]
            track_id=int(box.id[0]) if box.id is not None else None
            x1,y1,x2,y2=map(int,box.xyxy[0].tolist())
            detections.append({"bbox":[x1,y1,x2,y2],"class_id":cls_id,"label":str(label),
                               "confidence":float(box.conf[0]),"track_id":track_id})
        return detections
