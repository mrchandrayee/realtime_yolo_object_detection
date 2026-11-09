import cv2,csv,json
from collections import Counter
from .config import CONFIG,OUTPUT_DIR

def process_video(source,detector,output_name):
    cap=cv2.VideoCapture(str(source))
    if not cap.isOpened(): raise ValueError("Could not open video.")
    fps=cap.get(cv2.CAP_PROP_FPS) or CONFIG["video"]["fallback_fps"]
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target=OUTPUT_DIR/output_name
    writer=cv2.VideoWriter(str(target),cv2.VideoWriter_fourcc(*"mp4v"),fps,(width,height))
    if not writer.isOpened():
        cap.release();raise RuntimeError("Could not create output MP4.")
    frame_no=0;events=[];counts=Counter();step=max(1,int(CONFIG["video"]["process_every_n_frames"]))
    max_frames=int(CONFIG["video"]["max_frames"])
    while True:
        ok,frame=cap.read()
        if not ok or (max_frames and frame_no>=max_frames): break
        if frame_no%step==0:
            detections=detector.track(frame)
            for d in detections:
                x1,y1,x2,y2=d["bbox"]; label=d["label"]; conf=d["confidence"]; tid=d["track_id"]
                counts[label]+=1
                color=(55,220,120)
                cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                title=f"{label} {conf:.2f}"+(f" ID:{tid}" if tid is not None else "")
                top=max(0,y1-24); right=min(width,x1+max(110,len(title)*9))
                cv2.rectangle(frame,(x1,top),(right,y1),color,-1)
                cv2.putText(frame,title,(x1+4,max(16,y1-7)),cv2.FONT_HERSHEY_SIMPLEX,.48,(8,20,18),1,cv2.LINE_AA)
                events.append({"time_seconds":round(frame_no/fps,3),"frame":frame_no,
                    "class":label,"track_id":tid,"confidence":round(conf,4),"bbox":d["bbox"]})
        writer.write(frame);frame_no+=1
    cap.release();writer.release()
    csv_path=OUTPUT_DIR/"detections.csv"
    with csv_path.open("w",newline="",encoding="utf-8") as f:
        wr=csv.DictWriter(f,fieldnames=["time_seconds","frame","class","track_id","confidence","bbox"])
        wr.writeheader()
        for e in events:
            row=e.copy();row["bbox"]=json.dumps(row["bbox"]);wr.writerow(row)
    report={"frames_processed":frame_no,"fps":fps,"duration_seconds":round(frame_no/fps,2),
            "detections_by_class":dict(counts),"event_count":len(events),
            "events":events,"video_file":target.name,"csv_file":csv_path.name}
    (OUTPUT_DIR/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report
