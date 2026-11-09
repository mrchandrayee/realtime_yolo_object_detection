from fastapi import FastAPI,UploadFile,File,HTTPException
from fastapi.responses import HTMLResponse,FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import uuid
from .config import CONFIG,UPLOAD_DIR,OUTPUT_DIR
from .processor import process_video
app=FastAPI(title=CONFIG["app"]["title"])
app.mount("/static",StaticFiles(directory=Path(__file__).parent/"static"),name="static")
detector=None;detector_error=None
@app.on_event("startup")
def startup():
    global detector,detector_error
    try:
        from .detector import ObjectDetector
        detector=ObjectDetector()
    except Exception as exc: detector_error=str(exc)
@app.get("/",response_class=HTMLResponse)
def home(): return (Path(__file__).parent/"static"/"index.html").read_text(encoding="utf-8")
@app.get("/api/status")
def status(): return {"ready":detector is not None,"message":"YOLO model ready" if detector else detector_error}
@app.post("/api/process")
async def process(video:UploadFile=File(...)):
    if detector is None: raise HTTPException(503,detector_error or "Detector is unavailable")
    limit=CONFIG["app"]["max_upload_mb"]*1024*1024
    suffix=Path(video.filename or "clip.mp4").suffix or ".mp4"
    path=UPLOAD_DIR/f"{uuid.uuid4().hex}{suffix}";size=0
    with path.open("wb") as f:
        while chunk:=await video.read(1024*1024):
            size+=len(chunk)
            if size>limit:
                path.unlink(missing_ok=True);raise HTTPException(413,"Video exceeds configured limit")
            f.write(chunk)
    try:
        report=process_video(path,detector,f"{uuid.uuid4().hex}.mp4")
        report["video_url"]="/api/files/"+report["video_file"]
        report["csv_url"]="/api/files/"+report["csv_file"]
        report["json_url"]="/api/files/report.json"
        return report
    except Exception as exc: raise HTTPException(400,f"Processing failed: {exc}")
    finally: path.unlink(missing_ok=True)
@app.get("/api/files/{filename}")
def get_file(filename:str):
    p=OUTPUT_DIR/Path(filename).name
    if not p.exists(): raise HTTPException(404,"File not found")
    return FileResponse(p,filename=p.name)
