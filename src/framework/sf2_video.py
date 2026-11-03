"""Videos show real emulator frames; synthetic test images are labelled in tests only."""
from pathlib import Path
import shutil,subprocess
import cv2,numpy as np

class Video:
    def __init__(self,path,fps):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.fps=float(fps);self.frames=0;self.writer=None
        if not np.isfinite(self.fps) or self.fps<=0:raise ValueError('Invalid video FPS')
    def write(self,rgb,lines=(),keys1=(),keys2=()):
        canvas=np.zeros((660,800,3),np.uint8)
        canvas[:560]=cv2.resize(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),(800,560),interpolation=cv2.INTER_NEAREST)
        for i,line in enumerate(lines[:3]):
            text=str(line)[:115]
            cv2.putText(canvas,text,(8,578+20*i),cv2.FONT_HERSHEY_SIMPLEX,.43,(245,245,245),1,cv2.LINE_AA)
        for base,port,active in ((8,1,keys1),(405,2,keys2)):
            cv2.putText(canvas,f'P{port}',(base,647),cv2.FONT_HERSHEY_SIMPLEX,.42,(230,230,230),1)
            for j,k in enumerate(('LEFT','RIGHT','UP','DOWN','X','Y','Z','A','B','C')):
                x=base+27+j*36;on=k in active
                cv2.rectangle(canvas,(x,630),(x+33,654),(215,215,215) if on else (40,40,40),-1)
                cv2.putText(canvas,k,(x+2,646),cv2.FONT_HERSHEY_SIMPLEX,.29,(0,0,0) if on else (220,220,220),1)
        if self.writer is None:
            self.writer=cv2.VideoWriter(str(self.path),cv2.VideoWriter_fourcc(*'mp4v'),self.fps,(800,660))
            if not self.writer.isOpened():raise RuntimeError('Video writer could not open')
        self.writer.write(canvas);self.frames+=1
    def close(self):
        if self.writer is not None:self.writer.release();self.writer=None
    def finish(self):
        self.close()
        if self.frames==0:return None
        exe=shutil.which('ffmpeg')
        if not exe:return str(self.path)
        dst=self.path.with_name(self.path.stem+'_playable.mp4')
        p=subprocess.run([exe,'-y','-loglevel','error','-i',str(self.path),'-c:v','libx264','-preset','veryfast','-crf','23','-pix_fmt','yuv420p','-movflags','+faststart',str(dst)],capture_output=True,timeout=240)
        if p.returncode==0:return str(dst)
        return str(self.path)
