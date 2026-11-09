"""Make an unmistakably synthetic input contact sheet; not a paper figure."""
from pathlib import Path
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def safe(path):
    p=Path(path).absolute()
    if os.name=="nt" and not str(p).startswith("\\\\?\\"):
        return Path("\\\\?\\"+str(p))
    return p

root=safe(Path(__file__).parent)
data=np.load(root/"SYNTHETIC_sample.npz",allow_pickle=False)
assert bool(data["synthetic_data"])
canvas=Image.new("RGB",(656,600),"white")
draw=ImageDraw.Draw(canvas)
big=ImageFont.load_default(size=25)
small=ImageFont.load_default(size=15)
draw.rectangle((0,0,656,88),fill="#9C3333")
draw.text((20,12),"SYNTHETIC DEMO - NOT REAL SONAR",font=big,fill="white")
draw.text((20,52),"Artificial images. Not publication evidence.",font=small,fill="white")
for n,i in enumerate([0,1,2,18,19,20]):
    left=20+(n%3)*212;top=105+(n//3)*230
    tile=Image.fromarray((data["images"][i,0]*255).round().astype("uint8")).convert("RGB")
    tile=tile.resize((192,192),Image.Resampling.NEAREST)
    canvas.paste(tile,(left,top))
    draw=ImageDraw.Draw(canvas)
    for cx,cy,w,h in data["boxes"][i,:int(data["counts"][i])]:
        box=((cx-w/2)*192+left,(cy-h/2)*192+top,(cx+w/2)*192+left,(cy+h/2)*192+top)
        draw.rectangle(box,outline="#F1CC46",width=2)
    draw.text((left,top+197),f'{data["split"][i]} / W{i:02d} / GT={data["counts"][i]}',font=small,fill="#222222")
draw.text((20,575),"Yellow boxes are artificial labels; source images are 24 x 24.",font=small,fill="#9C3333")
canvas.save(root/"SYNTHETIC_preview.png")
print("Created synthetic-only preview")
