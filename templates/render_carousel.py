from playwright.sync_api import sync_playwright
from PIL import Image
import os
A=[f'a{i}' for i in range(1,8)]; B=[f'b{i}' for i in range(1,7)]
os.makedirs('out',exist_ok=True)
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(viewport={'width':1080,'height':1350})
    pg.goto('file:///home/claude/intro/slides.html'); pg.wait_for_timeout(600)
    for i in A+B:
        el=pg.locator('#'+i)
        # overflow check: any child extends past slide bottom-160?
        over=pg.evaluate("""(id)=>{const s=document.getElementById(id);const r=s.getBoundingClientRect();let worst=0;
          s.querySelectorAll('h1,.body,.rows,.chain,.cap,.warn').forEach(e=>{const b=e.getBoundingClientRect().bottom-r.top;if(b>worst)worst=b});
          s.querySelectorAll('*').forEach(e=>{if(e.scrollWidth>e.clientWidth+2 && getComputedStyle(e).overflow!=='visible')worst=9999});return worst}""", i)
        print(i,'content bottom',round(over))
        el.screenshot(path=f'out/{i}.png')
        prefix='소개' if i[0]=='a' else '원칙'
        Image.open(f'out/{i}.png').convert('RGB').save(f'out/{prefix}_{i[1:]}.jpg',quality=95)
    b.close()
# contact sheets
for grp,name in ((A,'소개'),(B,'원칙')):
    ims=[Image.open(f'out/{i}.png').resize((360,450)) for i in grp]
    cols=4; rows=(len(ims)+cols-1)//cols
    sheet=Image.new('RGB',(cols*370+10,rows*460+10),'#ddd')
    for k,im in enumerate(ims): sheet.paste(im,(10+(k%cols)*370,10+(k//cols)*460))
    sheet.save(f'out/preview_{name}.png')
