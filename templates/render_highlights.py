from playwright.sync_api import sync_playwright
ids = [('intro','소개'),('schedule','일정'),('market','시장'),('basics','기초')]
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={'width':1080,'height':1920})
    pg.goto('file:///home/claude/highlight/covers.html'); pg.wait_for_timeout(500)
    for i,_ in ids:
        pg.locator('#'+i).screenshot(path=f'/home/claude/highlight/highlight_{i}.png')
    html = "<html><body style='margin:0;background:#fff;font-family:Noto Sans CJK KR'><div style='display:flex;gap:44px;padding:48px 56px'>"
    for i,lab in ids:
        html += f"<div style='text-align:center'><div style='width:150px;height:150px;border-radius:50%;padding:5px;border:2px solid #ddd'><img src='file:///home/claude/highlight/highlight_{i}.png' style='width:150px;height:150px;border-radius:50%;display:block;object-fit:cover'></div><div style='margin-top:14px;font-size:22px;color:#111'>{lab}</div></div>"
    html += "</div></body></html>"
    open('/home/claude/highlight/preview.html','w').write(html)
    pg2 = b.new_page(viewport={'width':900,'height':300}, device_scale_factor=2)
    pg2.goto('file:///home/claude/highlight/preview.html'); pg2.wait_for_timeout(500)
    pg2.screenshot(path='/home/claude/highlight/highlight_preview.png', full_page=True)
    b.close()
