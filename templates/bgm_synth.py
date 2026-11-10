import numpy as np, wave
SR=44100
def note(n): return 440*2**((n-69)/12)
def env(n,a,r,total):
    e=np.ones(total); a=int(a*SR); r=int(r*SR)
    if a>0: e[:a]=np.linspace(0,1,a)
    if r>0: e[-r:]*=np.linspace(1,0,r)
    return e
def lp(x,k=0.05):  # one-pole lowpass
    y=np.zeros_like(x); acc=0.0
    for i in range(len(x)):
        acc+=k*(x[i]-acc); y[i]=acc
    return y
def lp_fast(x,cut):
    from scipy.signal import butter,sosfilt
    return sosfilt(butter(2,cut/(SR/2),output='sos'),x)
def hp_fast(x,cut):
    from scipy.signal import butter,sosfilt
    return sosfilt(butter(2,cut/(SR/2),btype='high',output='sos'),x)
def pad(chord,dur,amp=0.12):
    n=int(dur*SR); t=np.arange(n)/SR; s=np.zeros(n)
    for m in chord:
        f=note(m)
        for d in (-0.15,0,0.15):
            s+=np.sin(2*np.pi*f*(1+d/100)*t)+0.3*np.sin(2*np.pi*2*f*t)
    s=lp_fast(s,1800)*env(n,dur*0.25,dur*0.3,n)
    return amp*s/len(chord)/3
def write(name,x):
    x=x/np.max(np.abs(x))*0.85
    st=np.stack([x,x],1)
    with wave.open(name,'wb') as w:
        w.setnchannels(2);w.setsampwidth(2);w.setframerate(SR)
        w.writeframes((st*32767).astype(np.int16).tobytes())
prog=[[57,60,64,67],[53,57,60,64],[48,55,60,64],[55,59,62,67]]  # Am7 Fmaj7 Cadd C/G-ish G
# --- calm: 잔잔한 패드 + 부드러운 피아노 아르페지오, 72 BPM
bar=4*60/72; out=[]
rng=np.random.default_rng(1)
for rep in range(4):
    for ch in prog:
        n=int(bar*SR); seg=np.zeros(n); seg+=pad(ch,bar,0.5)[:n]
        for k,m in enumerate([ch[0]+12,ch[2]+12,ch[1]+12,ch[3]+12,ch[2]+12,ch[1]+12,ch[0]+12,ch[2]+12]):
            st=int(k*bar/8*SR); L=int(1.2*SR); t=np.arange(L)/SR
            tone=(np.sin(2*np.pi*note(m)*t)+0.25*np.sin(2*np.pi*2*note(m)*t))*np.exp(-t*3.5)*0.18
            e=min(n,st+L); seg[st:e]+=tone[:e-st]
        out.append(seg)
calm=np.concatenate(out); calm=lp_fast(calm,4000)
calm+=hp_fast(rng.normal(0,1,len(calm)),3000)*0.004   # 아주 약한 바이닐 노이즈
f=int(2*SR); calm[:f]*=np.linspace(0,1,f); calm[-f:]*=np.linspace(1,0,f)
write('calm.wav',calm)
# --- hook: 로파이 힙합 비트, 92 BPM
bpm=92; beat=60/bpm; bar=4*beat; bars=12; n=int(bars*bar*SR); mix=np.zeros(n)
def add(x,at):
    s=int(at*SR); e=min(n,s+len(x)); mix[s:e]+=x[:e-s]
def kick():
    L=int(0.35*SR); t=np.arange(L)/SR; f=55+70*np.exp(-t*30)
    return np.sin(2*np.pi*np.cumsum(f)/SR)*np.exp(-t*9)*0.9
def snare():
    L=int(0.25*SR); t=np.arange(L)/SR
    return (hp_fast(rng.normal(0,1,L),1500)*0.35+np.sin(2*np.pi*190*t)*0.3)*np.exp(-t*18)
def hat(o=False):
    L=int((0.18 if o else 0.05)*SR); t=np.arange(L)/SR
    return hp_fast(rng.normal(0,1,L),7000)*np.exp(-t*(15 if o else 60))*0.18
for b in range(bars):
    t0=b*bar; ch=prog[b%4]
    add(pad(ch,bar,0.55),t0)
    # 베이스
    L=int(bar*SR); t=np.arange(L)/SR; bf=note(ch[0]-12)
    bass=np.tanh(1.5*np.sin(2*np.pi*bf*t))*0.28*env(L,0.01,0.2,L)
    add(lp_fast(bass,400),t0)
    if b>=1:  # 첫 마디는 패드만 → 2마디부터 비트 진입
        for k in (0,2.5): add(kick(),t0+k*beat)
        for k in (1,3): add(snare(),t0+k*beat)
        for k in range(8): add(hat(k==7),t0+k*beat/2+(0.02 if k%2 else 0))
    # 짧은 코드 스탭
    for k in (0.5,1.75,2.5):
        L=int(0.25*SR); tt=np.arange(L)/SR; s=np.zeros(L)
        for m in ch: s+=np.sin(2*np.pi*note(m+12)*tt)
        add(s*np.exp(-tt*12)*0.06,t0+k*beat)
mix=lp_fast(mix,9000); f=int(0.6*SR); mix[-f:]*=np.linspace(1,0,f)
write('hook.wav',mix)
print(len(calm)/SR, len(mix)/SR)
