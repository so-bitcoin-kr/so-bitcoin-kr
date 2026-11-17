"""그래서비트 후킹형·예능형 릴스 배경음악 (직접 합성, 저작권 문제 없음)

숏폼 후킹 영상에서 많이 쓰이는 '브라질 펑크/폰크' 계열의 소리 요소
(808 카우벨 멜로디, 찌그러진 808 베이스, 강한 킥·클랩, 16분 하이햇)를
직접 합성합니다. 특정 곡의 멜로디는 쓰지 않고, 곡마다 리프·조성·템포를 새로 짭니다.

사용법:
    python3 bgm_phonk.py            # LIBRARY 전부 생성 → /home/claude/bgmlib/phonk_*.wav
    from bgm_phonk import build       # 영상 타이밍(드롭 위치)에 맞춰 직접 생성

영상 타이밍 맞추기: build(..., events=[('impact', 0.0), ('break', 9.6, 10.5), ('impact', 10.5)])
  - impact t : 그 순간 쿵 소리 + 크래시 (훅 첫 프레임, 반전 공개 순간)
  - break a b: a~b 사이 드럼·베이스를 빼고 카우벨만 필터로 먹먹하게 + 상승음 (긴장)
"""
import numpy as np
from scipy.signal import butter, sosfilt, fftconvolve
from scipy.io import wavfile

SR = 44100
rng = np.random.default_rng(7)

NOTE = {n: i for i, n in enumerate(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])}
def hz(name):
    n, o = name[:-1], int(name[-1])
    return 440.0 * 2 ** ((NOTE[n] + 12 * (o + 1) - 69) / 12)

def bp(x, lo, hi, order=2):
    return sosfilt(butter(order, [lo, hi], btype='band', fs=SR, output='sos'), x)
def hp(x, f, order=2):
    return sosfilt(butter(order, f, btype='high', fs=SR, output='sos'), x)
def lp(x, f, order=2):
    return sosfilt(butter(order, f, btype='low', fs=SR, output='sos'), x)

def sq(f, n):
    t = np.arange(n) / SR
    return np.sign(np.sin(2 * np.pi * f * t))

# ── 악기 ─────────────────────────────────────────────
def cowbell(f, dur=0.32):
    """808 카우벨: 두 사각파(비율 1.48) + 대역 필터 + 빠른 감쇠"""
    n = int(dur * SR); t = np.arange(n) / SR
    x = 0.6 * sq(f, n) + 0.4 * sq(f * 1.48, n)
    x = bp(x, f * 0.9, min(f * 4.5, 9000))
    env = np.exp(-t * 9) * 0.8 + np.exp(-t * 40) * 0.6
    return x * env

def bass808(f, dur, glide_to=None):
    """찌그러진 808: 사인 + 음정 하강 어택 + 포화(휴대폰 스피커에서도 들리게 배음 생성)"""
    n = int(dur * SR); t = np.arange(n) / SR
    fr = np.full(n, f)
    fr *= 1 + 1.2 * np.exp(-t * 60)                     # 어택 펀치
    if glide_to:
        g = np.clip((t - dur * 0.65) / (dur * 0.3), 0, 1)
        fr = fr * (1 - g) + glide_to * g
    ph = 2 * np.pi * np.cumsum(fr) / SR
    x = np.sin(ph)
    env = np.minimum(1, t / 0.004) * np.exp(-t * 1.6)
    x = np.tanh(4.8 * x * env) * 0.9
    return x

def kick(dur=0.35):
    n = int(dur * SR); t = np.arange(n) / SR
    f = 48 + 140 * np.exp(-t * 32)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 9)
    click = hp(rng.standard_normal(n), 2500) * np.exp(-t * 300) * 0.4
    return np.tanh(2.2 * (x + click))

def clap(dur=0.28):
    n = int(dur * SR); t = np.arange(n) / SR
    nz = rng.standard_normal(n)
    env = np.zeros(n)
    for d in (0, 0.011, 0.022):                          # 손뼉 3겹
        k = int(d * SR); env[k:] += np.exp(-(t[:n - k]) * 70)
    env += np.exp(-t * 14) * 0.5
    snap = np.sin(2 * np.pi * 190 * t) * np.exp(-t * 35) * 0.5
    return bp(nz, 900, 6000) * env * 0.9 + snap

def hat(dur=0.06, open_=False):
    n = int((0.22 if open_ else dur) * SR); t = np.arange(n) / SR
    x = hp(rng.standard_normal(n), 7000)
    return x * np.exp(-t * (14 if open_ else 90)) * (0.5 if open_ else 0.35)

def boom(dur=1.6):
    n = int(dur * SR); t = np.arange(n) / SR
    f = 32 + 90 * np.exp(-t * 8)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 2.2)
    nz = lp(rng.standard_normal(n), 900) * np.exp(-t * 6) * 0.5
    return np.tanh(2.5 * (x + nz))

def crash(dur=1.8):
    n = int(dur * SR); t = np.arange(n) / SR
    return hp(rng.standard_normal(n), 4000) * np.exp(-t * 2.5) * 0.35

def riser(dur):
    n = int(dur * SR); t = np.arange(n) / SR
    nz = rng.standard_normal(n)
    out = np.zeros(n); seg = 2048
    for i in range(0, n, seg):                           # 점점 밝아지는 노이즈
        c = 400 + 7000 * (i / n) ** 2
        out[i:i + seg] = bp(nz[i:i + seg], c, min(c * 2.2, 18000))
    return out * (t / dur) ** 2 * 0.6

def reverb(x, sec=1.4, mix=0.25):
    n = int(sec * SR); t = np.arange(n) / SR
    ir = rng.standard_normal(n) * np.exp(-t * 4.5)
    ir = lp(ir, 5000); ir /= np.abs(ir).sum() ** 0.5 * 30
    wet = fftconvolve(x, ir)[:len(x)]
    return x * (1 - mix) + wet * mix * 6

def place(buf, x, t0, gain=1.0):
    i = int(t0 * SR)
    if i >= len(buf) or i + len(x) <= 0: return
    j = min(len(buf), i + len(x)); s = max(0, -i)
    buf[max(i, 0):j] += x[s:j - i] * gain

# ── 곡 만들기 ──────────────────────────────────────────
def build(dur, bpm, riff, bassline, events=(), swing=0.0, hats='16', seed=7):
    """riff: 마디별 [(16분 위치, 음), ...] 목록(반복). bassline: 마디별 [(16분 위치, 음, 길이16분, 글라이드음|None)]"""
    global rng; rng = np.random.default_rng(seed)
    n = int(dur * SR) + SR
    cb, bs, dr, fx = (np.zeros(n) for _ in range(4))
    s16 = 60 / bpm / 4
    breaks = [(e[1], e[2]) for e in events if e[0] == 'break']
    in_break = lambda t: any(a <= t < b for a, b in breaks)
    kick_times = []
    bar = 0; tbar = 0.0
    while tbar < dur:
        r = riff[bar % len(riff)]; b = bassline[bar % len(bassline)]
        for step, note in r:
            t = tbar + step * s16 + (swing * s16 if step % 2 else 0)
            place(cb, cowbell(hz(note)), t, 1.25)
        for step, note, ln, gl in b:
            t = tbar + step * s16
            if in_break(t): continue
            place(bs, bass808(hz(note), ln * s16, hz(gl) if gl else None), t, 0.5)
            place(dr, kick(), t, 0.9); kick_times.append(t)
        for beat in (4, 12):                             # 2·4박 클랩
            t = tbar + beat * s16
            if not in_break(t): place(dr, clap(), t, 0.7)
        for k in range(16):
            t = tbar + k * s16
            if in_break(t): continue
            if hats == '16' or k % 2 == 0:
                place(dr, hat(), t, 0.9 if k % 4 == 2 else 0.55)
            if bar % 4 == 3 and k >= 12:                 # 4마디마다 롤
                for r3 in range(3): place(dr, hat(), t + r3 * s16 / 3, 0.5)
        bar += 1; tbar += 16 * s16

    # 카우벨: 약한 리버브(‘슬로우+리버브’ 느낌) + 킥에 맞춘 펌핑
    cb = reverb(cb, 1.6, 0.3)
    duck = np.ones(n); tt = np.arange(int(0.22 * SR)) / SR
    for kt in kick_times:
        i = int(kt * SR); seg = 1 - 0.45 * np.exp(-tt * 14)
        duck[i:i + len(seg)] = np.minimum(duck[i:i + len(seg)], seg[:max(0, n - i)])
    cb *= duck

    for e in events:
        if e[0] == 'impact':
            place(fx, boom(), e[1], 0.9); place(fx, crash(), e[1], 0.8)
        if e[0] == 'break':
            a, b = e[1], e[2]; ia, ib = int(a * SR), int(b * SR)
            cb[ia:ib] = lp(cb[ia:ib], 700) * 1.4           # 먹먹하게
            place(fx, riser(b - a), a, 0.9)

    mix = cb + bs + dr * 0.95 + fx
    mix = hp(mix, 28)
    mix = np.tanh(1.5 * mix / (np.abs(mix).max() + 1e-9)) # 마스터 포화
    mix = mix[:int(dur * SR)]
    fo = int(1.2 * SR); mix[-fo:] *= np.linspace(1, 0, fo)
    # 스테레오: 카우벨 리버브 성분만 살짝 넓게
    L = mix + 0.02 * np.roll(mix, 300); R = mix - 0.02 * np.roll(mix, 300)
    st = np.stack([L, R], 1); st /= np.abs(st).max() / 0.95
    return st.astype(np.float32)

def save(path, st):
    wavfile.write(path, SR, (st * 32767).astype(np.int16))

# ── 곡 목록 (곡마다 리프·조성·템포가 다름. 한 번 쓴 곡은 assets/bgm/usage.json에 기록) ──
LIBRARY = {
    # F단조 128BPM — 피자데이 v2 (11/18)
    'phonk_01_f단조_128': dict(bpm=128, riff=[
        [(0, 'C5'), (3, 'C5'), (6, 'G#4'), (8, 'C5'), (10, 'C#5'), (12, 'C5'), (14, 'G#4')],
        [(0, 'A#4'), (3, 'A#4'), (6, 'G4'), (8, 'A#4'), (10, 'C5'), (12, 'A#4'), (14, 'F4')],
        [(0, 'C5'), (3, 'C5'), (6, 'G#4'), (8, 'C5'), (10, 'D#5'), (12, 'C#5'), (14, 'C5')],
        [(0, 'G4'), (2, 'G#4'), (4, 'A#4'), (6, 'C5'), (8, 'G4'), (11, 'F4'), (14, 'G4')]],
        bassline=[[(0, 'F1', 6, None), (6, 'F1', 4, None), (10, 'F1', 6, 'C#2')],
                  [(0, 'C#2', 6, None), (6, 'C#2', 4, None), (10, 'D#2', 6, None)],
                  [(0, 'F1', 6, None), (6, 'F1', 4, None), (10, 'G#1', 6, None)],
                  [(0, 'C2', 6, None), (6, 'C2', 4, None), (10, 'C2', 6, 'F1')]]),
    # G단조 140BPM — 예능형 부장님 레버리지 v2 (11/20)
    'phonk_02_g단조_140': dict(bpm=140, riff=[
        [(0, 'D5'), (2, 'D5'), (4, 'A#4'), (7, 'D5'), (8, 'F5'), (10, 'D5'), (13, 'A#4')],
        [(0, 'C5'), (2, 'C5'), (4, 'A4'), (7, 'C5'), (8, 'D5'), (10, 'C5'), (13, 'G4')],
        [(0, 'D5'), (2, 'D5'), (4, 'A#4'), (7, 'D5'), (8, 'G5'), (10, 'F5'), (13, 'D5')],
        [(0, 'D#5'), (3, 'D5'), (6, 'C5'), (8, 'A#4'), (10, 'A4'), (12, 'G4'), (14, 'A4')]],
        bassline=[[(0, 'G1', 4, None), (4, 'G1', 3, None), (7, 'G1', 5, None), (12, 'G1', 4, 'A#1')],
                  [(0, 'F1', 4, None), (4, 'F1', 3, None), (7, 'F1', 5, None), (12, 'F1', 4, None)],
                  [(0, 'G1', 4, None), (4, 'G1', 3, None), (7, 'G1', 5, None), (12, 'A#1', 4, None)],
                  [(0, 'D#2', 4, None), (4, 'D#2', 3, None), (7, 'D2', 5, None), (12, 'D2', 4, 'G1')]]),
    # A단조 120BPM — 예비
    'phonk_03_a단조_120': dict(bpm=120, riff=[
        [(0, 'E5'), (3, 'E5'), (6, 'C5'), (8, 'E5'), (10, 'F5'), (12, 'E5'), (14, 'B4')],
        [(0, 'D5'), (3, 'D5'), (6, 'B4'), (8, 'D5'), (10, 'E5'), (12, 'D5'), (14, 'A4')]],
        bassline=[[(0, 'A1', 6, None), (6, 'A1', 4, None), (10, 'A1', 6, 'F1')],
                  [(0, 'F1', 6, None), (6, 'G1', 4, None), (10, 'E1', 6, None)]]),
}

if __name__ == '__main__':
    import os
    os.makedirs('/home/claude/bgmlib', exist_ok=True)
    for name, cfg in LIBRARY.items():
        save(f'/home/claude/bgmlib/{name}.wav', build(20.0, events=[('impact', 0.0)], **cfg))
        print(name)
