"""그래서비트 배경음악 생성기 (직접 만든 곡이라 저작권 걱정 없음).

사용: python3 templates/bgm_library.py [--out assets/bgm]
- calm_*.wav : 잔잔한 곡 (피아노 아르페지오 + 패드), 정보형 릴스용
- epic_*.wav : 웅장한 곡 (현악 오스티나토 + 저음 패드 + 팀파니), 후킹·예능형용
곡마다 조성·코드 진행·템포·악기 조합이 다릅니다. 어떤 영상에 어떤 곡을 썼는지는
assets/bgm/usage.json 에 기록하고, 이미 쓴 곡은 다시 쓰지 않습니다.
"""
import numpy as np, wave, argparse, os
from scipy.signal import butter, sosfilt, sawtooth

SR = 44100
rng = np.random.default_rng(0)

def hz(m): return 440 * 2 ** ((m - 69) / 12)
def lp(x, c): return sosfilt(butter(2, c / (SR / 2), output='sos'), x)
def hp(x, c): return sosfilt(butter(2, c / (SR / 2), btype='high', output='sos'), x)
def env(n, a, r):
    e = np.ones(n); a = int(a * SR); r = int(r * SR)
    if a: e[:a] = np.linspace(0, 1, a)
    if r: e[-r:] *= np.linspace(1, 0, r)
    return e

class Track:
    def __init__(self, seconds):
        self.n = int(seconds * SR); self.x = np.zeros(self.n)
    def add(self, y, at):
        s = int(at * SR); e = min(self.n, s + len(y))
        if s < self.n: self.x[s:e] += y[:e - s]

def pad(chord, dur, bright=1600, amp=0.5, saw=False):
    n = int(dur * SR); t = np.arange(n) / SR; s = np.zeros(n)
    for m in chord:
        for d in (-0.12, 0, 0.12):
            f = hz(m) * (1 + d / 100)
            s += (sawtooth(2 * np.pi * f * t) * 0.5 if saw else np.sin(2 * np.pi * f * t) + 0.25 * np.sin(4 * np.pi * f * t))
    return lp(s, bright) * env(n, dur * 0.3, dur * 0.35) * amp / (len(chord) * 3)

def pluck(m, dur=1.4, amp=0.16, decay=3.2):
    n = int(dur * SR); t = np.arange(n) / SR; f = hz(m)
    return (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t) + 0.1 * np.sin(6 * np.pi * f * t)) * np.exp(-t * decay) * amp

def string_note(m, dur, amp=0.12):
    n = int(dur * SR); t = np.arange(n) / SR; f = hz(m)
    vib = 1 + 0.004 * np.sin(2 * np.pi * 5.5 * t)
    s = sawtooth(2 * np.pi * f * np.cumsum(vib) / SR * SR / SR * 1.0) if False else sawtooth(2 * np.pi * np.cumsum(f * vib) / SR)
    return lp(s, 2600) * env(n, 0.02, dur * 0.4) * amp

def timpani(m=38, amp=0.7):
    n = int(1.6 * SR); t = np.arange(n) / SR
    f = hz(m) * (1 + 0.25 * np.exp(-t * 20))
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 3.0)
    hit = lp(rng.normal(0, 1, n), 900) * np.exp(-t * 25) * 0.4
    return (body + hit) * amp

def boom(amp=0.9):
    n = int(2.5 * SR); t = np.arange(n) / SR
    f = 30 + 60 * np.exp(-t * 6)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 1.6) * amp

def swell(dur=2.0, amp=0.25):
    n = int(dur * SR); s = hp(rng.normal(0, 1, n), 4000)
    return s * np.linspace(0, 1, n) ** 3 * amp

def to_wav(path, x):
    x = x / (np.max(np.abs(x)) + 1e-9) * 0.85
    st = np.stack([x, x], 1)
    with wave.open(path, 'wb') as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((st * 32767).astype(np.int16).tobytes())

PROGS = {  # 단조 위주 코드 진행 (루트 기준 반음 간격)
    'a': [(0, 'm'), (8, 'M'), (3, 'M'), (10, 'M')],   # i  VI  III VII
    'b': [(0, 'm'), (5, 'm'), (8, 'M'), (7, 'M')],    # i  iv  VI  V
    'c': [(0, 'm'), (10, 'M'), (8, 'M'), (7, 'M')],   # i  VII VI  V
    'd': [(0, 'M'), (9, 'm'), (5, 'M'), (7, 'M')],    # I  vi  IV  V (밝은 잔잔함)
}
def chord(root, q):
    return [root, root + (3 if q == 'm' else 4), root + 7, root + 12]

def calm(key, prog, bpm, seconds=48, pattern=(0, 2, 1, 3, 2, 1, 0, 2), bright=1800):
    T = Track(seconds); bar = 4 * 60 / bpm; t0 = 0.0; i = 0
    while t0 < seconds:
        r, q = PROGS[prog][i % 4]; c = chord(key + r, q)
        T.add(pad(c, bar + 0.5, bright, 0.55), t0)
        for k, p in enumerate(pattern):
            T.add(pluck(c[p] + 12, amp=0.14 + 0.03 * (k == 0)), t0 + k * bar / len(pattern))
        T.add(pluck(c[0] - 12, 3.0, 0.12, 1.2), t0)  # 저음
        t0 += bar; i += 1
    x = lp(T.x, 5000) + hp(rng.normal(0, 1, T.n), 3000) * 0.003
    f = int(2.5 * SR); x[:f] *= np.linspace(0, 1, f); x[-f:] *= np.linspace(1, 0, f)
    return x

def epic(key, prog, bpm, seconds=40, ostinato=(0, 0, 2, 0, 1, 0, 2, 3)):
    T = Track(seconds); beat = 60 / bpm; bar = 4 * beat; t0 = 0.0; i = 0
    while t0 < seconds:
        r, q = PROGS[prog][i % 4]; c = chord(key + r, q)
        T.add(pad([c[0] - 12, c[1] - 12, c[2] - 12], bar + 0.4, 900, 0.9, saw=True), t0)   # 저음 금관 느낌
        T.add(pad(c, bar + 0.4, 2400, 0.45), t0)                                           # 합창 느낌 패드
        if i >= 1:   # 둘째 마디부터 현악 오스티나토
            for k in range(8):
                T.add(string_note(c[ostinato[k]] + 12, beat / 2 * 0.95), t0 + k * beat / 2)
        if i >= 2:   # 셋째 마디부터 타악
            T.add(timpani(key + r - 24 + 12), t0); T.add(timpani(key + r - 24 + 12, 0.45), t0 + 2.5 * beat)
        if i % 4 == 3: T.add(swell(bar * 0.9), t0 + bar * 0.1)
        if i % 4 == 0 and i > 0: T.add(boom(), t0)
        t0 += bar; i += 1
    x = lp(T.x, 9000)
    f = int(1.5 * SR); x[:f] *= np.linspace(0, 1, f); x[-int(2 * SR):] *= np.linspace(1, 0, int(2 * SR))
    return x

LIBRARY = [
    ('calm_01_a단조_72', lambda: calm(57, 'a', 72)),
    ('calm_02_d단조_64', lambda: calm(62, 'b', 64, pattern=(0, 1, 2, 3, 2, 1, 2, 1), bright=1500)),
    ('calm_03_f장조_78', lambda: calm(53, 'd', 78, pattern=(0, 2, 3, 2, 1, 2, 3, 2), bright=2200)),
    ('calm_04_e단조_68', lambda: calm(52, 'c', 68, pattern=(3, 2, 1, 0, 1, 2, 1, 2))),
    ('epic_01_c단조_96', lambda: epic(48, 'c', 96)),
    ('epic_02_d단조_88', lambda: epic(50, 'b', 88, ostinato=(0, 2, 0, 3, 0, 2, 1, 2))),
    ('epic_03_g단조_104', lambda: epic(55, 'a', 104, ostinato=(0, 1, 2, 1, 0, 1, 2, 3))),
    ('epic_04_e단조_92', lambda: epic(52, 'b', 92, ostinato=(2, 0, 2, 0, 3, 0, 1, 0))),
]

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default='assets/bgm'); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for name, fn in LIBRARY:
        to_wav(os.path.join(a.out, name + '.wav'), fn()); print('만듦', name)
