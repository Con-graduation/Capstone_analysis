#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practice Evaluation MVP (Chord accuracy + Rhythm accuracy)
- 입력:
    --audio <path>
    --bpm <float>
    --chords "Em,C,G,D"   # 1~4개 (메이저/마이너만: A,Bb,B,... + 'm')
    [--beats-per-chord 4] # 한 코드가 차지하는 비트 수(기본 4)
    [--align-to-onsets 1] # 온셋 기반 박자 위상 정렬 사용(기본 1=사용)
- 출력(JSON stdout):
    {
      "summary": {...},
      "per_chord": [{"chord":"Em","frame_accuracy":0.82,"frames":1234,"duration_sec":12.3}, ...],
      "chord_overall_accuracy": 0.76,
      "rhythm": {
        "beat_grid_bpm": 92.0,
        "detected_bpm": 92.7,
        "tempo_stability_cv": 0.06,
        "timing_error_ms": {"mean":45.2,"p95":112.8},
        "drift_ms_per_min": 210.5,
        "score_0_100": 78
      }
    }
"""

import argparse, json, sys, math, os
from typing import List, Tuple, Dict, Optional
import numpy as np
import librosa
import scipy.signal as sps
import bisect

PC_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

# -------------- 유틸 --------------
def eprint(*a, **k): print(*a, file=sys.stderr, **k)

def normalize_chord_label(label: str) -> str:
    """
    "Am" -> "Am", "Amin" -> "Am", "A-" -> "Am", "Amaj"->"A", "Bb"->"A#"
    (단순 메이저/마이너 24개만 지원)
    """
    l = label.strip()
    l = l.replace("maj","").replace("M","")
    l = l.replace("min","m").replace("-","m")
    l = l.replace("Db","C#").replace("Eb","D#").replace("Gb","F#").replace("Ab","G#").replace("Bb","A#")
    if l.endswith("m"): return l[:-1].upper() + "m"
    return l.upper()

def build_chord_templates() -> Tuple[np.ndarray, List[str]]:
    triad_major = np.zeros(12); triad_major[[0,4,7]] = 1.0
    triad_minor = np.zeros(12); triad_minor[[0,3,7]] = 1.0
    templates, labels = [], []
    for i, root in enumerate(PC_NAMES):
        v = np.roll(triad_major, i); v = v / (np.linalg.norm(v)+1e-8)
        templates.append(v); labels.append(root)
    for i, root in enumerate(PC_NAMES):
        v = np.roll(triad_minor, i); v = v / (np.linalg.norm(v)+1e-8)
        templates.append(v); labels.append(root+"m")
    return np.stack(templates, axis=0), labels

def compute_chroma(y: np.ndarray, sr: int, hop: int) -> Tuple[np.ndarray, np.ndarray]:
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop, n_chroma=12)
    chroma = chroma / (chroma.sum(axis=0, keepdims=True)+1e-8)
    times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop)
    return chroma, times

def predict_chords(chroma: np.ndarray, templates: np.ndarray, thr: float=0.0) -> Tuple[np.ndarray, np.ndarray]:
    scores = templates @ chroma  # (24,T)
    idx = scores.argmax(axis=0)
    sc = scores.max(axis=0)
    if thr>0: idx = np.where(sc>=thr, idx, -1)
    return idx.astype(int), sc

def smooth_labels(labels_idx: np.ndarray, win: int=9) -> np.ndarray:
    if win%2==0: win+=1
    return sps.medfilt(labels_idx, kernel_size=win)

def onset_times_sec(y: np.ndarray, sr: int) -> np.ndarray:
    on_frames = librosa.onset.onset_detect(y=y, sr=sr, units="frames", backtrack=True)
    return librosa.frames_to_time(on_frames, sr=sr)

# -------------- 코드 세그먼트 --------------
def build_segments_repeating(chords: List[str], bpm: float, beats_per_chord: int,
                             duration: float, offset_sec: float=0.0) -> List[Dict]:
    """오디오 전체 길이를 덮을 때까지 코드 시퀀스를 반복해 세그먼트 생성"""
    segs = []
    t = offset_sec
    dur_per = beats_per_chord * 60.0 / bpm
    i = 0
    while t < duration:
        c = normalize_chord_label(chords[i % len(chords)])
        segs.append({"start": t, "end": min(t+dur_per, duration), "chord": c})
        t += dur_per; i += 1
    return segs

# -------------- 박자 위상 정렬 --------------
def estimate_phase_offset(onsets: np.ndarray, period: float) -> float:
    """[0, period)에서 위상을 grid search로 추정 (100 스텝)"""
    if len(onsets)==0: return 0.0
    best_phi, best_err = 0.0, 1e9
    steps = 120
    for k in range(steps):
        phi = (period * k) / steps
        # 각 온셋을 가장 가까운 비트에 붙여 본 오차
        diffs = []
        for o in onsets:
            n = round((o - phi) / period)
            beat_t = phi + n*period
            diffs.append(abs(o - beat_t))
        m = np.mean(diffs) if diffs else 1e9
        if m < best_err:
            best_err = m; best_phi = phi
    return best_phi

def match_onsets_to_grid(onsets: np.ndarray, grid: np.ndarray, max_dist: float) -> np.ndarray:
    """각 그리드 비트에 가장 가까운 온셋까지의 시간차(초), 임계 초과는 NaN"""
    deltas = np.full(len(grid), np.nan, dtype=float)
    for i, t in enumerate(grid):
        j = bisect.bisect_left(onsets, t)
        cand = []
        if j>0: cand.append(onsets[j-1])
        if j<len(onsets): cand.append(onsets[j])
        if cand:
            d = min(cand, key=lambda x: abs(x-t))
            if abs(d-t) <= max_dist:
                deltas[i] = d - t
    return deltas

# -------------- 리듬 메트릭 --------------
def rhythm_metrics(y: np.ndarray, sr: int, set_bpm: float, duration: float) -> Dict:
    period = 60.0 / set_bpm
    onsets = onset_times_sec(y, sr)
    onsets.sort()
    # 위상 정렬
    phi = estimate_phase_offset(onsets, period)
    # 그리드 생성
    grid = np.arange(phi, duration + period, period)
    # 비트별 가장 가까운 온셋과의 차
    deltas = match_onsets_to_grid(onsets, grid, max_dist=period/2.0)
    # 타이밍 오차
    valid = np.isfinite(deltas)
    mean_abs = float(np.nan) if not valid.any() else float(np.mean(np.abs(deltas[valid]))*1000.0)
    p95 = float(np.nan) if not valid.any() else float(np.percentile(np.abs(deltas[valid])*1000.0, 95))
    # 드리프트(분당 ms): deltas ~ a*t + b 선형회귀의 기울기 a(초당) → ms/min
    drift_ms_per_min = None
    if valid.sum() >= 5:
        t = grid[valid]
        x = np.vstack([t, np.ones_like(t)]).T
        yv = deltas[valid]
        a, b = np.linalg.lstsq(x, yv, rcond=None)[0]  # 초당 초
        drift_ms_per_min = float(a * 60.0 * 1000.0)
    # 검출 BPM 및 안정성(CV)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)
    det_bpm, det_beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, trim=False)
    beat_times = librosa.frames_to_time(det_beats, sr=sr)
    cv = None
    if len(beat_times) >= 4:
        ints = np.diff(beat_times)
        cv = float(np.std(ints) / (np.mean(ints)+1e-9))
    # 점수(0~100, 간이)
    score = 100.0
    if not math.isnan(mean_abs): score -= min(50.0, mean_abs*0.3)       # 10ms당 -3점 (대략)
    if not math.isnan(p95):      score -= min(25.0, (p95-80.0)*0.10) if p95>80 else 0.0
    if cv is not None:           score -= min(15.0, cv*150.0)           # CV 0.10 -> -15점
    if drift_ms_per_min is not None: score -= min(10.0, abs(drift_ms_per_min)*0.02)
    score = float(max(0.0, min(100.0, score)))

    return {
        "beat_grid_bpm": float(set_bpm),
        "phase_offset_sec": float(phi),
        "detected_bpm": float(det_bpm),
        "tempo_stability_cv": cv,
        "timing_error_ms": {"mean": mean_abs, "p95": p95},
        "drift_ms_per_min": drift_ms_per_min,
        "score_0_100": score
    }

# -------------- 코드 정확도 --------------
def per_chord_accuracy(pred_idx: np.ndarray, times: np.ndarray,
                       label_map: List[str], segments: List[Dict]) -> Tuple[List[Dict], float]:
    # 프레임별 GT 인덱스
    gt = np.full_like(pred_idx, -1)
    for seg in segments:
        s, e, lab = seg["start"], seg["end"], seg["chord"]
        mask = (times >= s) & (times < e)
        if lab in label_map:
            gt[mask] = label_map.index(lab)
    valid = gt >= 0
    overall = float((pred_idx[valid] == gt[valid]).sum() / max(1, valid.sum()))
    # 코드별 집계
    out = []
    for lab in sorted(set([seg["chord"] for seg in segments])):
        li = label_map.index(lab) if lab in label_map else None
        if li is None: continue
        mask = valid & (gt == li)
        frames = int(mask.sum())
        if frames == 0:
            acc = None; dur = 0.0
        else:
            acc = float((pred_idx[mask] == li).sum() / frames)
            tsel = times[mask]
            dur = float((tsel[-1] - tsel[0]) if len(tsel)>1 else 0.0)
        out.append({"chord": lab, "frame_accuracy": acc, "frames": frames, "duration_sec": dur})
    return out, overall

# -------------- 메인 --------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--bpm", type=float, required=True)
    ap.add_argument("--chords", type=str, required=True, help='예: "Em,C,G,D" (1~4개)')
    ap.add_argument("--beats-per-chord", type=int, default=4)
    ap.add_argument("--hop", type=int, default=2048)
    ap.add_argument("--thr", type=float, default=0.0, help="코드 라벨 신뢰도 임계 (0~1)")
    ap.add_argument("--smooth", type=int, default=9, help="라벨 중앙값 필터 크기(홀수)")
    ap.add_argument("--align-to-onsets", type=int, default=1, help="박자 위상 정렬 사용(1/0)")
    args = ap.parse_args()

    # 입력 로드
    y, sr = librosa.load(args.audio, sr=None, mono=True)
    duration = float(librosa.get_duration(y=y, sr=sr))

    # 코드 예측
    chroma, times = compute_chroma(y, sr, args.hop)
    templates, label_map = build_chord_templates()
    pred_idx, _ = predict_chords(chroma, templates, thr=args.thr)
    if args.smooth and args.smooth>1: pred_idx = smooth_labels(pred_idx, args.smooth)

    # 세그먼트 구성(위상 정렬 옵션 반영)
    period = 60.0/args.bpm
    phi = 0.0
    if args.align_to_onsets:
        ons = onset_times_sec(y, sr)
        phi = estimate_phase_offset(ons, period)
    chords_norm = [normalize_chord_label(c) for c in args.chords.split(",") if c.strip()]
    segments = build_segments_repeating(chords_norm, args.bpm, args.beats_per_chord, duration, offset_sec=phi)

    # 코드 정확도
    chord_list, chord_overall = per_chord_accuracy(pred_idx, times, label_map, segments)

    # 리듬 정확도
    rhythm = rhythm_metrics(y, sr, args.bpm, duration)

    # 요약
    result = {
        "summary": {
            "audio": os.path.abspath(args.audio),
            "sr": sr,
            "duration_sec": duration,
            "frame_hop_sec": float(args.hop/sr),
            "bpm_set": float(args.bpm),
            "beats_per_chord": int(args.beats_per_chord),
            "phase_offset_sec": rhythm.get("phase_offset_sec", 0.0),
            "chord_sequence": chords_norm
        },
        "per_chord": chord_list,
        "chord_overall_accuracy": chord_overall,
        "rhythm": {
            "beat_grid_bpm": rhythm["beat_grid_bpm"],
            "detected_bpm": rhythm["detected_bpm"],
            "tempo_stability_cv": rhythm["tempo_stability_cv"],
            "timing_error_ms": rhythm["timing_error_ms"],
            "drift_ms_per_min": rhythm["drift_ms_per_min"],
            "score_0_100": rhythm["score_0_100"]
        }
    }

    # 표준출력으로 JSON만!
    print(json.dumps(result, ensure_ascii=False))

if __name__ == "__main__":
    main()
