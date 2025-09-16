import json, subprocess, sys, os

def test_practice_eval_runs():
    script = "src/analysis/practice_eval.py"
    if not os.path.exists(script):
        raise RuntimeError("practice_eval.py not found; place the file first.")

    # 샘플 파일이 없다면 테스트 스킵(레포 초기상태 고려)
    sample = "data/raw/take1.mp3"
    if not os.path.exists(sample):
        return

    out = subprocess.check_output([
        sys.executable, script,
        "--audio", sample, "--bpm", "92",
        "--chords", "Em,C,G,D", "--beats-per-chord", "8",
        "--thr", "0.05", "--smooth", "11"
    ])
    obj = json.loads(out.decode("utf-8"))
    assert "per_chord" in obj and "rhythm" in obj
