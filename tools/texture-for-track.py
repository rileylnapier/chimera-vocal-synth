"""Textures that fit a track: read its key, tuning and tempo, then render `chimera texture` for each kind.

    python tools/texture-for-track.py song.wav [pad choir drone chops birds] [--outdir out/textures/<song>]
                                      [any chimera texture flag: --bars 8 --chords "i VI III VII" --seed 2 ...]

Key, tuning and tempo come from flow-analysis (FLOW_ANALYSIS, default ~/roam/apps/flow-analysis); run this with
that project's Python (its .venv has librosa). The tuning is passed as --tune, so the voices match a track that
isn't at A440. Pass --key or --bpm to override what was detected.
"""
import os
import subprocess
import sys
from pathlib import Path

KINDS = {"pad", "choir", "drone", "chops", "birds"}
REPO = Path(__file__).resolve().parent.parent


def detect(path: str) -> dict:
    flow = Path(os.environ.get("FLOW_ANALYSIS", "~/roam/apps/flow-analysis")).expanduser() / "src"
    sys.path.insert(0, str(flow))
    try:
        from flow_analysis.audio import load
        from flow_analysis.features import tempo_and_beats
        from flow_analysis.key_detect import detect_key
    except ImportError as e:
        sys.exit(f"needs flow-analysis at {flow} and its Python ({e}); or pass --key and --bpm yourself")
    audio, sr = load(path)  # (n, 2)
    mono = audio.mean(axis=1)
    key = detect_key(mono, sr)
    bpm, _ = tempo_and_beats(mono, sr)
    return {"key": key["name"], "camelot": key["camelot"], "confidence": key["confidence"],
            "tune": round(100 * key["tuning_semitones"], 1), "bpm": round(bpm) if abs(bpm - round(bpm)) < 0.15 else bpm}


def main(argv: list[str]) -> None:
    if not argv or argv[0].startswith("-"):
        sys.exit(__doc__)
    track, rest = argv[0], argv[1:]
    kinds = [a for a in rest if a in KINDS] or ["pad", "choir", "drone", "chops", "birds"]
    flags = [a for a in rest if a not in KINDS]
    outdir = f"out/textures/{Path(track).stem}"
    if "--outdir" in flags:
        i = flags.index("--outdir")
        outdir = flags[i + 1]
        del flags[i:i + 2]
    found = detect(track)
    print(f"{Path(track).name}: {found['key']} ({found['camelot']}, {100 * found['confidence']:.0f}% sure), "
          f"tuning {found['tune']:+.1f} cents, {found['bpm']} bpm", flush=True)
    auto = []
    if "--key" not in flags:
        auto += ["--key", found["key"]]
    if "--tune" not in flags:
        auto += ["--tune", str(found["tune"])]
    if "--bpm" not in flags:
        auto += ["--bpm", str(found["bpm"])]
    exe = REPO / "target/release/chimera"
    if not exe.exists():
        subprocess.run(["cargo", "build", "--release", "-q", "-p", "chimera-cli"], cwd=REPO, check=True)
    for kind in kinds:
        subprocess.run([str(exe), "texture", kind, outdir, *auto, *flags], cwd=REPO, check=True)


if __name__ == "__main__":
    main(sys.argv[1:])
