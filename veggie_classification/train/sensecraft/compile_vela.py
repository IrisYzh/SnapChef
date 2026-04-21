"""
Compile an int8-quantized TFLite model for Grove Vision AI V2 (Himax WE2 / Ethos-U55).

Output: <input_stem>_vela.tflite  — upload this file to SenseCraft.

Usage:
    python compile_vela.py ../classification_V3/veggie_mnv2_sensecraft.tflite
    python compile_vela.py ../classification_V3/veggie_mnv2_sensecraft.tflite --outdir ./out
"""

import argparse
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path


# Grove Vision AI V2 (Himax HX6538 / WE2) ships an Ethos-U55 with 64 MACs.
# SRAM/Flash sizes follow Seeed's reference Vela config for the board.
VELA_INI = textwrap.dedent("""\
    [System_Config.Grove_Vision_AI_V2]
    core_clock=400e6
    axi0_port=Sram
    axi1_port=OffChipFlash
    Sram_clock_scale=1.0
    Sram_burst_length=32
    Sram_read_latency=32
    Sram_write_latency=32
    OffChipFlash_clock_scale=0.125
    OffChipFlash_burst_length=128
    OffChipFlash_read_latency=64
    OffChipFlash_write_latency=64

    [Memory_Mode.Shared_Sram]
    const_mem_area=Axi1
    arena_mem_area=Axi0
    cache_mem_area=Axi0
""")


def ensure_vela() -> str:
    exe = shutil.which("vela")
    if exe:
        return exe
    print("[*] ethos-u-vela not found — installing...", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ethos-u-vela"])
    exe = shutil.which("vela")
    if not exe:
        sys.exit("vela still not on PATH after install; check your pip environment.")
    return exe


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", type=Path, help="Path to int8-quantized .tflite")
    ap.add_argument("--outdir", type=Path, default=Path("./vela_out"))
    ap.add_argument("--accelerator", default="ethos-u55-64",
                    choices=["ethos-u55-32", "ethos-u55-64", "ethos-u55-128", "ethos-u55-256"])
    args = ap.parse_args()

    if not args.model.is_file():
        sys.exit(f"model not found: {args.model}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    ini_path = (args.outdir / "vela.ini").resolve()
    ini_path.write_text(VELA_INI)

    vela = ensure_vela()
    cmd = [
        vela,
        "--accelerator-config", args.accelerator,
        "--optimise", "Performance",
        "--config", str(ini_path),
        "--system-config", "Grove_Vision_AI_V2",
        "--memory-mode", "Shared_Sram",
        "--output-dir", str(args.outdir),
        str(args.model),
    ]
    print("[*] running:", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)

    produced = args.outdir / f"{args.model.stem}_vela.tflite"
    if produced.is_file():
        print(f"\n[✓] done → {produced}")
        print("    upload this file to SenseCraft.")
    else:
        sys.exit(f"vela finished but {produced} is missing")


if __name__ == "__main__":
    main()
