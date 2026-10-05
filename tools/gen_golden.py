"""生成 TFLite golden 数据并与生成的 C 推理可执行文件对拍。

用法（tiny_ml 环境，需要 tensorflow）：
    python tools/gen_golden.py model/mnist_cnn.tflite --dir model/mnist_cnn_c
流程：
    1. 固定种子生成均匀随机输入 -> <dir>/input.bin（float32 C 序裸数据）
    2. tf.lite.Interpreter 前向 -> <dir>/golden.npy
    3. 若 <dir>/output.bin 已存在（先运行 forward.exe），做 allclose 对拍
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="TFLite golden 生成 + C 输出对拍")
    parser.add_argument("model", nargs="?", default="model/mnist_cnn.tflite")
    parser.add_argument("--dir", default=None, help="产物目录（默认 <model_stem>_c/）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    args = parser.parse_args()

    model_path = Path(args.model)
    out_dir = Path(args.dir) if args.dir else model_path.parent / f"{model_path.stem}_c"

    import tensorflow as tf

    interp = tf.lite.Interpreter(model_path=str(model_path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    in_shape = tuple(inp["shape"])
    print(f"输入 {in_shape} {inp['dtype']}，输出 {tuple(out['shape'])} {out['dtype']}")

    # 1. 固定种子随机输入（复现实验）
    rng = np.random.default_rng(args.seed)
    x = rng.uniform(-1.0, 1.0, size=in_shape).astype(np.float32)
    in_bin = out_dir / "input.bin"
    in_bin.write_bytes(x.tobytes())
    print(f"input.bin  -> {in_bin}")

    # 2. TFLite golden
    interp.set_tensor(inp["index"], x)
    interp.invoke()
    golden = interp.get_tensor(out["index"])
    if out["dtype"] != np.float32:          # 量化模型输出反量化到 float
        scale, zero = out["quantization"]
        golden = (golden.astype(np.float32) - zero) * scale
    golden_path = out_dir / "golden.npy"
    np.save(golden_path, golden)
    print(f"golden.npy -> {golden_path}  shape={golden.shape}")

    # 3. 与 C 可执行文件输出对拍
    out_bin = out_dir / "output.bin"
    if not out_bin.exists():
        print(f"[提示] 未找到 {out_bin}，先运行 forward.exe 后重跑本脚本完成对拍")
        return 0
    c_out = np.frombuffer(out_bin.read_bytes(), dtype=np.float32)
    if c_out.size != golden.size:
        print(f"[错误] 输出元素数不一致: C={c_out.size} golden={golden.size}")
        return 1
    ok = np.allclose(c_out, golden.ravel(), rtol=args.rtol, atol=args.atol)
    diff = np.abs(c_out - golden.ravel())
    print(f"\n对拍结果: {'PASS' if ok else 'FAIL'}  "
          f"max|diff|={diff.max():.3e}  mean|diff|={diff.mean():.3e}  "
          f"(rtol={args.rtol}, atol={args.atol})")
    if not ok:
        bad = np.argsort(diff)[-5:][::-1]
        for i in bad:
            print(f"  out[{i}] C={c_out[i]:.6f} golden={golden.ravel()[i]:.6f}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
