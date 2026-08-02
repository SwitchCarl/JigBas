"""
环境验证脚本：
1. 检查所有依赖包是否可正常 import
2. 验证 Wespeaker 声纹嵌入提取（256维）
3. 验证 FunASR Paraformer-large ASR 转写

用法：
  python verify_env.py                        # 使用默认 test_audio.wav
  python verify_env.py D:\audio\test.wav      # 指定音频文件
"""

import os
import sys
import time
import warnings
from contextlib import redirect_stdout
import numpy as np

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "Models")
FUNASR_MODEL_DIR = os.path.join(MODELS_DIR, "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch")


def _generate_test_audio(output_path, duration=3.0, sample_rate=16000):
    """生成 440Hz 正弦波测试音频"""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    import soundfile as sf
    sf.write(output_path, signal, sample_rate)
    print(f"  测试音频已生成: {output_path} ({duration}s, {sample_rate}Hz)")
    return output_path


def check_imports():
    """检查所有依赖包"""
    print("=" * 50)
    print("[1/3] 依赖包检查")
    packages = [
        ("torch",      "PyTorch"),
        ("torchaudio", "TorchAudio"),
        ("funasr",     "FunASR"),
        ("wespeaker",  "Wespeaker"),
        ("webrtcvad",  "WebRTC VAD"),
        ("soundfile",  "SoundFile"),
        ("librosa",    "Librosa"),
        ("scipy",      "SciPy"),
        ("pandas",     "Pandas"),
        ("tqdm",       "tqdm"),
    ]

    all_ok = True
    for pkg, name in packages:
        try:
            __import__(pkg)
            print(f"  [OK] {name} ({pkg})")
        except ImportError as e:
            print(f"  [FAIL] {name} ({pkg}): {e}")
            all_ok = False

    if all_ok:
        print("  所有依赖包检查通过!")
    else:
        print("  部分依赖包缺失，请运行: pip install -r requirements.txt")

    return all_ok


def test_wespeaker(audio_path):
    """验证 Wespeaker 声纹嵌入提取"""
    print("=" * 50)
    print("[2/3] 声纹模型验证 (ECAPA-TDNN)")
    try:
        import wespeaker

        with open(os.devnull, "w") as f, redirect_stdout(f):
            model = wespeaker.load_model("chinese")
        print(f"  设备: {model.device}")

        print("  提取声纹嵌入...")
        start = time.time()
        embedding = model.extract_embedding(audio_path)
        elapsed = time.time() - start

        emb_np = embedding.cpu().numpy() if hasattr(embedding, 'cpu') else np.array(embedding)
        print(f"  嵌入维度: {emb_np.shape}")
        print(f"  提取耗时: {elapsed:.3f}s")
        print(f"  嵌入范数: {np.linalg.norm(emb_np):.4f}")
        print(f"  [OK] 声纹模型验证通过")
        return True
    except Exception as e:
        print(f"  [FAIL] 声纹模型验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_funasr(audio_path):
    """验证 FunASR Paraformer-large ASR 转写"""
    print("=" * 50)
    print("[3/3] FunASR ASR 模型验证")
    try:
        from funasr import AutoModel

        print("  加载模型...")
        if os.path.isdir(FUNASR_MODEL_DIR) and os.listdir(FUNASR_MODEL_DIR):
            print(f"  使用本地模型: {FUNASR_MODEL_DIR}")
            model = AutoModel(model=FUNASR_MODEL_DIR)
        else:
            print("  本地模型不存在，自动下载...")
            model = AutoModel(
                model="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
            )
        print("  模型加载成功")

        print("  执行转写...")
        start = time.time()
        result = model.generate(input=audio_path)
        elapsed = time.time() - start

        print(f"  转写耗时: {elapsed:.3f}s")
        if result and len(result) > 0:
            text = result[0].get("text", "") if isinstance(result[0], dict) else str(result[0])
            print(f"  转写结果: '{text}'")
        else:
            print("  转写结果: (空)")

        print("  [OK] FunASR 转写功能正常")
        return True
    except Exception as e:
        print(f"  [FAIL] FunASR 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_verification(audio_path=None):
    """执行完整验证流程，返回 (wespeaker_ok, funasr_ok)"""
    print(f"Python: {sys.version}")
    print(f"模型目录: {MODELS_DIR}")
    print()

    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. 依赖检查
    if not check_imports():
        print("\n依赖检查未通过，请先安装缺失的包。")
        return False, False
    print()

    # 2. 确定测试音频
    if audio_path is None:
        audio_path = os.path.join(PROJECT_ROOT, "test_audio.wav")
    if not os.path.exists(audio_path):
        _generate_test_audio(audio_path)
    else:
        print(f"  使用测试音频: {audio_path}")
    print()

    # 3. 模型验证
    wespeaker_ok = test_wespeaker(audio_path)
    print()
    funasr_ok = test_funasr(audio_path)
    print()

    # 汇总
    print("=" * 50)
    print("验证结果汇总:")
    print(f"  Wespeaker: {'OK' if wespeaker_ok else 'FAILED'}")
    print(f"  FunASR:    {'OK' if funasr_ok else 'FAILED'}")

    if wespeaker_ok and funasr_ok:
        print("\n环境验证通过! 可以开始后续开发。")
    else:
        print("\n部分验证未通过，请检查上述错误信息。")

    return wespeaker_ok, funasr_ok


def main():
    print("环境验证开始...")
    print()
    audio_path = sys.argv[1] if len(sys.argv) > 1 else None
    wespeaker_ok, funasr_ok = run_verification(audio_path)
    if not (wespeaker_ok and funasr_ok):
        sys.exit(1)


if __name__ == "__main__":
    main()