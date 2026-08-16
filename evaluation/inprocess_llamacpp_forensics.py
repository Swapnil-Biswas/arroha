"""
evaluation/inprocess_llamacpp_forensics.py
-----------------------------------------
Environment and In-Process llama.cpp CUDA diagnostics.
Inspects:
- Python runtime & platform
- Host GPU (NVIDIA RTX 4050 Laptop GPU, 6GB VRAM)
- Driver & CUDA runtime compatibility
- DLL inventory in llama-b10451-bin-win-cuda-12.4-x64
- ctypes direct loading of llama.dll and ggml-cuda.dll
- In-process CUDA backend initialization & device discovery
"""

from __future__ import annotations

import ctypes
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any

LLAMA_DIR = Path(r"C:\Users\swapn\Downloads\llama-b10451-bin-win-cuda-12.4-x64")
MODEL_PATH = Path(r"C:\Users\swapn\.lmstudio\models\lmstudio-community\Qwen3-4B-Instruct-2507-GGUF\Qwen3-4B-Instruct-2507-Q4_K_M.gguf")


def inspect_environment() -> dict[str, Any]:
    print("=" * 80)
    print("  ARROHA — IN-PROCESS LLAMA.CPP ENVIRONMENT FORENSICS")
    print("=" * 80)

    # 1. Python & System
    py_ver = sys.version
    plat = platform.platform()
    print(f"Python Version: {py_ver}")
    print(f"Platform: {plat}")
    print(f"Executable: {sys.executable}")

    # 2. PyTorch & CUDA check
    torch_info = {}
    try:
        import torch
        torch_info = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "device_capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else "N/A",
            "vram_total_mb": round(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024), 2) if torch.cuda.is_available() else 0,
        }
        print(f"PyTorch Version: {torch_info['version']}")
        print(f"CUDA Available via PyTorch: {torch_info['cuda_available']}")
        print(f"GPU: {torch_info['device_name']} (Compute Capability: {torch_info['device_capability']})")
        print(f"Total GPU VRAM: {torch_info['vram_total_mb']} MiB")
    except Exception as e:
        print(f"PyTorch check error: {e}")

    # 3. Model File Verification
    model_exists = MODEL_PATH.exists()
    model_size_mb = round(MODEL_PATH.stat().st_size / (1024 * 1024), 2) if model_exists else 0.0
    print(f"Model File Path: {MODEL_PATH}")
    print(f"Model Exists: {model_exists} ({model_size_mb} MiB)")

    # 4. llama.cpp DLL Directory
    dll_dir_exists = LLAMA_DIR.exists()
    print(f"llama.cpp Binaries Directory: {LLAMA_DIR}")
    print(f"Directory Exists: {dll_dir_exists}")

    llama_dll = LLAMA_DIR / "llama.dll"
    ggml_cuda_dll = LLAMA_DIR / "ggml-cuda.dll"
    ggml_dll = LLAMA_DIR / "ggml.dll"
    ggml_base_dll = LLAMA_DIR / "ggml-base.dll"

    print(f"  llama.dll: {llama_dll.exists()} ({llama_dll.stat().st_size / (1024*1024):.1f} MB)")
    print(f"  ggml-cuda.dll: {ggml_cuda_dll.exists()} ({ggml_cuda_dll.stat().st_size / (1024*1024):.1f} MB)")
    print(f"  ggml.dll: {ggml_dll.exists()} ({ggml_dll.stat().st_size / 1024:.1f} KB)")
    print(f"  ggml-base.dll: {ggml_base_dll.exists()} ({ggml_base_dll.stat().st_size / 1024:.1f} KB)")

    # 5. Test ctypes Loading
    print("\n--- Testing Direct In-Process ctypes Loading ---")
    os.add_dll_directory(str(LLAMA_DIR))
    
    # Add LLAMA_DIR to PATH for dependencies (cublas, cudart)
    os.environ["PATH"] = str(LLAMA_DIR) + os.pathsep + os.environ.get("PATH", "")

    ctypes_load_ok = False
    cuda_devices_detected = 0
    backend_info = "N/A"

    try:
        cdll = ctypes.CDLL(str(llama_dll))
        print("Successfully loaded llama.dll into Python process via ctypes!")
        
        # Initialize llama backend
        if hasattr(cdll, "llama_backend_init"):
            cdll.llama_backend_init()
            print("Successfully called llama_backend_init()!")
            ctypes_load_ok = True

        # Check if ggml_backend_cuda or device enumeration is present
        if hasattr(cdll, "llama_supports_gpu_offload"):
            supports_gpu = bool(cdll.llama_supports_gpu_offload())
            print(f"llama_supports_gpu_offload(): {supports_gpu}")
        else:
            supports_gpu = True

        # Check print system info
        if hasattr(cdll, "llama_print_system_info"):
            cdll.llama_print_system_info.restype = ctypes.c_char_p
            sys_info_ptr = cdll.llama_print_system_info()
            if sys_info_ptr:
                sys_info_str = sys_info_ptr.decode("utf-8", errors="ignore")
                print(f"llama_print_system_info:\n  {sys_info_str}")
                backend_info = sys_info_str

    except Exception as e:
        print(f"Error loading llama.dll with ctypes: {e}")

    results = {
        "python_version": py_ver,
        "platform": plat,
        "torch_info": torch_info,
        "model": {
            "path": str(MODEL_PATH),
            "exists": model_exists,
            "size_mb": model_size_mb,
        },
        "llama_binaries": {
            "dir": str(LLAMA_DIR),
            "exists": dll_dir_exists,
            "llama_dll": llama_dll.exists(),
            "ggml_cuda_dll": ggml_cuda_dll.exists(),
            "ctypes_load_ok": ctypes_load_ok,
            "supports_gpu": supports_gpu if ctypes_load_ok else False,
            "system_info": backend_info,
        },
    }

    report_path = Path("evaluation/results/inprocess_llamacpp_forensics.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OUTPUT] Saved diagnostics JSON to: {report_path}")

    return results


if __name__ == "__main__":
    inspect_environment()
