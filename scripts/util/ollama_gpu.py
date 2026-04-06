from __future__ import annotations

import os
import re
import subprocess


def _describe_ollama_gpu_state() -> str:
    """Query actual Ollama runtime state from 'ollama ps' output."""
    try:
        result = subprocess.run(
            ["ollama", "ps"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0 and result.stdout:
            output = result.stdout.lower()
            # Look for GPU percentage in format: "43%/57% cpu/gpu"
            gpu_match = re.search(r"\d+%/(\d+)%\s+cpu/gpu", output)
            if gpu_match:
                gpu_pct = int(gpu_match.group(1))
                if gpu_pct > 0:
                    return "enabled (GPU)"
                else:
                    return "disabled (CPU only)"
            # Check for 100% CPU only
            elif re.search(r"100%\s+cpu", output):
                return "disabled (CPU only)"
    except Exception:
        pass
    
    return "unknown"
