from __future__ import annotations

import os
import subprocess
from pathlib import Path

from flask import Flask, jsonify, request

import platform

app = Flask(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]

if platform.system() == "Windows":
    DEFAULT_EXE = REPO_ROOT / "out" / "build" / "x64_release" / "bin" / "Release" / "LLMTest.exe"
else:
    DEFAULT_EXE = REPO_ROOT / "build" / "bin" / "LLMTest"

LLMTEST_EXE = Path(os.environ.get("LLMTEST_EXE", str(DEFAULT_EXE)))

def run_llmtest(prompt: str) -> str:
    if not LLMTEST_EXE.exists():
        raise FileNotFoundError(f"LLMTest executable not found: {LLMTEST_EXE}")

    completed = subprocess.run(
        [str(LLMTEST_EXE), prompt],
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    )

    response = completed.stdout.strip()
    if not response:
        return ""
    return response


@app.post("/generate")
def generate():
    payload = request.get_json(silent=True) or {}
    prompt = payload.get("prompt", "").strip()

    if not prompt:
        return jsonify({"error": "Missing 'prompt' in JSON body"}), 400

    try:
        response = run_llmtest(prompt)
        return jsonify(
            {
                "prompt": prompt,
                "response": response,
            }
        )
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 500
    except subprocess.TimeoutExpired:
        return jsonify({"error": "LLM generation timed out"}), 504
    except subprocess.CalledProcessError as exc:
        return jsonify(
            {
                "error": "LLMTest failed",
                "stderr": exc.stderr.strip() if exc.stderr else "",
            }
        ), 500
    

@app.get("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)