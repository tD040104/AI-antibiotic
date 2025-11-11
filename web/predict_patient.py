import sys
import json
import base64
import os
from datetime import datetime
from typing import Any
import argparse

import numpy as np
import contextlib
import io
import re

# Thêm thư mục gốc của project vào Python path
# Script này nằm trong web/, cần tìm thư mục cha (chứa main.py)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # Lên một cấp từ web/ về thư mục gốc
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from main import MultiAgentOrchestrator


# nếu file có argparse, thêm option --b64 (nếu chưa có)
parser = argparse.ArgumentParser()
parser.add_argument('--b64', action='store_true', help='Read base64-encoded JSON from stdin')
# giữ các option hiện có (nếu script trước dùng khác, đảm bảo không lặp)
try:
    args, unknown = parser.parse_known_args()
except Exception:
    args = parser.parse_args()


def _json_default(obj: Any):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return str(obj)


def _quote_unquoted_json_keys(s: str) -> str:
    """
    Cố gắng thêm dấu nháy kép cho các key object chưa có nháy trong chuỗi gần-JSON.
    Ví dụ: {AMX/AMP: 1, FOX: 0} -> {"AMX/AMP": 1, "FOX": 0}
    Chỉ áp dụng heuristic, không đảm bảo cho mọi trường hợp.
    """
    # Regex bắt key không có nháy kép trước dấu ':' sau dấu '{' hoặc ','
    # Nhóm 1: prefix ({ hoặc , + khoảng trắng)
    # Nhóm 2: key (không chứa nháy kép và không gồm colon)
    # Nhóm 3: suffix (khoảng trắng) trước ':'
    pattern = re.compile(r'([{\[,]\s*)([^"\s][^:"]*?)(\s*):')

    def replacer(match: re.Match) -> str:
        prefix = match.group(1)
        key = match.group(2)
        suffix = match.group(3)
        # Giữ nguyên nếu key đã có nháy kép (phòng hờ) — regex đã lọc nhưng kiểm tra thêm
        if key.startswith('"') and key.endswith('"'):
            return f'{prefix}{key}{suffix}:'
        # Thêm nháy kép, đồng thời escape các nháy kép bên trong nếu có
        key_escaped = key.replace('"', r'\"')
        return f'{prefix}"{key_escaped}"{suffix}:'

    # Lặp nhiều lần phòng khi có nhiều cấp lồng nhau
    prev = s
    for _ in range(3):
        s = pattern.sub(replacer, s)
        if s == prev:
            break
        prev = s
    return s


def _insert_missing_commas(s: str) -> str:
    """
    Cố gắng chèn dấu phẩy bị thiếu giữa các cap key-value liên tiếp.
    Ví dụ: {"AMX/AMP": 1 "FOX": 0} -> {"AMX/AMP": 1, "FOX": 0}
    Heuristic theo từng cấp nông, không đảm bảo mọi trường hợp lồng sâu.
    """
    # 1) value đơn giản (số/true/false/null/chuỗi) trước một key tiếp theo
    pattern_simple = re.compile(
        r'("([^"\\]|\\.)*"\s*:\s*(?:-?\d+(?:\.\d+)?|true|false|null|"(?:[^"\\]|\\.)*"))\s+(?=")',
        flags=re.IGNORECASE,
    )
    # 2) value là object/array nông trước một key tiếp theo
    pattern_brackets = re.compile(
        r'("([^"\\]|\\.)*"\s*:\s*(?:\{[^{}]*\}|\[[^\[\]]*\]))\s+(?=")',
    )

    prev = s
    for _ in range(3):
        s = pattern_simple.sub(r'\1, ', s)
        s = pattern_brackets.sub(r'\1, ', s)
        if s == prev:
            break
        prev = s
    return s


def _write_log(s):
    try:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        log_dir = os.path.join(base_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, 'predict_input.log')
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"{datetime.utcnow().isoformat()}Z -- {s}\n")
    except Exception:
        pass  # không làm vỡ luồng chính nếu log lỗi


# Đọc stdin/argument theo flag
raw_stdin = ''
if args.b64:
    raw_stdin = sys.stdin.read()
else:
    # nếu script vẫn chấp nhận --input-json / --input-file, xử lý theo args tương ứng
    # fallback: đọc stdin (không base64)
    try:
        raw_stdin = sys.stdin.read()
    except Exception:
        raw_stdin = ''

_write_log(f"RAW_STDIN_REPR: {repr(raw_stdin)[:2000]}")

decoded_input = raw_stdin
if getattr(args, 'b64', False):
    try:
        decoded_bytes = base64.b64decode(raw_stdin)
        decoded_input = decoded_bytes.decode('utf-8', errors='replace')
        _write_log(f"DECODED_INPUT_REPR: {repr(decoded_input)[:2000]}")
    except Exception as e:
        sys.stderr.write(f"Không thể giải mã base64 từ stdin: {e}\n")
        _write_log(f"BASE64_DECODE_ERROR: {e}")
        sys.exit(1)

if decoded_input is None or decoded_input.strip() == "":
    sys.stderr.write("Không có input JSON từ PHP (sau giải mã nếu có).\n")
    _write_log("EMPTY_INPUT_AFTER_DECODING")
    sys.exit(1)

try:
    patient_data = json.loads(decoded_input)
except json.JSONDecodeError as e:
    # Ghi log chi tiết để debug: chuỗi nhận được và thông tin lỗi JSON
    sys.stderr.write(f"JSON không hợp lệ: {e}\n")
    _write_log(f"JSON_DECODE_ERROR: {e} -- DATA_REPR: {repr(decoded_input)[:2000]}")
    # In thêm base64 của dữ liệu đã giải mã (để tiện so sánh)
    try:
        b64_of_decoded = base64.b64encode(decoded_input.encode('utf-8', errors='replace')).decode('ascii')
        _write_log(f"BASE64_OF_DECODED: {b64_of_decoded[:2000]}")
    except Exception:
        pass
    sys.exit(1)

try:
    # Chặn mọi stdout phát sinh từ quá trình load để không làm bẩn JSON output
    with contextlib.redirect_stdout(io.StringIO()):
        orchestrator = MultiAgentOrchestrator.load_from_state(args.state_path)
except Exception as exc:
    print(json.dumps({"status": "error", "message": f"Không thể tải mô hình: {exc}"}))
    sys.exit(2)

try:
    # Chặn mọi stdout phát sinh từ quá trình predict để không làm bẩn JSON output
    with contextlib.redirect_stdout(io.StringIO()):
        prediction_result = orchestrator.predict(patient_data)
    print(json.dumps({"status": "ok", "data": prediction_result}, default=_json_default))
except Exception as exc:
    print(json.dumps({"status": "error", "message": f"Lỗi dự đoán: {exc}"}))
    sys.exit(3)

# Debug / placeholder result (thay bằng gọi orchestrator.predict(...) trong thực tế)
try:
    result = {
        "status": "ok",
        "message": "Debug response from predict_patient.py",
        "patient_received": patient_data
    }
    out = json.dumps(result, ensure_ascii=False)
    # In ra stdout (PHP sẽ đọc)
    sys.stdout.write(out)
    sys.stdout.flush()
    # exit 0
    sys.exit(0)
except Exception as e:
    sys.stderr.write(f"ERROR_CREATING_OUTPUT: {e}\n")
    _write_log(f"ERROR_CREATING_OUTPUT: {e}")
    sys.exit(1)


