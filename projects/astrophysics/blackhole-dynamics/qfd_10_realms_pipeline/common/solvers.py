"""Lightweight solvers + reporting for the QFD realms."""
from dataclasses import dataclass, asdict
from typing import Dict, Any
import json, os, math, time
from datetime import datetime

@dataclass
class RealmResult:
    name: str
    status: str
    fixed_params: Dict[str, float]
    notes: str

def write_report(result: RealmResult, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    path = os.path.join(out_dir, f"{ts}_{result.name}.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(asdict(result), f, indent=2)
    return path
