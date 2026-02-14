# src/utils/logging.py
import csv, os, time
from typing import Dict, Any

class EpisodeLogger:
    def __init__(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(out_dir, f"episode_{stamp}.csv")
        self._file = open(self.path, "w", newline="")
        self._writer = None

    def log(self, row: Dict[str, Any]):
        if self._writer is None:
            self._writer = csv.DictWriter(self._file, fieldnames=list(row.keys()))
            self._writer.writeheader()
        self._writer.writerow(row)

    def close(self):
        self._file.close()
