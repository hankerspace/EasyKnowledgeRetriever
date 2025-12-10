import csv
import logging
import time
import os
import asyncio
from .logger import logger

class TokenTracker:
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.lock = asyncio.Lock()
        self.pending_updates = []
        self._ensure_log_file()
        self._start_writer()

    def _ensure_log_file(self):
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, "token_usage.csv")
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timestamp",
                        "model",
                        "binding",
                        "input_tokens",
                        "output_tokens",
                        "total_tokens",
                        "cost",
                    ]
                )

    def _start_writer(self):
        asyncio.create_task(self._periodic_flush())

    async def _periodic_flush(self):
        while True:
            await asyncio.sleep(5)  # Flush every 5 seconds
            async with self.lock:
                if self.pending_updates:
                    try:
                        with open(self.log_file, "a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerows(self.pending_updates)
                        self.pending_updates = []
                    except Exception as e:
                        logger.error(f"Failed to write token logs: {e}")

    async def track(
        self,
        model: str,
        binding: str,
        input_tokens: int,
        output_tokens: int,
        cost: float = 0.0,
    ):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        total_tokens = input_tokens + output_tokens
        async with self.lock:
            self.pending_updates.append(
                [
                    timestamp,
                    model,
                    binding,
                    input_tokens,
                    output_tokens,
                    total_tokens,
                    cost,
                ]
            )
