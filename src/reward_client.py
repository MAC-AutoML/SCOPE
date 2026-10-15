"""HTTP client for the reward service. One group shares one diffusion seed."""
import math
import threading
import time
import uuid
from typing import List

import requests
from service_auth import auth_headers


class RewardClient:
    def __init__(self, endpoints: List[str], timeout: int = 1800, max_retries: int = 2):
        if not endpoints:
            raise ValueError("At least one reward endpoint is required")
        self.endpoints = endpoints
        self.timeout = timeout
        self.max_retries = max_retries
        self._lock = threading.Lock()
        self._ptr = 0
        self._sessions = []
        for _endpoint in endpoints:
            session = requests.Session()
            session.trust_env = False
            adapter = requests.adapters.HTTPAdapter(pool_connections=8, pool_maxsize=8, max_retries=0)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            self._sessions.append(session)

    def _next(self):
        with self._lock:
            index = self._ptr
            self._ptr = (self._ptr + 1) % len(self.endpoints)
            return self.endpoints[index], self._sessions[index]

    def score(self, prompts, ori_prompts, metadata_list, seed):
        if not prompts:
            return {"scores": []}
        url, session = self._next()
        body = {
            "prompts": prompts,
            "ori_prompts": ori_prompts,
            "metadata_list": metadata_list,
            "seed": seed,
            "save_batch_id": uuid.uuid4().hex[:16],
        }
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                response = session.post(url, json=body, timeout=self.timeout, headers=auth_headers())
                response.raise_for_status()
                data = response.json()
                scores = data.get("scores")
                if not isinstance(scores, list) or len(scores) != len(prompts):
                    raise ValueError(f"Invalid scores: {data}")
                if not all(math.isfinite(float(score)) for score in scores):
                    raise ValueError("Non-finite reward score")
                return {"scores": [float(score) for score in scores]}
            except Exception as error:
                last_error = error
                print(f"[RewardClient] Attempt {attempt + 1}/{self.max_retries + 1} failed: {error!r}")
                if attempt < self.max_retries:
                    time.sleep(0.2 * (attempt + 1))
                else:
                    raise RuntimeError(f"RewardClient failed after retries: {last_error!r}") from error
        raise RuntimeError(f"RewardClient failed: {last_error!r}")
