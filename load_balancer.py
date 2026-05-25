"""
负载均衡器：在多个 Ollama API 端点之间轮询分发请求。

策略：阻塞式轮询
  每个请求完成（成功或失败）后才切换到下一个端点。
  适合 120B 大模型——并发请求只会在模型侧排队，轮询反而更高效。

故障转移：
  某端点连续失败 FAIL_LIMIT 次后，打印警告并跳过该端点，
  直到它再次成功响应时自动恢复。
"""

import requests

# 两个 Ollama 端点，轮流使用
# 但实际上是自适应的
ENDPOINTS = [
    "http://localhost:11434/api/generate",
    "http://10.54.79.131:11434/api/generate",
]

# 某端点连续失败超过此次数后自动降级
FAIL_LIMIT = 3


class LoadBalancer:
    def __init__(self, endpoints: list[str], fail_limit: int = FAIL_LIMIT):
        self._endpoints = list(endpoints)
        self._fail_limit = fail_limit
        self._index = 0                           # 当前轮到哪个端点
        self._failures = [0] * len(endpoints)     # 各端点连续失败计数

    def post(self, **kwargs) -> requests.Response:
        """
        向当前端点发送 POST 请求。
        成功后切换到下一个端点；失败则尝试其他端点。
        所有端点均失败时抛出 RuntimeError。
        """
        n = len(self._endpoints)
        last_exc: Exception = RuntimeError("无可用端点")

        for _ in range(n):
            idx = self._index
            url = self._endpoints[idx]

            # 连续失败次数过多，跳过该端点
            if self._failures[idx] >= self._fail_limit:
                print(f"  [lb] 跳过故障端点 {url}", flush=True)
                self._index = (self._index + 1) % n
                continue

            try:
                resp = requests.post(url, **kwargs)
                resp.raise_for_status()
                # 成功：重置失败计数，切换到下一个端点备用
                self._failures[idx] = 0
                self._index = (self._index + 1) % n
                return resp
            except Exception as e:
                self._failures[idx] += 1
                if self._failures[idx] >= self._fail_limit:
                    print(f"  [lb] {url} 连续失败 {self._failures[idx]} 次，已降级", flush=True)
                self._index = (self._index + 1) % n
                last_exc = e

        raise RuntimeError(f"所有端点均失败：{last_exc}")


# 全局单例，供 reader_agent 直接导入使用
balancer = LoadBalancer(ENDPOINTS)
