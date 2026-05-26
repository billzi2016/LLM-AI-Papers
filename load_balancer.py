"""
端点配置：列出所有可用的 Ollama API 端点。

main_agent 会为每个端点启动一个独立线程，真正并行处理。
注释掉一行即减少一个端点，添加一行即增加，代码自适应。
"""

ENDPOINTS = [
    # "http://localhost:11434/api/generate",
    "http://10.54.79.131:11434/api/generate",
]
