FROM python:3.11-slim

WORKDIR /app

# 安装 git（committer.py 需要）
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 只复制代码文件，data/ 目录通过 volume 挂载
COPY load_balancer.py reader_agent.py committer.py main_agent.py ./

# 配置 git（commit 时需要身份信息）
RUN git config --global user.email "agent@llm-papers.local" && \
    git config --global user.name "Paper Agent"

# data/papers.json 和仓库通过挂载传入，例如：
#   docker run --rm \
#     -v $(pwd):/app \
#     -v $(pwd)/data:/app/data:ro \
#     llm-ai-papers
CMD ["python3", "main_agent.py"]
