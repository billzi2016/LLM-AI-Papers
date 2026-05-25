# LLM-AI-Papers

自动化 AI 论文精读 Agent —— 实时追踪 arXiv 前沿研究，生成深入浅出的中文解读，按发表日期自动提交到 GitHub，构建个人科研知识库。

## 项目结构

```
LLM-AI-Papers/
├── data/                        # 数据目录（gitignore，含版权内容）
│   ├── raw.txt                  # 原始论文笔记
│   ├── papers.json              # 提取 + 匹配后的结构化数据
│   ├── main.py                  # 数据处理流水线入口
│   ├── process_raw.py           # Ollama 提取 title/summary
│   └── check_arxiv.py           # arXiv 快照匹配
├── load_balancer.py             # 双 Ollama 端点轮询负载均衡
├── reader_agent.py              # 论文解读生成（Prompt + 调用）
├── committer.py                 # Markdown 生成 + git commit
├── main_agent.py                # Agent 主入口
├── requirements.txt
├── pyproject.toml
└── Dockerfile
```

生成的解读文章按月份存放：
```
2024-01/
  attention-is-all-you-need.md
2024-03/
  ...
```

## 两条流水线

### 流水线一：数据处理（`data/`）

从原始笔记提取论文，匹配 arXiv 获取规范标题、摘要和日期。

```bash
cd data
python3 main.py
```

输出：`data/papers.json`，字段：`title / arxiv_id / summary / abstract / date`

### 流水线二：解读生成 Agent

读取 `papers.json`，为每篇论文调用大模型生成中文解读，按发表日期 commit。

```bash
python3 main_agent.py
```

完成后手动 push：

```bash
git push
```

## 环境要求

- Python 3.11+
- Ollama 运行 `gpt-oss:120b`（本地或远程）
- git

```bash
pip install -r requirements.txt
```

## Ollama 配置

默认使用两个端点轮询，在 `load_balancer.py` 中修改：

```python
ENDPOINTS = [
    "http://localhost:11434/api/generate",
    "http://10.54.79.131:11434/api/generate",
]
```

某端点不可用时自动降级到另一个。

## Docker 运行

```bash
docker build -t llm-ai-papers .
docker run --rm -v $(pwd):/app llm-ai-papers
```

## 断点续跑

Agent 每完成一篇立即保存进度到 `.reader_progress.json`。  
Ctrl+C 中断后重新运行，自动跳过已完成的论文。

失败的论文记录在 `.reader_failed.json`，可单独处理后重跑。
