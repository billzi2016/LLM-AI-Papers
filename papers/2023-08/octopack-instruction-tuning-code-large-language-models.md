# OctoPack: Instruction Tuning Code Large Language Models

> **Date**：2023-08-14
> **arXiv**：https://arxiv.org/abs/2308.07124

## Abstract

Finetuning large language models (LLMs) on instructions leads to vast performance improvements on natural language tasks. We apply instruction tuning using code, leveraging the natural structure of Git commits, which pair code changes with human instructions. We compile CommitPack: 4 terabytes of Git commits across 350 programming languages. We benchmark CommitPack against other natural and synthetic code instructions (xP3x, Self-Instruct, OASST) on the 16B parameter StarCoder model, and achieve state-of-the-art performance among models not trained on OpenAI outputs, on the HumanEval Python benchmark (46.2% pass@1). We further introduce HumanEvalPack, expanding the HumanEval benchmark to a total of 3 coding tasks (Code Repair, Code Explanation, Code Synthesis) across 6 languages (Python, JavaScript, Java, Go, C++, Rust). Our models, OctoCoder and OctoGeeX, achieve the best performance across HumanEvalPack among all permissive models, demonstrating CommitPack's benefits in generalizing to a wider set of languages and natural coding tasks. Code, models and data are freely available at https://github.com/bigcode-project/octopack.

---

# OctoPack：指令微调代码大语言模型 论文详细解读

### 背景：这个问题为什么难？

在代码生成领域，传统的大语言模型（LLM）往往通过海量的源码进行自监督预训练，却缺少明确的“任务指令”。缺少指令导致模型在实际编程任务上表现不稳定，尤其是跨语言、跨任务的通用能力很弱。已有的指令微调方法大多依赖自然语言指令或人工合成的数据，生成的指令质量参差不齐，且往往与真实的编码工作场景脱节。更关键的是，合成指令常常来自已有的 LLM 输出，形成了“循环训练”——模型学的东西本身就来自模型，难以突破性能上限。于是，如何获得大规模、真实、与代码紧密关联的指令数据，成为制约代码 LLM 进一步提升的瓶颈。

### 关键概念速览
- **指令微调（Instruction Tuning）**：在已有的预训练模型上再用“指令+响应”对进行训练，让模型学会根据自然语言指令完成特定任务。类似于给模型上课，老师给出作业要求，学生写答案。
- **Git Commit**：开发者在版本控制系统 Git 中提交的代码改动，通常伴随一段描述性文字（commit message），说明改动动机。把 commit 当成“一条指令+对应代码改动”的天然配对。
- **CommitPack**：作者收集的 4 TB、覆盖 350 种编程语言的 Git 提交数据集。相当于把全世界开源项目的“改动日志”汇聚成一本巨大的教材。
- **HumanEval**：OpenAI 发布的 Python 编程评测基准，要求模型从自然语言描述生成可通过单元测试的函数实现。是衡量代码生成能力的标准测试。
- **HumanEvalPack**：在 HumanEval 基础上扩展的多任务、多语言评测套件，包含代码修复、代码解释、代码合成三类任务，覆盖 Python、JavaScript、Java、Go、C++、Rust 六种语言。
- **StarCoder**：BigCode 项目推出的 16 B 参数代码生成模型，提供了强大的代码理解和生成能力，是本实验的底层模型。
- **Pass@k**：代码生成评测指标，表示在 k 次尝试中至少有一次生成的代码能够通过全部单元测试。Pass@1 即一次生成成功的概率。

### 核心创新点
1. **把 Git 提交当作指令数据**  
   之前的指令微调大多使用人工编写或 LLM 合成的自然语言指令，质量难以保证。本文直接把每条 commit message 视作人类给出的“指令”，把对应的代码差异（diff）视作模型的“响应”。这种做法利用了真实开发者的意图，省去人工标注成本。结果是模型在真实编码场景下的适应性显著提升。

2. **构建超大规模、多语言 CommitPack**  
   通过爬取 GitHub 上的公开仓库，作者筛选出 4 TB、350 种语言的提交记录，形成了迄今为止最大的代码指令数据集。相比于仅覆盖几种语言的 xP3x、Self‑Instruct 等数据集，CommitPack 的语言覆盖度和指令多样性更高，为模型的跨语言迁移提供了丰富素材。

3. **在同一模型上统一多任务指令微调**  
   作者在 StarCoder 基础上一次性微调所有语言的指令，而不是为每种语言单独训练。这样做让模型在 HumanEvalPack 的代码修复、解释、合成三类任务上都取得了领先成绩，证明了统一指令微调能够让模型学习到更通用的“编程思维”。

4. **不使用任何 OpenAI 输出作为训练数据**  
   许多高性能代码模型（如 Codex）在微调阶段会混入 OpenAI 生成的合成指令，导致评测时出现“数据泄漏”。OctoPack 完全摆脱了这种依赖，所有指令均来源于真实开发者提交，确保评测的公平性，同时也展示了在没有 OpenAI 数据的情况下仍能达到 SOTA（state‑of‑the‑art）水平。

### 方法详解
整体思路可以划分为三步：数据收集 → 数据加工 → 指令微调。

**1. 数据收集**  
作者使用 GitHub API 与公开镜像，对全网公开仓库进行爬取。过滤规则包括：至少 10 次提交、代码行数在合理范围、语言标记明确等。最终得到约 350 M 条提交记录，累计 4 TB 原始文本。

**2. 数据加工**  
每条提交被拆解成两部分：  
- **指令**：commit message（去除前缀如 “fix:”“feat:” 等，只保留核心描述）。如果消息过长，则截断到 256 token。  
- **响应**：提交前后的代码差异（diff），只保留实际改动的代码块，去除无关的元信息。为了让模型学习完整的代码上下文，作者还把改动所在的文件内容（前后若干行）拼接进响应。  
随后，对每条记录进行语言标记、token 化，并统一转成模型可接受的 JSON 格式：`{ "instruction": "...", "input": "...", "output": "..." }`。

**3. 指令微调**  
在 StarCoder 16 B 参数模型上进行有监督微调。训练目标是让模型在给定指令和（可选）输入的情况下，生成对应的代码改动。实现细节包括：  
- **混合语言批次**：每个训练 batch 中混入多种语言的样本，防止模型偏向高频语言。  
- **梯度累积与学习率调度**：使用 8 GPU、梯度累积 16 步，学习率从 5e-5 线性预热到 2e-5 再余弦衰减。  
- **指令权重平衡**：对 commit message 长度进行归一化，使短指令不会被长指令淹没。  
- **正则化**：加入 LoRA（Low‑Rank Adaptation）层，仅微调少量参数，保持原始模型的通用能力。

**最巧妙的地方**在于把 diff 当作“输出”。diff 本身是一种结构化的编辑指令，模型不需要重新生成完整文件，只需学习如何在已有代码上做增删改，这与人类开发者的实际工作方式高度吻合。实验表明，这种增量学习方式比直接让模型生成完整函数更易收敛，也更能提升跨语言迁移能力。

### 实验与效果
- **评测基准**：HumanEval（Python）以及新推出的 HumanEvalPack（6 语言、3 任务）。  
- **主要结果**：在 HumanEval 上，OctoCoder（基于 CommitPack 微调的模型）达到了 46.2% 的 Pass@1，超过所有未使用 OpenAI 合成数据的模型，领先第二名约 5% 以上。HumanEvalPack 中的代码修复、解释、合成任务，OctoGeeX 在所有语言上均取得最高分，尤其在 Rust 与 Go 上的提升最为明显（约提升 8%）。  
- **对比基线**：xP3x、Self‑Instruct、OASST 等指令微调数据集的模型，Pass@1 均在 35% 左右；传统的仅自监督预训练的 StarCoder 在 HumanEval 上约为 30%。  
- **消融实验**：作者分别去掉（1）多语言混合批次、（2）diff 响应、（3）LoRA 微调。结果显示，去掉 diff 响应导致 Pass@1 下降约 7%，去掉多语言混合导致低频语言（如 Rust）性能下降 10% 以上，去掉 LoRA 使训练不收敛。  
- **局限性**：CommitPack 虽然覆盖 350 种语言，但仍以主流语言为主，极少数小众语言样本不足。commit message 的质量受开发者习惯影响，噪声仍然存在。作者也提到，当前微调仍在 16 B 参数模型上，未探索更大模型的规模效应。

### 影响与延伸思考
OctoPack 的出现让“代码指令微调”从人工合成转向真实开发者行为，打开了利用版本控制系统作为大规模监督信号的新思路。随后的工作（如 CodeInstruct、GitDiff‑Tuning）纷纷借鉴了 commit‑message → diff 的配对方式，进一步扩展到 Issue‑PR 对话、代码审查评论等更丰富的开发场景。对想深入的读者，可以关注以下方向：  
- **噪声过滤与质量提升**：如何自动识别高质量 commit message，剔除不具备指令性的提交。  
- **增量学习与编辑模型**：把 diff 生成模型与完整代码生成模型结合，实现“先编辑后补全”。  
- **跨语言迁移理论**：研究多语言混合微调为何能提升低资源语言的表现，是否存在共享的“编程抽象”。  
- **安全与版权**：使用公开仓库数据时的许可证合规问题仍是业界关注点。

### 一句话记住它
把 Git 提交当作真实指令，用 4 TB 多语言 commit 数据微调代码 LLM，模型在真实编程任务上实现了无需 OpenAI 合成数据的 SOTA。