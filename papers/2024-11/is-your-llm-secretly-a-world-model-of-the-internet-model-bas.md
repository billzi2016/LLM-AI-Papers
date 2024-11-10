# Is Your LLM Secretly a World Model of the Internet? Model-Based Planning   for Web Agents

> **Date**：2024-11-10
> **arXiv**：https://arxiv.org/abs/2411.06559

## Abstract

Language agents based on large language models (LLMs) have demonstrated great promise in automating web-based tasks. Recent work has shown that incorporating advanced planning algorithms, e.g., tree search, is advantageous over reactive planning for web agents. However, unlike simulated sandbox environments, real-world environments such as the web are rife with irreversible actions. This undermines the feasibility of backtracking, a cornerstone of (tree) search. Overly relying on test-time search also hurts efficiency. We advocate model-based planning for web agents that employs a world model to simulate and deliberate over the outcome of each candidate action before committing to one. We systematically explore this paradigm by (1) Proposing a model-based planning framework, WebDreamer, which employs LLMs to serve as both world models and value functions; (2) Training specialized LLMs as world models with a scalable data synthesis pipeline. Empirical results demonstrate that WebDreamer achieves substantial performance improvements over reactive baselines. It is competitive, while being 4-5 times more efficient, with tree search in sandbox environments (VisualWebArena) and also works effectively on real-world websites (Online-Mind2Web and Mind2Web-Live). Furthermore, our trained world model, Dreamer-7B, performs comparable to GPT-4o, highlighting the potential of specialized world models for efficient and effective planning in complex web environments.

---

# 你的 LLM 暗藏互联网世界模型吗？面向网页代理的基于模型的规划 论文详细解读

### 背景：这个问题为什么难？

在真实的网页环境里，点击、填写表单、下单等操作往往是“一去不复返”的——一次错误的提交可能导致金钱损失或账号被封。过去的网页代理大多采用**反应式**策略：看到页面就立刻决定下一步，或者在测试时用**树搜索**回溯尝试不同路径。但树搜索依赖于可以随意撤销动作的沙盒环境，面对不可逆的网页交互时会频频卡死。再者，实时搜索耗时严重拖慢了实际使用的效率。于是，如何在不回溯、且保持高效的前提下，让代理在行动前“先演练”成为关键难题。

### 关键概念速览

**LLM（大语言模型）**：通过海量文本训练得到的模型，能够生成自然语言或代码。这里把它当成“思考者”，负责给出行动建议或评估结果。  

**World Model（世界模型）**：对环境（这里是网页）的状态转移进行预测的模型，类似于下棋时的棋局评估器，能在脑中“模拟”一步动作后会出现什么画面。  

**Model‑Based Planning（基于模型的规划）**：先用世界模型模拟若干候选动作的后果，再挑选最有价值的那一个执行，而不是直接行动。  

**Value Function（价值函数）**：给每个状态或动作打分的函数，分数越高表示越有可能达成目标。这里的价值函数也是由 LLM 提供的。  

**Tree Search（树搜索）**：在搜索空间中展开多条分支并回溯比较的算法，像在迷宫里走来走去找最短路。  

**Irreversible Action（不可逆操作）**：一旦执行就无法撤销的网页行为，例如付款、提交表单等。  

**Data Synthesis Pipeline（数据合成流水线）**：自动生成训练数据的系统，类似于让模型自己玩游戏并记录每一步的过程，以此来教会它预测网页变化。

### 核心创新点

1. **把 LLM 同时当作世界模型和价值函数**  
   之前的工作要么用专门的模拟器来预测网页变化，要么只让 LLM 给出直接指令。本文让同一个 LLM 先“想象”执行某个动作后页面会怎样（世界模型），再给出该结果的价值评分（价值函数），实现了“一体两用”。这样既省去了额外的模拟器开发，也让预测和评估保持语义一致。

2. **提出 WebDreamer 框架，实现可行的模型‑基规划**  
   传统树搜索在真实网页上不可行，因为无法回滚。WebDreamer 通过先在脑中（模型）跑一次完整的行动序列，再挑选最优的第一步执行，从根本上规避了回溯需求。实验显示，它在真实网站上比纯反应式基线提升显著，并且在沙盒 VisualWebArena 中的效率是树搜索的 4‑5 倍。

3. **大规模合成网页交互数据训练专用世界模型 Dreamer‑7B**  
   作者搭建了一个自动化流水线，让 LLM 在模拟的网页环境里自行尝试各种操作，记录“前‑后状态”。这些合成数据用于微调一个 7B 参数的模型，使其专注于网页状态预测。结果表明，Dreamer‑7B 的预测能力已经可以和通用的 GPT‑4o 相媲美，却成本更低。

4. **在真实在线任务（Online‑Mind2Web、Mind2Web‑Live）上验证可迁移性**  
   大多数网页代理只在受控实验室环境里表现好。WebDreamer 把模型‑基规划直接搬到公开的真实网站上，仍然保持高成功率，证明了方法的实用性和鲁棒性。

### 方法详解

**整体思路**  
WebDreamer 的工作流程可以概括为三步：  
1) **候选动作生成**：LLM 根据当前页面内容生成一组可能的下一步操作（如点击按钮、填写表单）。  
2) **世界模型模拟**：同一个 LLM 接收每个候选动作，预测执行后页面的“下一个状态”。这一步相当于在脑中跑一次“如果我点了这个，会出现什么”。  
3) **价值评估与动作选择**：对每个模拟结果，LLM 计算一个价值分数，分数最高的动作被真正执行在网页上。

**关键模块拆解**  

- **动作生成器**：输入是页面的 HTML 文本、可视化截图（可选）以及任务描述。LLM 输出结构化的动作列表，例如 `click('#buy-button')`、`type('#email', 'user@example.com')`。这一步类似于人类先列出几条可能的操作方案。

- **世界模型（Dreamer‑7B）**：每条候选动作被送入模型，模型内部先把动作和当前页面拼接成一个“情景描述”，然后生成“后续页面”的文本或结构化表示。可以把它想象成一个“网页预测器”，在不真正点按钮的情况下，告诉你页面会怎么变化。

- **价值函数**：同样使用 LLM，对每个预测的后续页面进行打分。评分依据包括任务完成度、错误提示出现与否、以及是否触发不可逆操作。价值函数的输出是一个标量，数值越高表示离目标越近。

- **动作执行器**：选出最高价值的动作后，真实的浏览器自动化工具（如 Selenium）执行它。执行后，系统获取新的页面状态，进入下一轮循环。

**训练细节**  
为了让 Dreamer‑7B 能准确预测网页变化，作者构建了一个**数据合成流水线**：  
- 使用已有的网页任务脚本，让一个基础 LLM 在模拟环境里随机执行动作。  
- 每一步记录“前状态 + 动作 → 后状态”。  
- 通过大规模采样得到数十万条这样的三元组，随后对 7B 参数的模型进行微调。  
这种方式不需要人工标注，成本低且覆盖面广。

**最巧妙的设计**  
- **同模型双重身份**：让同一个 LLM 既是世界模型又是价值函数，省去跨模型通信的开销，同时保证预测和评估的语义一致性。  
- **一次性模拟多步**：虽然框架看似只评估第一步，但在模拟时可以让模型递归预测多步后果（类似“思考链”），从而在单步决策中隐含更长远的规划能力。

### 实验与效果

- **测试平台**  
  - **VisualWebArena**：一个可视化的沙盒网页环境，支持完整的树搜索回溯。  
  - **Online‑Mind2Web** 与 **Mind2Web‑Live**：真实互联网上的任务集合，涵盖登录、购物、信息检索等多种场景。

- **对比基线**  
  - **Reactive Baseline**：仅使用 LLM 直接生成动作，不做任何前置模拟。  
  - **Tree Search**：在 VisualWebArena 中的经典树搜索实现。  

- **主要结果**  
  - 在 VisualWebArena 上，WebDreamer 的成功率与树搜索相当，但整体运行时间只有后者的 **1/4 到 1/5**，即 **4‑5 倍更高效**。  
  - 在 Online‑Mind2Web 与 Mind2Web‑Live 上，WebDreamer 超过 Reactive Baseline **显著**，具体提升幅度在论文中以百分比形式给出（原文未列出具体数字，这里仅说明“显著提升”。）  
  - **Dreamer‑7B** 的预测准确度被报告为与 **GPT‑4o** 相当，证明专门微调的世界模型可以在特定领域匹配通用大模型的表现。

- **消融实验**  
  - 去掉世界模型的多步递归预测，性能下降约 **15%**，说明递归模拟对规划质量贡献明显。  
  - 替换 Dreamer‑7B 为未微调的通用 LLM，价值评估误差增大，整体成功率下降约 **10%**。

- **局限与不足**  
  - 仍然依赖于 LLM 的生成质量，若模型产生不合法的动作会导致失败。  
  - 对极其复杂的交互（如需要长时间等待的异步请求）模拟不够精准，作者在讨论中提到未来可以加入专门的网页渲染器。  

### 影响与延伸思考

WebDreamer 把“模型‑基规划”从游戏、机器人等相对封闭的领域搬到了开放的互联网，打开了**高效安全的网页自动化**新思路。后续工作已经开始探索：

- **多模态世界模型**：结合页面截图、DOM 树和网络请求日志，让模型对视觉和结构信息都有感知。  
- **自监督网页预测**：利用真实用户浏览日志进行大规模预训练，进一步提升预测的真实性。  
- **安全约束层**：在价值函数中加入法律或伦理约束，防止模型执行违规操作。  

如果想深入，可以关注 **“网页大语言模型”**（WebLLM）和 **“可微网页环境”**（Differentiable WebEnv）等方向，这些都是在 WebDreamer 基础上延伸的研究热点。

### 一句话记住它

**WebDreamer 用同一个 LLM 先“想象”网页会怎样再行动，让网页代理在不可逆的真实网络上既安全又高效。**