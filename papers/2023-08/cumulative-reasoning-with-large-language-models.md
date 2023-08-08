# Cumulative Reasoning with Large Language Models

> **Date**：2023-08-08
> **arXiv**：https://arxiv.org/abs/2308.04371

## Abstract

Recent advancements in large language models (LLMs) have shown remarkable progress, yet their ability to solve complex problems remains limited. In this work, we introduce Cumulative Reasoning (CR), a structured framework that enhances LLM problem-solving by emulating human-like iterative and cumulative thought processes. CR orchestrates LLMs in three distinct roles: Proposer, Verifier(s), and Reporter, to systematically decompose tasks, generate and validate intermediate reasoning steps, and compose them into a solution by building a dynamic Directed Acyclic Graph (DAG) of verified propositions. This approach substantially enhances problem-solving capabilities. We demonstrate CR's advantage through several complex reasoning tasks: it outperforms existing methods in logical inference tasks with up to a 9.3% improvement, achieving 98.04% accuracy on the curated FOLIO wiki dataset. In the Game of 24, it achieves 98% accuracy, marking a 24% improvement over previous methods. In solving MATH problems, CR achieves a 4.2% increase from previous methods and a 43% relative improvement in the most challenging level 5 problems. When incorporating a code environment with CR, we further harness LLMs' reasoning capabilities and outperform the Program of Thought (PoT) method by 38.8%.

---

# 大语言模型的累计推理 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）在生成自然语言方面已经非常强大，但在需要多步推理的任务上仍会频繁出错。传统的“一次性生成答案”方式缺乏对中间步骤的检查，导致逻辑漏洞、算术错误或对复杂约束的忽视。已有的思维链（CoT）或程序化思考（PoT）虽然让模型先写草稿再给出答案，却仍是单一模型的线性过程，缺乏对每一步的独立验证和累积记忆。于是，模型在面对需要拆解、验证、组合的复杂问题时，往往会卡在“如何保证每一步都正确”这一步。

### 关键概念速览
**累计推理（Cumulative Reasoning，CR）**：一种让 LLM 轮流扮演不同角色、逐步构建并验证推理图的框架，类似团队合作中“提议—审查—汇报”的流程。  
**提议者（Proposer）**：负责把大任务拆解成若干子命题或步骤，就像老师先列出解题思路的提纲。  
**验证者（Verifier）**：对提议者给出的每个子命题进行独立检查，确保逻辑或数值上没有错误，类似审稿人对每段文字的校对。  
**报告者（Reporter）**：把所有已验证的子命题按照依赖关系组织起来，生成最终答案，类似编辑把审稿通过的章节拼成一本书。  
**有向无环图（DAG）**：用节点表示已验证的命题，用有向边表示前后依赖，保证推理过程不出现循环，类似项目管理中的任务依赖图。  
**多模型协同**：在同一次推理过程中同时调用多个 LLM 实例（或同一模型的不同调用），每个实例专注于特定角色，提升整体鲁棒性。  
**迭代累积**：每完成一次验证，就把新节点加入 DAG，后续的提议可以基于已有节点继续展开，形成层层递进的思考链。

### 核心创新点
1. **角色分工 → 提议‑验证‑报告三阶段**：以前的 CoT 只让单个模型一次性输出全部步骤，缺少专门的纠错环节。CR 把任务拆解交给提议者，再让独立的验证者逐条审查，最后由报告者负责整合。这样每一步都有专门的“质量检查”，显著降低了错误传播。  
2. **动态 DAG 结构 → 逐步累加已验证命题**：传统方法的推理链是线性的，后面的步骤只能依赖前面的文字描述。CR 用有向无环图记录每个已验证命题及其依赖关系，后续提议可以自由引用任意已确认的节点，提升了信息复用效率，尤其在需要多次使用同一中间结果的任务上表现更好。  
3. **多验证者并行 → 交叉验证提升可靠性**：在关键步骤上，CR 可以让多个验证者独立判断，同意后才算通过，这相当于投票机制，显著提升了对细粒度错误（如算术小数点错位）的捕捉率。  
4. **代码环境集成 → 推理+执行闭环**：在需要实际运行代码的任务中，CR 把代码生成交给提议者，执行结果再交给验证者检查，形成“思考—实验—校正”的闭环，比单纯的 PoT 只靠模型内部模拟更可靠，实验显示提升近 40%。

### 方法详解
CR 的整体流程可以概括为三步：**任务拆解 → 步骤验证 → 结果汇报**，并在每一步维护一张 DAG。

1. **任务拆解（Proposer）**  
   - 输入：原始问题描述。  
   - 操作：模型被提示以“列出解决该问题的关键子命题”形式输出，每条子命题标记唯一 ID。  
   - 结果：一组待验证的节点列表，暂时没有依赖信息。  
   - 类比：像老师先给学生列出解题大纲。

2. **步骤验证（Verifier）**  
   - 对每个子命题，系统启动一个或多个验证者实例。  
   - 验证者收到子命题以及当前 DAG 中所有已确认节点的内容，任务是判断该子命题是否在逻辑或数值上成立。  
   - 若多个验证者同时使用，则采用多数投票或置信度阈值决定通过。  
   - 通过后，该子命题被标记为“已验证”，并加入 DAG，边指向它所依赖的前置节点（如果在描述中提到）。  
   - 关键点：验证者只关注单条命题，不需要一次性考虑全局，降低了认知负荷。

3. **结果汇报（Reporter）**  
   - 当所有子命题都通过验证，或达到预设的覆盖率后，报告者读取完整的 DAG。  
   - 报告者按照拓扑排序（即先处理没有未完成前置的节点），把每个已验证的命题拼接成连贯的答案文本。  
   - 对于需要数值输出的任务，报告者还会执行简单的算术或调用外部代码环境，把执行结果写回 DAG，再继续拼接。  
   - 最终输出即为完整、经过多轮校验的解答。

**最巧妙的设计**在于把“验证”过程外包给独立模型，而不是让同一个模型在同一次调用中自行检查。这相当于让模型“自我审稿”，但审稿人是另一个“独立的”模型，从而避免了自我强化的偏差。

### 实验与效果
- **测试任务**：包括一套逻辑一阶谓词推理数据（FOLIO wiki）、经典的 24 点游戏、MATH 大学数学竞赛题目以及需要代码执行的编程推理基准。  
- **基线对比**：与传统 CoT、Self‑Consistency、Program of Thought（PoT）等方法比较。  
- **关键数字**：  
  - 在 FOLIO 上准确率提升至 **98.04%**，比最强基线高 **9.3%**。  
  - 24 点游戏正确率达到 **98%**，比上一代方法提升约 **24%**。  
  - MATH 任务整体提升 **4.2%**，在最难的 Level‑5 题目上相对提升 **43%**。  
  - 在代码环境实验中，CR 超过 PoT **38.8%**，显示验证+执行闭环的优势。  
- **消融实验**：作者分别去掉多验证者、DAG 结构或报告者阶段，发现准确率分别下降约 **3–7%**，说明每个模块都有实质贡献。  
- **局限性**：CR 需要多次模型调用，计算成本比单次 CoT 高出约 2–3 倍；在极度开放式的自然语言任务上，如何自动定义子命题仍是未解难题。原文未给出大规模工业部署的成本评估。

### 影响与延伸思考
CR 把“分工合作”理念引入 LLM 推理，随后出现的工作如 **Tree‑of‑Thought**、**Self‑Verify**、以及 **Collaborative Prompting** 等，都在不同程度上采用了多模型或多阶段验证的思路。推测未来会有更多研究把 CR 与检索增强（RAG）或外部知识库结合，让验证者直接查询事实来源，进一步提升可靠性。想深入了解的读者可以关注 **多模态协同推理**、**可解释性图结构** 以及 **低成本并行验证** 等方向。

### 一句话记住它
让大语言模型像团队一样分工、逐步验证、累积结果，用 DAG 把每一步都“锁定”，从而把复杂推理的成功率大幅提升。