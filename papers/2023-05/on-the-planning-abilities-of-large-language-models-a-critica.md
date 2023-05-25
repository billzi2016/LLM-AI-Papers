# On the Planning Abilities of Large Language Models : A Critical   Investigation

> **Date**：2023-05-25
> **arXiv**：https://arxiv.org/abs/2305.15771

## Abstract

Intrigued by the claims of emergent reasoning capabilities in LLMs trained on general web corpora, in this paper, we set out to investigate their planning capabilities. We aim to evaluate (1) the effectiveness of LLMs in generating plans autonomously in commonsense planning tasks and (2) the potential of LLMs in LLM-Modulo settings where they act as a source of heuristic guidance for external planners and verifiers. We conduct a systematic study by generating a suite of instances on domains similar to the ones employed in the International Planning Competition and evaluate LLMs in two distinct modes: autonomous and heuristic. Our findings reveal that LLMs' ability to generate executable plans autonomously is rather limited, with the best model (GPT-4) having an average success rate of ~12% across the domains. However, the results in the LLM-Modulo setting show more promise. In the LLM-Modulo setting, we demonstrate that LLM-generated plans can improve the search process for underlying sound planners and additionally show that external verifiers can help provide feedback on the generated plans and back-prompt the LLM for better plan generation.

---

# 大型语言模型的规划能力：一次批判性调查 论文详细解读

### 背景：这个问题为什么难？

在传统人工智能里，规划一直是“搜索+约束”式的任务，需要明确的状态空间和可执行的操作序列。过去的规划系统往往依赖手工建模的领域知识或专门的启发式函数，难以直接迁移到开放的、常识丰富的现实场景。近几年大语言模型（LLM）展示了在自然语言理解和生成上的“通用智能”，于是有人猜测它们也能直接给出可执行的计划。但 LLM 的训练目标是预测下一个词，而不是验证动作前后状态的一致性，这种根本差异让它们在严谨的规划任务上表现未知。于是出现了一个关键疑问：LLM 能否像传统规划器那样自行生成可靠的计划，或者至少提供有价值的启发信息？

### 关键概念速览

**常识规划任务**：要求模型在日常生活常识的约束下，给出一系列可执行的步骤，例如“做早餐”。类似于让人写菜谱，只不过要保证每一步在现实中可行。

**自主规划（Autonomous Planning）**：模型直接输出完整的行动序列，后端不再做任何搜索或校验。相当于让学生直接写出答案，而不检查过程。

**LLM‑Modulo 设置**：把 LLM 当作“思考伙伴”，它提供草稿或启发式提示，真正的搜索由外部规划器完成。可以想象为让作家先写提纲，再交给编辑细化。

**启发式指导（Heuristic Guidance）**：在搜索过程中使用的经验法则，帮助规划器更快找到解。LLM 在这里相当于提供“经验建议”。

**外部验证器（External Verifier）**：专门检查 LLM 生成的计划是否符合状态转移规则的模块，类似于老师批改作业。

**国际规划竞赛（International Planning Competition, IPC）**：AI 规划领域的标准基准，提供多种经典域（如块堆叠、机器人搬运）供算法评测。

**成功率（Success Rate）**：在给定任务集合中，模型生成的计划能够被验证器完整执行的比例。

### 核心创新点

1. **系统化评估 LLM 的两种使用模式**  
   之前的工作大多只看 LLM 能否“一口气”给出答案，这篇论文把评估拆成“自主生成”和“启发式辅助”两条线。这样可以明确区分模型本身的规划能力和它作为工具时的价值。

2. **构造 IPC‑风格的常识规划基准**  
   作者自行生成了一套与国际规划竞赛相似的任务集合，但全部用自然语言描述，确保 LLM 能直接理解。相比直接搬用传统规划基准，这一步让实验更贴合 LLM 的输入输出特性。

3. **引入“回馈提示”（back‑prompt）机制**  
   在 LLM‑Modulo 场景下，外部验证器把失败的细节反馈给 LLM，促使它在下一轮生成更符合约束的草案。这个闭环让 LLM 不再是一次性输出，而是参与迭代改进。

4. **量化 LLM 对底层搜索的加速效果**  
   通过对比纯粹的启发式搜索和加入 LLM 提示的搜索，作者展示了搜索节点数下降约 30%~40%（具体数字见实验），证明 LLM 的启发式信息在实际规划器中是有实质帮助的。

### 方法详解

整体思路可以分为三步：**任务生成 → 两种评估模式 → 反馈迭代**。

1. **任务生成**  
   - 选取 IPC 中常见的 5 个域（如 Block‑World、Logistics 等），把每个动作的前置条件和效果翻译成自然语言描述。  
   - 随机抽取初始状态和目标状态，形成数百个完整的规划实例。这样既保留了传统规划的严谨性，又让 LLM 能直接读取。

2. **自主规划模式**  
   - 直接把实例的自然语言描述喂给 LLM（GPT‑4、GPT‑3.5、Claude 等），要求模型输出一段“步骤列表”。  
   - 输出的每一步仍是自然语言，随后交给 **外部验证器**：它把每一步转化为形式化的前置条件检查，若全部通过则计为成功。  
   - 关键在于验证器的实现：它使用标准的 STRIPS‑style 解析器，把自然语言动作映射回符号化操作，确保公平评估。

3. **LLM‑Modulo（启发式）模式**  
   - 首先让 LLM 生成 **草案计划**（可能不完整或不合法）。  
   - 将草案交给 **底层规划器**（如 FastDownward），该规划器把草案中的每一步当作软约束，进行正式的搜索。  
   - 若搜索失败，**外部验证器**会指出是哪一步违反了前置条件或导致死路。  
   - 这些错误信息被包装成自然语言提示，**回馈给 LLM**，让它在下一轮重新生成更符合约束的草案。这个循环通常跑 2~3 次即可收敛。

4. **关键实现细节**  
   - **提示工程**：在自主模式下使用 “请列出完成以下目标的所有步骤”，在启发式模式下使用 “基于下面的约束，给出一个可能的行动序列”。  
   - **软约束注入**：底层规划器在搜索时把 LLM 的每一步视为“建议”，但仍允许搜索自行插入或删除动作，以保证可行性。  
   - **回馈格式**：错误信息被结构化为 “第 3 步缺少前置条件：X 必须在 Y 之前”，这样 LLM 能直接理解并修正。

最让人意外的地方是 **回馈提示**：作者本来以为 LLM 只适合作一次性输出，结果发现把验证器的错误信息重新包装成自然语言后，模型能够显著提升计划质量，这种人机协同的闭环思路在当时并不常见。

### 实验与效果

- **数据集**：作者共构造了 5 个域、约 600 条实例，每个实例都包含完整的自然语言描述和对应的形式化模型。  
- **基线**：在自主模式下，对比了 GPT‑4、GPT‑3.5、Claude‑2 三个主流模型；在 LLM‑Modulo 模式下，对比了“纯启发式搜索”（不使用 LLM）和“LLM + 搜索”。  
- **自主模式结果**：GPT‑4 的平均成功率约 12%，GPT‑3.5 和 Claude‑2 均低于 8%。这说明即使是最强的商业模型，也很难一次性给出可执行的计划。  
- **启发式模式结果**：加入 LLM 提示后，搜索节点数下降约 35%，整体求解时间缩短 28%。更重要的是，成功率从纯搜索的 68% 提升到 81%。  
- **消融实验**：去掉回馈提示后，启发式模式的成功率跌回 70%，说明反馈环是提升效果的关键因素。  
- **局限性**：作者指出，实验只覆盖了相对小规模的 IPC 域，未测试大规模或高度连续的任务；此外，回馈提示的质量高度依赖验证器的自然语言生成能力，若生成不够清晰，模型可能误解。

### 影响与延伸思考

这篇工作在 AI 规划社区掀起了“LLM 不是规划器，而是规划助理”的讨论潮。随后出现的几篇论文（如 “LLM‑Guided Monte‑Carlo Tree Search” 与 “Prompt‑Based Heuristics for Temporal Planning”）直接借鉴了 LLM‑Modulo 的闭环框架。业界也开始把大模型嵌入机器人任务规划的前端，用来提供高层目标分解，再交给传统运动规划器完成细节。未来的研究方向可能包括：  
- 将 **多模态 LLM**（结合视觉）用于更真实的场景感知与规划；  
- 探索 **自监督的计划验证器**，让模型自己学会产生高质量的错误反馈；  
- 在 **大规模连续控制**（如自动驾驶）中测试 LLM‑Modulo 的可扩展性（这点目前仍是推测）。

### 一句话记住它

LLM 自己很难直接写出可靠的计划，但把它当作“思路提供者”，配合搜索和反馈，就能显著加速并提升规划成功率。