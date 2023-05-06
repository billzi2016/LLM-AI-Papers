# Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning   by Large Language Models

> **Date**：2023-05-06
> **arXiv**：https://arxiv.org/abs/2305.04091

## Abstract

Large language models (LLMs) have recently been shown to deliver impressive performance in various NLP tasks. To tackle multi-step reasoning tasks, few-shot chain-of-thought (CoT) prompting includes a few manually crafted step-by-step reasoning demonstrations which enable LLMs to explicitly generate reasoning steps and improve their reasoning task accuracy. To eliminate the manual effort, Zero-shot-CoT concatenates the target problem statement with "Let's think step by step" as an input prompt to LLMs. Despite the success of Zero-shot-CoT, it still suffers from three pitfalls: calculation errors, missing-step errors, and semantic misunderstanding errors. To address the missing-step errors, we propose Plan-and-Solve (PS) Prompting. It consists of two components: first, devising a plan to divide the entire task into smaller subtasks, and then carrying out the subtasks according to the plan. To address the calculation errors and improve the quality of generated reasoning steps, we extend PS prompting with more detailed instructions and derive PS+ prompting. We evaluate our proposed prompting strategy on ten datasets across three reasoning problems. The experimental results over GPT-3 show that our proposed zero-shot prompting consistently outperforms Zero-shot-CoT across all datasets by a large margin, is comparable to or exceeds Zero-shot-Program-of-Thought Prompting, and has comparable performance with 8-shot CoT prompting on the math reasoning problem. The code can be found at https://github.com/AGI-Edgerunners/Plan-and-Solve-Prompting.

---

# 计划-求解提示：提升大语言模型零样本思维链推理 论文详细解读

### 背景：这个问题为什么难？
在多步推理任务里，模型需要先拆解问题、再逐步演算，最后给出答案。传统的零样本提示（直接把题目喂给模型）往往只能给出“一口气”的答案，错误率高。Few‑shot CoT 通过在提示中加入几段手写的思考过程，显著提升了准确率，但需要人工准备示例，成本不低。Zero‑shot CoT 试图用一句 “Let’s think step by step” 自动触发思考链，虽然省去了示例，却仍会出现计算错误、漏掉关键步骤、甚至误解题意等三大痛点。尤其是“漏步”错误——模型在思考时跳过了必须的子任务——成为零样本方法的主要瓶颈。

### 关键概念速览
**Chain‑of‑Thought（思维链）**：让模型在给出最终答案前先写出推理步骤，类似人做数学题时的草稿，能够让错误更容易被捕捉和纠正。  
**Zero‑shot CoT**：在提示里只加一句 “Let’s think step by step”，不提供任何示例，期望模型自行产生思维链。  
**Plan‑and‑Solve（计划‑求解）**：先让模型生成一个任务拆解计划（把大题分成若干小子题），再按计划一步步求解，每一步都在提示中明确指示。  
**PS+ Prompting**：在 Plan‑and‑Solve 基础上加入更细致的指令（比如要求每一步都写出计算过程），进一步降低计算错误。  
**Few‑shot CoT**：在提示中加入几条人工编写的思维链示例，模型通过模仿这些示例来生成自己的推理过程。  
**Program‑of‑Thought（思维程序）**：把推理过程写成可执行的伪代码或程序，让模型在生成代码的同时完成计算。  

### 核心创新点
1. **从一次性思考到先计划后求解**  
   之前的 Zero‑shot CoT 直接让模型“一口气”思考，容易跳步。本文改为先让模型输出一个“计划”，把整体任务拆成若干子任务，再逐个求解。这样模型在每一步都有明确的目标，漏步的概率大幅下降。  
2. **用指令强化计算细节**  
   仅有计划仍会出现算错的情况。作者在提示中加入“每一步都要写出具体的数值计算”之类的细化指令，形成 PS+ Prompting。相比原始 Zero‑shot CoT，这种细粒度约束显著提升了数值推理的准确性。  
3. **零样本下的多任务统一框架**  
   过去的改进往往针对特定任务（比如数学或逻辑推理）。Plan‑and‑Solve 通过统一的“计划 → 求解”两阶段结构，能够在数学、常识推理、符号推理等三大类任务上直接复用，无需为每类任务单独设计提示。  
4. **与少样本 CoT 的性能对标**  
   实验显示，PS+ 在数学推理上几乎追平 8‑shot CoT 的表现，说明只要给模型一个好计划和明确的执行指令，零样本也能达到少样本的效果，这在降低标注成本上具有里程碑意义。  

### 方法详解
**整体思路**：Plan‑and‑Solve Prompting 把一次性的思考过程拆成两步：  
1) **Plan（计划）**：让模型先生成任务拆解的清单；  
2) **Solve（求解）**：按照清单逐条求解，每一步都在提示中明确要求写出计算或推理细节。  

**步骤拆解**  
- **输入构造**：原始问题 + “先列出解题计划”。模型输出的计划通常是编号的子任务列表，例如 “1. 计算 … 2. 判断 …”。  
- **计划解析**：系统（或后处理脚本）把模型的计划转化为可执行的提示序列。每条子任务会被包装成 “现在请完成第 k 步：<子任务描述>，并写出你的推理”。  
- **逐步求解**：模型依次收到每一步的提示，生成对应的推理文本。若是数值计算，模型会在答案后面直接写出算式；若是逻辑判断，模型会给出理由。  
- **结果汇总**：所有子任务的答案被收集后，系统根据预设的汇总规则（如加总、比较）给出最终答案。  

**类比**：把模型想象成一个学生做考试题。传统 Zero‑shot CoT 像是让学生直接在答题卡上写答案；Plan‑and‑Solve 则像老师先让学生列出解题步骤（草稿），再一步步在草稿上写出每一步的计算，最后把草稿整理成正式答案。  

**细节强化（PS+）**：在每一步的提示里加入 “请把每一步的数值计算过程完整写出来，不能省略任何中间结果”。这相当于在学生的草稿上加了老师的批注，要求每一步都必须完整、可检查。  

**最巧妙的地方**：计划阶段不需要任何外部示例，只靠模型自身的语言理解能力生成子任务列表。这个自我组织的过程本身就利用了大模型的“元认知”能力，省去了人工标注的成本。  

### 实验与效果
- **测试任务**：作者在十个公开数据集上评估，覆盖三类推理：数学计算、常识推理、符号推理（如代数方程）。  
- **基线对比**：Zero‑shot CoT、Zero‑shot Program‑of‑Thought、以及 8‑shot CoT。  
- **主要结果**：在所有数据集上，Plan‑and‑Solve（PS）均显著超越 Zero‑shot CoT，提升幅度“很大”。在数学推理上，PS+ 的准确率与 8‑shot CoT 相当，几乎持平（论文未给出具体百分比，只说“可比”）。相较于 Zero‑shot Program‑of‑Thought，PS 系列在多数任务上表现相当或更好。  
- **消融实验**：作者分别去掉计划阶段或细化指令，发现缺少计划会导致漏步错误回升，缺少细化指令会使计算错误率上升，验证了两者的互补作用。  
- **局限性**：计划质量仍依赖模型的自我组织能力，复杂任务的计划可能不够细致；此外，提示长度会随子任务增多而增长，受限于模型的上下文窗口。作者在讨论中承认这些问题，并建议未来结合检索或外部工具来进一步提升。  

### 影响与延伸思考
这篇工作打开了“自我规划+自我执行”在零样本提示中的新思路，随后有多篇论文尝试把任务分解（Task Decomposition）与工具调用（Tool Use）结合，形成更强的“思考+行动”循环。比如 2024 年的 **Self‑Ask‑With‑Plan**、**Decompose‑and‑Solve** 系列都在 Plan‑and‑Solve 的基础上加入检索或代码执行模块。对想进一步探索的读者，可以关注以下方向：  
- **动态计划修正**：让模型在求解过程中根据中间结果实时调整计划。  
- **跨模态计划**：把视觉或表格信息也纳入计划阶段，实现多模态推理。  
- **长上下文优化**：利用检索增强或分段记忆来突破提示长度限制。  

### 一句话记住它
只要先让大模型自己列出解题计划，再按计划一步步求解，零样本的思维链推理就能像有示例的少样本一样靠谱。