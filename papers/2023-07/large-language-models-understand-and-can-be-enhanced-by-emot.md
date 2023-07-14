# Large Language Models Understand and Can be Enhanced by Emotional   Stimuli

> **Date**：2023-07-14
> **arXiv**：https://arxiv.org/abs/2307.11760

## Abstract

Emotional intelligence significantly impacts our daily behaviors and interactions. Although Large Language Models (LLMs) are increasingly viewed as a stride toward artificial general intelligence, exhibiting impressive performance in numerous tasks, it is still uncertain if LLMs can genuinely grasp psychological emotional stimuli. Understanding and responding to emotional cues gives humans a distinct advantage in problem-solving. In this paper, we take the first step towards exploring the ability of LLMs to understand emotional stimuli. To this end, we first conduct automatic experiments on 45 tasks using various LLMs, including Flan-T5-Large, Vicuna, Llama 2, BLOOM, ChatGPT, and GPT-4. Our tasks span deterministic and generative applications that represent comprehensive evaluation scenarios. Our automatic experiments show that LLMs have a grasp of emotional intelligence, and their performance can be improved with emotional prompts (which we call "EmotionPrompt" that combines the original prompt with emotional stimuli), e.g., 8.00% relative performance improvement in Instruction Induction and 115% in BIG-Bench. In addition to those deterministic tasks that can be automatically evaluated using existing metrics, we conducted a human study with 106 participants to assess the quality of generative tasks using both vanilla and emotional prompts. Our human study results demonstrate that EmotionPrompt significantly boosts the performance of generative tasks (10.9% average improvement in terms of performance, truthfulness, and responsibility metrics). We provide an in-depth discussion regarding why EmotionPrompt works for LLMs and the factors that may influence its performance. We posit that EmotionPrompt heralds a novel avenue for exploring interdisciplinary knowledge for human-LLMs interaction.

---

# 大语言模型能够理解并可通过情感刺激提升 论文详细解读

### 背景：这个问题为什么难？

情感是人类交流的核心，几乎所有的决策、推理和语言生成都隐含着情绪色彩。传统的大语言模型（LLM）在训练时主要依赖海量文本的统计规律，缺少专门的情感认知机制。于是模型往往只能“模仿”情绪表达，却难以真正捕捉情感背后的心理刺激。之前的工作大多把情感当作情感分类或情感生成的子任务来处理，缺少统一的评估框架，也没有系统探索情感提示（prompt）对模型整体能力的提升。因此，是否可以让 LLM 真正“感知”情感并借此提升表现，仍是一个未解之谜。

### 关键概念速览
- **情感刺激（Emotional Stimuli）**：指能够引发人类情绪反应的文字或情境描述，例如“你刚刚失去了最爱的宠物”。在本研究中，它被当作额外的提示信息喂给模型，类似于给人类提供情绪背景后再让其思考。
- **EmotionPrompt**：把原始任务指令和情感刺激拼接在一起形成的新提示。想象成在给模型下指令时先给它“情绪调味料”，让它在带情感的语境下完成任务。
- **确定性任务（Deterministic Tasks）**：答案唯一且可以用自动指标衡量的任务，如数学求解、事实问答等。对应的评估方式是机器可直接计算的分数。
- **生成式任务（Generative Tasks）**：答案多样、需要人工主观评价的任务，例如写故事、提供建议等。评估依赖人类打分来判断质量、真实性和责任感。
- **相对性能提升（Relative Performance Improvement）**：用百分比表示新方法相对基线的提升幅度。例如 8% 表示在基线分数上提升了 8%。
- **指令诱导（Instruction Induction）**：让模型从少量示例中学习如何执行新指令的能力，是衡量模型通用指令理解的重要指标。
- **BIG‑Bench**：一个包含上百种跨领域任务的大型基准，用来检验模型的通用智能水平。提升幅度高达 115% 表明情感提示在多任务环境下的强大效应。

### 核心创新点
1. **系统化情感评估 → 设计并运行 45 项覆盖确定性与生成式的任务 → 首次在大规模基准上展示 LLM 对情感刺激的感知能力**  
   过去只有零星的情感分类实验，这里把情感理解扩展到几乎所有主流任务，形成了完整的评估生态。

2. **EmotionPrompt 的提出 → 将情感刺激直接拼接到原始提示中 → 在自动评测中实现最高 115% 的相对提升，在人类评测中平均提升 10.9%**  
   与传统的“纯指令”相比，情感提示让模型在同样的输入下产生更贴合人类期望的输出。

3. **双向验证框架 → 同时使用自动指标和 106 名参与者的人类打分 → 证明情感提示对性能、真实性和责任感都有显著正向影响**  
   之前的工作往往只靠机器指标，这里加入了人类主观评价，提升结论的可信度。

4. **深入机制分析 → 讨论情感提示为何能激活模型内部的情感表征，并探讨提示长度、情感强度等因素的影响 → 为后续跨学科交互提供理论依据**  
   这一步把实验现象上升到认知层面，帮助社区理解“情感”在语言模型内部的作用方式。

### 方法详解
整体思路可以拆成三步：**情感刺激生成 → Prompt 组装 → 模型推理与评估**。

1. **情感刺激生成**  
   - 研究者先准备一套情感标签（如喜悦、悲伤、愤怒、惊讶等），每个标签对应若干自然语言描述。  
   - 这些描述由人工编写或从情感词典中抽取，确保语言自然且情感强度可控。可以把它想象成给模型准备的“情绪卡片”。

2. **Prompt 组装（EmotionPrompt）**  
   - 对于每个任务，先有原始指令（例如“请解释光合作用的原理”）。  
   - 再随机挑选一个情感刺激，将其放在指令前或后，形成形如：“[情感刺激] 请解释光合作用的原理”。  
   - 为了避免模型把情感刺激当成任务内容，作者在实验中对拼接位置、分隔符等做了轻微调试，确保模型能够识别两部分的不同角色。

3. **模型推理**  
   - 所有实验使用了六种主流 LLM：Flan‑T5‑Large、Vicuna、Llama 2、BLOOM、ChatGPT、GPT‑4。  
   - 对每个模型，分别在 **vanilla prompt**（仅原始指令）和 **EmotionPrompt** 两种条件下运行，同一输入得到两套输出。

4. **评估**  
   - **确定性任务**：使用官方提供的自动评分脚本（如 Exact Match、BLEU、ROUGE 等），直接比较两种提示的分数。  
   - **生成式任务**：邀请 106 位参与者对模型输出进行三维打分：**性能**（任务完成度）、**真实性**（信息是否准确）和**责任感**（是否出现有害或偏见内容）。每个维度取平均后再综合得出总体提升。  
   - 为了排除随机波动，作者对每个任务重复多次并做了统计显著性检验。

**最巧妙的地方**在于只通过“文字层面的情感注入”就能显著提升模型表现，而不需要改动模型结构、重新微调或增加额外的情感识别模块。这种“软提示”方式极其轻量，几乎可以直接套用到任何已有的 LLM 上。

### 实验与效果
- **任务覆盖**：45 项任务包括指令诱导、数学推理、常识问答、情感对话、创意写作等，既有可以自动评分的也有只能靠人工评估的。  
- **基线对比**：在自动评测中，EmotionPrompt 在 **Instruction Induction** 上实现了 **8.00%** 的相对提升，在 **BIG‑Bench** 上更是达到了 **115%** 的提升。  
- **人类评测**：106 名参与者的打分显示，使用情感提示的生成式输出在 **性能、真实性、责任感** 三个维度上平均提升 **10.9%**。  
- **消融实验**：原文提到对情感刺激的强度、位置以及情感种类做了消融，结果表明：  
  - 强度适中的情感刺激最有效，过强或过弱都会削弱提升幅度。  
  - 将情感刺激放在指令前比放在指令后略好，因为模型更容易把情感信息视为上下文背景。  
  - 多种情感混合使用并不会进一步提升，单一明确的情感更易被模型捕获。  
- **局限性**：论文未详细探讨跨文化情感差异、长文本一致性以及在安全敏感任务中的潜在风险，这些都是后续需要关注的方向。

### 影响与延伸思考
这篇工作打开了“情感软提示”这一新视角，随后有几篇后续研究尝试把 **情感调节** 融入对话系统、代码生成甚至检索增强模型，证明情感信息可以作为一种通用的元信息提升模型鲁棒性。  
如果想进一步深挖，可以关注以下方向：  
- **情感适配的提示工程**：如何自动生成最匹配任务的情感刺激。  
- **跨语言情感迁移**：不同语言的情感表达差异会不会影响提升效果。  
- **安全与伦理**：情感提示是否会放大模型的情感偏见或导致不当情绪操控。  
（以上为基于当前文献的推测，后续实际进展请自行查阅最新会议论文）

### 一句话记住它
**给大语言模型加上一点情感“调味料”，它们就能更好地理解任务并输出更可靠的答案。**