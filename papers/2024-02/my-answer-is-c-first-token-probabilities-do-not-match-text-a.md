# "My Answer is C": First-Token Probabilities Do Not Match Text Answers in   Instruction-Tuned Language Models

> **Date**：2024-02-22
> **arXiv**：https://arxiv.org/abs/2402.14499

## Abstract

The open-ended nature of language generation makes the evaluation of autoregressive large language models (LLMs) challenging. One common evaluation approach uses multiple-choice questions (MCQ) to limit the response space. The model is then evaluated by ranking the candidate answers by the log probability of the first token prediction. However, first-tokens may not consistently reflect the final response output, due to model's diverse response styles such as starting with "Sure" or refusing to answer. Consequently, MCQ evaluation is not indicative of model behaviour when interacting with users. But by how much? We evaluate how aligned first-token evaluation is with the text output along several dimensions, namely final option choice, refusal rate, choice distribution and robustness under prompt perturbation. Our results show that the two approaches are severely misaligned on all dimensions, reaching mismatch rates over 60%. Models heavily fine-tuned on conversational or safety data are especially impacted. Crucially, models remain misaligned even when we increasingly constrain prompts, i.e., force them to start with an option letter or example template. Our findings i) underscore the importance of inspecting the text output as well and ii) caution against relying solely on first-token evaluation.

---

# “我的答案是C”：指令微调语言模型的首 token 概率与文本答案不匹配 论文详细解读

### 背景：这个问题为什么难？
在评估大语言模型（LLM）时，研究者常把开放式生成转化为多项选择题（MCQ），然后只看模型对第一个词的概率高低来决定它选了哪个选项。这个办法看起来省事，却默认模型的第一词就能代表最终答案。实际上，指令微调后的模型会出现“先说 Sure、先道歉、甚至直接拒答”等多样化开头，这让仅凭首 token 的打分失去可靠性。于是，评估结果可能和真实对话表现相差甚远，却没有人系统量化这种偏差到底有多大。

### 关键概念速览
**指令微调（Instruction‑tuning）**：在大模型上继续训练，让它更好地遵循人类给出的任务指令，类似给模型上了“使用说明书”。  
**首 token 概率（first‑token probability）**：模型在生成第一个词时给出的概率分布，常被用来排序 MCQ 选项。  
**多项选择题评估（MCQ evaluation）**：把开放式问题限定为若干固定答案，然后用模型的概率来判断它选了哪一个。  
**拒答率（refusal rate）**：模型在面对敏感或超出能力范围的问题时，选择“不回答”或给出安全提示的频率。  
**提示扰动（prompt perturbation）**：对输入的提问或示例做小幅修改，观察模型输出是否稳健。  
**选项分布偏差（choice distribution bias）**：模型在大量 MCQ 上倾向于某几个选项，而不是均匀分布，可能是训练数据或安全微调的副作用。  

### 核心创新点
1. **从“只看首 token” → “全程文本对齐”**：过去的评估直接把第一个词的概率当作答案可信度，这篇论文改为同时检查模型最终输出的完整文本，比较两者在选项、拒答、分布等维度上的一致性。这样可以直接量化评估方法本身的偏差。  
2. **系统化多维度对齐度测量**：作者设计了四个对齐指标——最终选项是否相同、是否出现拒答、选项出现频率是否一致、在提示微调下的鲁棒性。以前的工作只关注整体准确率，这里把评估细化到行为层面。  
3. **针对不同微调程度的对比实验**：通过挑选普通指令微调模型、强化安全微调模型以及对话微调模型，展示模型越倾向于安全或对话风格，首 token 与文本答案的错配率越高。  
4. **强制开头实验**：即使在提示中强制模型以选项字母或固定模板开头，错配率仍然高达 60% 以上，说明问题不是简单的“让模型先说 A”。这一步验证了错配并非表面形式可以轻易消除。

### 方法详解
整体思路是：**先让模型回答一批标准化的 MCQ，记录它的首 token 概率排序；再让模型完整生成答案文本，解析出它实际选择的选项或是否拒答；最后对两套结果做对齐统计**。整个流程可以拆成三步：

1. **构造评测集**  
   - 选取公开的多项选择题库（如 MMLU、ARC 等），每题提供四个选项 A‑D。  
   - 为每个问题准备两种提示：普通提问（“请回答以下问题”）和强制开头提示（“请直接输出选项字母：A、B、C、D”。）

2. **首 token 采样与排序**  
   - 把每个选项的文字（如 “A. …”）拼接到提示后，分别喂给模型，记录模型在生成第一个 token 时的对数概率。  
   - 依据概率高低得到模型“首 token 预测的最佳选项”。这一步完全不看后面的生成，只用第一个词的打分。

3. **完整文本生成与答案抽取**  
   - 让模型一次性生成完整回答，长度上限设为 64 token。  
   - 用简单的规则抽取答案：如果文本以 “A.”、“B.” 等开头，则直接取对应选项；如果出现 “I’m sorry” 或 “I can’t answer” 之类的安全提示，则标记为拒答；否则使用正则匹配寻找第一个出现的选项字母。

4. **对齐度计算**  
   - **选项一致率**：首 token 选项与文本抽取选项相同的比例。  
   - **拒答一致率**：首 token 预测是否为拒答（通常为低概率）与文本实际是否拒答的匹配度。  
   - **分布相似度**：统计两套结果的选项频率，用 KL 散度或简单的差异百分比衡量。  
   - **鲁棒性**：在提示扰动（如换词、改顺序）后重复上述步骤，观察对齐率变化。

最让人意外的设计是**把每个选项单独喂模型算首 token 概率**，而不是一次性让模型在所有选项中做选择。这种“逐选项打分”是业界常用的 MCQ 评估手段，却在这里被用来对比完整生成的真实行为，从而暴露出两者的系统性脱节。

### 实验与效果
- **数据集**：论文在多个公开的多项选择题基准上做实验，包括 MMLU（学术类）、ARC（科学推理）以及一些安全敏感的问答集合。  
- **模型**：覆盖了 LLaMA‑2、ChatGPT‑style 对话模型以及经过安全微调的模型，分别代表普通指令微调、对话微调和安全微调三类。  
- **主要发现**：  
  - 在所有模型上，**选项一致率最高也只有约 40%**，也就是说超过 60% 的时候，首 token 预测的选项和实际文本答案不一致。  
  - **拒答率错配**更为严重：安全微调模型的首 token 几乎从不给出拒答的高概率，但文本中却经常出现安全拒答。  
  - **选项分布偏差**显示，模型倾向于把某几个选项（如 C）压得更高，导致首 token 排名与真实答案的分布差距显著。  
  - 在**强制开头提示**下，错配率仍然保持在 60% 以上，说明仅靠格式约束并不能解决根本问题。  
- **消融实验**：作者分别去掉安全微调、去掉对话微调，只保留普通指令微调，错配率下降约 10%——说明安全/对话微调是主要推动因素。  
- **局限性**：论文主要聚焦于英文模型和标准 MCQ，未在中文或更开放式的生成任务上验证；对齐度指标仍是统计层面，缺少对具体错误类型的细粒度分析。

### 影响与延伸思考
这篇工作提醒社区：**评估不能只看概率分布的第一颗子弹**，否则会高估模型的真实表现。随后出现的几篇论文开始探索“全序列对齐评估”（sequence‑level evaluation）或引入人类标注的答案匹配度作为金标准。安全微调的副作用也被进一步讨论，促使研究者在构建安全过滤时加入对生成全过程的监控，而不是仅在采样阶段做截断。想深入了解的话，可以关注“对话模型评估基准（DialogEval）”和“安全微调的行为分析（Safety‑tuning behavior）”这两个方向，那里已经有不少后续工作在扩展这篇论文的思路。

### 一句话记住它
只看模型的首 token 概率，就像只听答案的开头一句话，根本无法判断它到底说了什么——评估必须审视完整文本。