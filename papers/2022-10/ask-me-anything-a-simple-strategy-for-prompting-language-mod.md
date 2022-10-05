# Ask Me Anything: A simple strategy for prompting language models

> **Date**：2022-10-05
> **arXiv**：https://arxiv.org/abs/2210.02441

## Abstract

Large language models (LLMs) transfer well to new tasks out-of-the-box simply given a natural language prompt that demonstrates how to perform the task and no additional training. Prompting is a brittle process wherein small modifications to the prompt can cause large variations in the model predictions, and therefore significant effort is dedicated towards designing a painstakingly "perfect prompt" for a task. To mitigate the high degree of effort involved in prompt-design, we instead ask whether producing multiple effective, yet imperfect, prompts and aggregating them can lead to a high quality prompting strategy. Our observations motivate our proposed prompting method, ASK ME ANYTHING (AMA). We first develop an understanding of the effective prompt formats, finding that question-answering (QA) prompts, which encourage open-ended generation ("Who went to the park?") tend to outperform those that restrict the model outputs ("John went to the park. Output True or False."). Our approach recursively uses the LLM itself to transform task inputs to the effective QA format. We apply the collected prompts to obtain several noisy votes for the input's true label. We find that the prompts can have very different accuracies and complex dependencies and thus propose to use weak supervision, a procedure for combining the noisy predictions, to produce the final predictions for the inputs. We evaluate AMA across open-source model families (e.g., EleutherAI, BLOOM, OPT, and T0) and model sizes (125M-175B parameters), demonstrating an average performance lift of 10.2% over the few-shot baseline. This simple strategy enables the open-source GPT-J-6B model to match and exceed the performance of few-shot GPT3-175B on 15 of 20 popular benchmarks. Averaged across these tasks, the GPT-J-6B model outperforms few-shot GPT3-175B. We release our code here: https://github.com/HazyResearch/ama_prompting

---

# 随便问我：一种简易的语言模型提示策略 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）在没有额外微调的情况下，只要给出一个自然语言的示例提示，就能完成新任务，这听起来很理想。但实际操作时，提示词（prompt）的微小改动往往会导致模型输出天差地别，导致研究者和工程师花费大量时间去“调参”。传统做法是手工寻找一个“完美”提示，却没有系统的方法评估或利用多个次优提示的价值。因此，如何降低提示设计的成本、提升提示的鲁棒性，成为阻碍 LLM 大规模落地的关键瓶颈。

### 关键概念速览
- **Few‑shot Prompt（少样本提示）**：在提示中给出少量任务示例，让模型直接推理。类似于老师只给学生几道例题，学生据此完成新题。
- **QA Prompt（问答式提示）**：把任务包装成开放式问答，例如“谁去了公园？”而不是强制模型输出“True/False”。相当于让模型自由发挥，而不是只给出是非二选一。
- **Weak Supervision（弱监督）**：利用多个噪声较大的标签来源，通过统计模型合成更可靠的标签。好比把几个人的意见综合，得到更可信的结论。
- **Noisy Vote（噪声投票）**：每个提示产生的预测被视为一次投票，虽然单票可能错误，但多数票或加权票往往更准。
- **Prompt Transformation（提示转换）**：使用 LLM 本身把原始任务输入改写成 QA 形式的提示。相当于让模型自己把难懂的题目翻译成自己擅长的语言。
- **Open‑source LLM（开源大模型）**：如 EleutherAI、BLOOM、OPT、T0、GPT‑J 等，参数规模从 125M 到 175B 不等，提供了实验的多样性。

### 核心创新点
1. **从单一完美提示到多提示聚合**  
   - 之前：大多数工作只追求一个精心设计的提示，耗时且不稳。  
   - 本文：主动生成一批“有效但不完美”的提示，然后把它们的预测合并。  
   - 改变：把提示设计从“找唯一最佳”转变为“收集多样意见”，显著提升鲁棒性。

2. **发现 QA 提示的普适优势**  
   - 之前：很多提示采用限制输出的格式（如“输出 True/False”），导致模型受限。  
   - 本文：系统实验表明，开放式问答式提示在多数任务上表现更好。  
   - 改变：提供了一个易于复制的提示模板原则，帮助后续工作快速选取高效提示形式。

3. **让模型自我生成 QA 提示**  
   - 之前：提示往往需要人工手工改写成 QA 形式。  
   - 本文：递归调用同一个 LLM，把原始输入自动转化为 QA 提示。  
   - 改变：大幅降低人工改写成本，使得多提示生成可以规模化。

4. **引入弱监督进行噪声投票融合**  
   - 之前：多数工作直接多数表决或加权平均，忽视不同提示之间的依赖性。  
   - 本文：使用弱监督框架（如 Snorkel）建模提示之间的准确率和相关性，得到更可靠的最终标签。  
   - 改变：在提示质量差异大、相互依赖强的情况下仍能稳健提升性能。

### 方法详解
**整体思路**：先让 LLM 自动把每条任务输入改写成若干 QA 提示；对每个 QA 提示让模型生成答案；把所有答案视为噪声投票；最后用弱监督模型把这些投票融合成最终预测。

**步骤拆解**：

1. **输入→多样 QA 提示**  
   - 给定原始任务（如文本分类），构造一个“转换提示”，让 LLM 输出同一输入的不同问法。例如，原句“这篇评论是正面的”，转换为“这篇评论的情感是积极还是消极？”、“这篇评论表达了什么情绪？”等。  
   - 通过在转换提示中加入随机化指令（如“换一种说法”），一次调用可得到多个变体。

2. **每个 QA 提示 → 模型预测**  
   - 对每个生成的 QA 提示，使用同一 LLM（或不同规模的模型）进行 few‑shot 推理，得到答案字符串。答案可能是直接的标签（“积极”）或更自由的文本，需要后处理映射到任务标签空间。

3. **噪声投票收集**  
   - 将每个提示的答案视为一次投票。因为提示质量不一，投票本身噪声较大，单纯多数表决可能不理想。

4. **弱监督融合**  
   - 为每个提示构建一个“标签生成函数”，记录它在训练集上的表现（准确率、误报率）。  
   - 使用弱监督的生成模型（如基于概率图模型的标签模型）学习每个提示的可靠度以及提示之间的相关性。  
   - 通过该模型对测试集的投票进行加权，输出最终标签。

**关键技巧**：

- **递归使用 LLM**：提示转换本身也是一次 LLM 推理，形成“模型自助改写”闭环，省去人工工程。  
- **弱监督的依赖建模**：不像简单加权平均，弱监督能够捕捉到两个提示经常一起出错的情况，避免错误放大。  
- **跨模型、跨规模实验**：作者在不同开源模型上复现同一流程，证明方法不依赖特定模型结构。

### 实验与效果
- **评测任务**：覆盖 20 项常用基准，包括情感分析、自然语言推理、问答、文本分类等。  
- **模型阵容**：EleutherAI、BLOOM、OPT、T0 系列以及 GPT‑J‑6B，规模从 125M 到 175B 参数。  
- **基线**：标准 few‑shot 提示（单一手工设计的提示）以及 GPT‑3‑175B few‑shot（官方基准）。  
- **主要结果**：平均提升 10.2% 相比 few‑shot 基线；在 15/20 任务上，开源的 GPT‑J‑6B（6B 参数）匹配或超越 GPT‑3‑175B（175B 参数）的 few‑shot 表现。整体来看，AMA 让小模型的性价比大幅提升。  
- **消融实验**：作者分别去掉（1）提示转换步骤、（2）弱监督融合、（3）仅使用 QA 提示，发现每一环节都贡献显著提升，尤其是弱监督去除后性能跌回到单提示水平。  
- **局限性**：方法依赖于 LLM 能够生成高质量的 QA 提示；在极端长文本或高度结构化任务上提示转换的成功率下降；弱监督模型训练需要一定的标注数据来估计提示准确率，完全零标注情形尚未验证。

### 影响与延伸思考
- **社区反响**：AMA 的代码开源后，被多篇后续工作引用，用于提升小模型在零样本/少样本设置下的表现。  
- **后续方向**：  
  1. **自动化提示质量评估**：探索无需标注数据的自监督方式估计每个提示的可靠度。  
  2. **跨模态提示聚合**：把文本、图像、音频等不同模态的提示一起纳入弱监督框架。  
  3. **提示搜索空间的理论分析**：研究为什么 QA 提示普遍更好，是否与模型的训练目标（自回归生成）本质匹配。  
- **值得关注的工作**：后续出现的 “Self‑Consistency” 以及 “Chain‑of‑Thought Prompting” 等，都在不同角度利用多提示或多推理路径提升鲁棒性，和 AMA 的思路相呼应。

### 一句话记住它
把任务改写成多个问答式提示，让模型自己投票，再用弱监督把噪声票合成答案，简单又能让小模型跑赢大模型的 few‑shot。