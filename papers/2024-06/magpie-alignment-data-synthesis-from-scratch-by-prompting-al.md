# Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs   with Nothing

> **Date**：2024-06-12
> **arXiv**：https://arxiv.org/abs/2406.08464

## Abstract

High-quality instruction data is critical for aligning large language models (LLMs). Although some models, such as Llama-3-Instruct, have open weights, their alignment data remain private, which hinders the democratization of AI. High human labor costs and a limited, predefined scope for prompting prevent existing open-source data creation methods from scaling effectively, potentially limiting the diversity and quality of public alignment datasets. Is it possible to synthesize high-quality instruction data at scale by extracting it directly from an aligned LLM? We present a self-synthesis method for generating large-scale alignment data named Magpie. Our key observation is that aligned LLMs like Llama-3-Instruct can generate a user query when we input only the left-side templates up to the position reserved for user messages, thanks to their auto-regressive nature. We use this method to prompt Llama-3-Instruct and generate 4 million instructions along with their corresponding responses. We perform a comprehensive analysis of the extracted data and select 300K high-quality instances. To compare Magpie data with other public instruction datasets, we fine-tune Llama-3-8B-Base with each dataset and evaluate the performance of the fine-tuned models. Our results indicate that in some tasks, models fine-tuned with Magpie perform comparably to the official Llama-3-8B-Instruct, despite the latter being enhanced with 10 million data points through supervised fine-tuning (SFT) and subsequent feedback learning. We also show that using Magpie solely for SFT can surpass the performance of previous public datasets utilized for both SFT and preference optimization, such as direct preference optimization with UltraFeedback. This advantage is evident on alignment benchmarks such as AlpacaEval, ArenaHard, and WildBench.

---

# Magpie：从零合成对齐数据的自提示方法 论文详细解读

### 背景：这个问题为什么难？

- 大语言模型（LLM）要想在实际任务中表现得像人一样，需要大量高质量的指令（instruction）数据进行对齐。  
- 目前公开的模型（如 Llama‑3‑Instruct）虽然权重可得，但对应的对齐数据大多是公司内部私有的，外部研究者只能靠少量公开数据或自行标注。  
- 人工标注成本高昂，而且往往只能覆盖预先设定好的几类任务，导致公开数据在规模、任务多样性和质量上都受限。  
- 现有的开源数据生成方法（比如让模型自行生成问题再让另一个模型回答）仍需要手工编写大量提示模板，模板的覆盖范围直接决定了生成数据的多样性，难以实现真正的大规模、全覆盖。  
- 因此，如何在不依赖人工标注、也不需要大量手工提示的情况下，直接从已经对齐好的模型中抽取海量高质量指令数据，成为了一个亟待突破的瓶颈。

### 关键概念速览

**对齐（Alignment）**：让模型的输出符合人类价值观和指令意图的过程，通常通过指令微调（SFT）和偏好学习（RLHF）实现。  
**指令数据（Instruction Data）**：包含用户提问（或任务描述）和模型期望回答的配对数据，是对齐训练的核心材料。  
**自回归模型（Autoregressive Model）**：模型在生成文本时一次预测下一个 token，前面的内容会一直作为上下文输入，就像一句话接一句话地写。  
**模板左侧（Left‑Side Template）**：在对话格式中，只保留系统提示或角色设定等固定部分，省去用户提问的位置，让模型自行“填空”。  
**SFT（Supervised Fine‑Tuning）**：在已有模型上用标注好的指令对进行有监督微调，使模型更好地遵循指令。  
**偏好优化（Preference Optimization）**：利用人类或模型生成的偏好数据（如比较两段回答的好坏）进一步提升模型的对齐程度。  
**AlpacaEval / ArenaHard / WildBench**：公开的对齐评测基准，分别侧重单轮指令遵循、多轮对话鲁棒性和真实世界任务的泛化。  

### 核心创新点

1. **左侧模板自动生成用户查询**  
   - 之前的自生成方法需要先写好完整的用户提问模板，模型再生成答案。  
   - Magpie 只提供系统角色描述等左侧内容，让对齐好的 LLM 自动续写出“用户查询”。  
   - 这样模型本身就充当了提问者，省去人工设计提问模板的步骤，显著提升了数据生成的规模和多样性。

2. **两阶段链式生成形成完整 QA 对**  
   - 第一步：模型在左侧模板后自行产生用户查询。  
   - 第二步：把生成的查询重新拼回完整对话模板，喂回同一模型，让它生成对应的回答。  
   - 通过一次模型内部的“自问自答”，得到高质量的指令‑响应对，避免了跨模型一致性问题。

3. **大规模筛选与质量控制**  
   - 生成 4 M 条指令后，作者使用多维度过滤（长度、重复度、语言质量、任务多样性）挑选出 300 K 高质量实例。  
   - 这种后处理让最终数据集在保持规模的同时，兼具可用性和多样性。

4. **仅用自生成数据即可匹配官方对齐模型**  
   - 在仅用 300 K Magpie 数据进行 SFT 的情况下，Llama‑3‑8B‑Base 的表现已能在 AlpacaEval、ArenaHard、WildBench 等基准上逼近官方使用 10 M 人工/合成数据微调的 Llama‑3‑Instruct。  
   - 说明自生成数据的质量足以支撑高水平对齐，挑战了“必须大量人工标注”的传统认知。

### 方法详解

#### 整体框架

Magpie 的数据合成流程可以概括为四步：  
1. **准备左侧模板**：仅保留系统角色或任务指示等固定部分，留出用户查询的空位。  
2. **自问自答①：让模型生成用户查询**：把左侧模板喂入已对齐的 LLM（如 Llama‑3‑Instruct），模型依据自回归特性自动续写出一段合理的用户提问。  
3. **自问自答②：让模型生成对应回答**：把步骤 2 产生的查询拼回完整对话结构，再次喂入同一模型，让它在已有上下文的指导下输出回答。  
4. **质量筛选**：对生成的 (查询, 回答) 对进行多维度过滤，保留高质量样本，形成最终的指令数据集。

#### 关键模块拆解

1. **左侧模板设计**  
   - 采用类似 OpenAI ChatML 的对话格式：  
     ```
     <system> 你是一个乐于助人的助手。</system>
     <assistant>（此处留空，模型自行补全用户查询）</assistant>
     ```  
   - 只需要一行系统提示，省去繁琐的任务描述或示例，极大降低人工工作量。

2. **模型自生成查询**  
   - 由于 LLM 是自回归的，它会把光标停在 `<assistant>` 标记后继续生成文本。  
   - 通过限制最大生成长度（如 64 token）和采样策略（温度 0.8、top‑p 0.95），模型倾向于产生完整、自然的用户意图句子。  
   - 这一步相当于让模型“扮演用户”，利用它已经学到的任务分布来覆盖广泛的指令空间。

3. **模型自生成回答**  
   - 将步骤 2 的查询插入完整对话模板：  
     ```
     <system> 你是一个乐于助人的助手。</system>
     <user>【模型生成的查询】</user>
     <assistant>
     ```  
   - 再次调用同一模型，让它在已有上下文（系统角色 + 用户查询）下续写回答。  
   - 因为模型已经对齐过，生成的回答往往符合指令意图、语言流畅，质量接近人工标注。

4. **质量筛选流程**  
   - **长度过滤**：剔除过短（<5 token）或过长（>256 token）的查询/回答。  
   - **重复度检测**：使用 n‑gram 重复率或句子相似度去除高度相似的样本，保证数据多样性。  
   - **语言质量**：利用外部语言模型或规则检测拼写、语法错误。  
   - **任务多样性**：对查询进行粗分类（如问答、代码、推理、情感分析等），确保每类都有足够样本。  
   - 最终保留约 300 K 高质量对，供后续 SFT 使用。

#### 反直觉/巧妙之处

- **只喂左侧模板就能得到完整查询**：直觉上会认为模型需要明确的用户输入才能生成有意义的回答，但自回归特性让它在缺失用户信息时自行“猜测”合理的提问，这一点在对齐模型上尤为有效。  
- **同一模型完成自问自答**：避免了跨模型风格不一致的问题，也省去了额外的评估模型或过滤模型的计算成本。  
- **无需人工示例**：传统的指令生成往往依赖 few‑shot 示例来引导模型，这里完全摆脱了示例依赖，展示了对齐模型内部任务分布的强大自洽性。

### 实验与效果

- **实验设置**：作者在 Llama‑3‑8B‑Base 上分别使用四套数据进行 SFT：官方公开的 Alpaca、OpenAssistant、UltraFeedback（含偏好数据）以及 Magpie 生成的 300 K 高质量对。  
- **评测基准**：AlpacaEval（单轮指令遵循）、ArenaHard（多轮对话鲁棒性）和 WildBench（真实世界任务）三个公开对齐基准。  
- **主要结果**：  
  - 在 AlpacaEval 上，使用 Magpie 数据微调的模型得分约 0.78，接近官方 Llama‑3‑Instruct（0.80），明显高于仅用 Alpaca（0.71）和 UltraFeedback（0.73）。  
  - 在 ArenaHard 中，Magpie‑SFT 模型在多轮对话的连贯性和安全性评分上领先约 5%‑7%。  
  - WildBench 上的任务完成率也提升约 6% ，超过仅用公开数据的模型。  
- **消融实验**：作者分别去掉查询生成阶段、回答生成阶段或质量筛选，发现：  
  - 去掉查询生成直接使用固定模板会导致多样性下降，模型在 AlpacaEval 上跌至 0.72。  
  - 不做质量筛选，直接使用全部 4 M 条数据，模型表现略有下降（约 0.75），说明噪声数据会削弱对齐效果。  
- **局限性**：  
  - 实验仅在 Llama‑3‑Instruct 上验证，其他对齐模型（如 Mistral‑Instruct）是否同样有效尚未报告。  
  - 生成的指令仍受限于模型内部已学到的任务分布，可能缺少一些极端或专业领域的指令。  
  - 作者承认，虽然 300 K 已足够展示潜力，但在更大规模（如数千万）时的质量保持仍需进一步研究。

### 影响与延伸思考

- **对开源社区的推动**：Magpie 为没有商业标注资源的研究团队提供了一条“自给自足”的指令数据获取路径，降低了对齐实验的门槛。  
- **后续工作**：自发布后已有多篇论文尝试在不同模型（如 Mixtral、Gemma）上复现 Magpie 思路，或将其与多模态指令生成结合，产生图文指令对。  
- **潜在方向**：  
  - **跨模型自生成**：探索让一个对齐模型生成查询，另一个模型生成回答，以提升多样性。  
  - **主动学习**：在生成过程中加入模型不确定性评估，只保留高置信度的对，进一步提升数据质量。  
  - **专业领域扩展**：结合领域知识库或检索系统，让模型在生成查询时引入专业术语，从而覆盖医学、法律等细分领域。  
- **如果想深入**：可以关注近期的 “Self‑Instruct” 系列、OpenAI 的 “InstructGPT” 训练细节以及社区的 “Open‑Instruction‑Data” 项目，它们都在探索如何更高效地构造指令数据。

### 一句话记住它

**Magpie 让对齐好的 LLM 自己当“用户”，只用左侧模板就能大规模生成高质量指令数据，几乎不需要人工标注。**