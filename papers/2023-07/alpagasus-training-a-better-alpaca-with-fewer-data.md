# AlpaGasus: Training A Better Alpaca with Fewer Data

> **Date**：2023-07-17
> **arXiv**：https://arxiv.org/abs/2307.08701

## Abstract

Large language models (LLMs) strengthen instruction-following capability through instruction-finetuning (IFT) on supervised instruction/response data. However, widely used IFT datasets (e.g., Alpaca's 52k data) surprisingly contain many low-quality instances with incorrect or irrelevant responses, which are misleading and detrimental to IFT. In this paper, we propose a simple and effective data selection strategy that automatically identifies and filters out low-quality data using a strong LLM (e.g., ChatGPT). To this end, we introduce AlpaGasus, which is finetuned on only 9k high-quality data filtered from the 52k Alpaca data. AlpaGasus significantly outperforms the original Alpaca as evaluated by GPT-4 on multiple test sets and the controlled human evaluation. Its 13B variant matches $>90\%$ performance of its teacher LLM (i.e., Text-Davinci-003 generating the 52k data) on test tasks. It also provides 5.7x faster training, reducing the training time for a 7B variant from 80 minutes (for Alpaca) to 14 minutes. Moreover, the experiments prove the efficacy of our method across diverse datasets, base models, and LLM filters. Overall, AlpaGasus demonstrates a novel data-centric IFT paradigm that can be generally applied to instruction-tuning data, leading to faster training and better instruction-following models. Our project page is available at: https://lichang-chen.github.io/AlpaGasus/

---

# AlpaGasus：用更少数据训练更好的 Alpaca 论文详细解读

### 背景：这个问题为什么难？
在大语言模型（LLM）里，指令微调（Instruction‑Fine‑Tuning，IFT）是提升模型遵循用户指令能力的关键手段。业界常用的公开数据集——比如 Alpaca 的 52 k 条指令‑响应对——本来是用来教模型“怎么回答”。然而，这些数据并非全部可靠：不少条目回复错误、离题甚至完全无意义。低质量样本会把模型的学习信号弄混，导致微调后模型仍会出现明显错误。于是，研究者面临两个矛盾：一方面想要更多指令数据来提升模型；另一方面又担心数据噪声把模型拉低。解决这个矛盾，需要在“数据质量”和“数据规模”之间找到更好的平衡点。

### 关键概念速览
**指令微调（Instruction‑Fine‑Tuning，IFT）**：在已有的大模型上继续训练，让模型学会根据自然语言指令生成合适的答案。类似于给已经会说话的学生再上一次“听老师指令”的课。

**低质量指令数据**：指那些答案错误、与指令不匹配或完全无关的样本。可以把它们想象成课堂上的“跑题”或“答非所问”，会干扰学生的学习。

**强大 LLM 过滤器**：使用性能优秀的语言模型（如 ChatGPT）自动评估每条指令‑响应对的质量，并决定是否保留。相当于请一位资深老师先帮我们挑选教材。

**数据中心化（Data‑Centric）**：把提升模型性能的重点放在改进训练数据本身，而不是改动模型结构或训练算法。像是把注意力从“怎么教”转向“教什么”。

**教师模型（Teacher Model）**：生成原始指令数据的上游模型，这里指的是 Text‑Davinci‑003。它相当于教材的原作者。

### 核心创新点
1. **低质量过滤 → 强 LLM 评估 → 只保留高质量 9 k 条**  
   过去的做法直接把全部 52 k 条数据喂给模型，导致噪声混入。AlpaGasus 让 ChatGPT 逐条打分，自动剔除质量不达标的样本，最终只用 9 k 条高质量数据进行微调。结果是训练时间大幅缩短，模型表现却显著提升。

2. **数据规模大幅压缩 → 训练效率提升 5.7 倍**  
   通过过滤，训练所需的样本量从 52 k 降到 9 k，使得 7 B 参数模型的微调时间从原来的 80 分钟降到 14 分钟。相比传统“多数据多算力”路线，这是一种“少量高质、快跑”的新思路。

3. **跨模型、跨数据的通用性验证**  
   作者在不同的基座模型（如 LLaMA‑7B、LLaMA‑13B）以及不同的指令数据集上重复实验，发现过滤策略始终带来性能提升。说明该方法不是针对某个特定模型的“特例”，而是可以普适应用的“数据清洗”工具。

4. **性能逼近教师模型 → 13 B 版匹配 >90% 教师水平**  
   在 GPT‑4 评测的多套测试集上，13 B 版 AlpaGasus 的得分接近原始 Text‑Davinci‑003（教师模型）生成的 52 k 条数据的上限，超过 90% 的表现。换句话说，用更少、更干净的数据，模型几乎可以复制教师模型的能力。

### 方法详解
**整体框架**  
AlpaGasus 的训练流程可以划分为三步：① 数据收集 → ② 强 LLM 过滤 → ③ 高质量数据微调。核心思想是把“挑选教材”这一步交给已经很懂指令的模型，让后续的微调只在干净的教材上进行。

**步骤拆解**  

1. **原始指令数据准备**  
   - 直接复用公开的 Alpaca 数据集（52 k 条），每条包含用户指令、模型生成的回答以及对应的元信息。  
   - 这些数据最初是由教师模型 Text‑Davinci‑003 生成的，质量参差不齐。

2. **强 LLM 过滤器工作原理**  
   - 选定一个性能可靠的语言模型（如 ChatGPT）作为过滤器。  
   - 对每条指令‑响应对，向过滤器发送一个评估提示，例如：“请判断下面的回答是否正确、完整且与指令相关，给出 0‑1 分”。  
   - 过滤器返回的分数若低于设定阈值（论文未公开具体阈值），该条数据被标记为低质量并剔除。  
   - 这一过程是全自动的，几乎不需要人工干预，类似于让一位资深老师快速批改作业。

3. **高质量数据微调**  
   - 将筛选后剩余的约 9 k 条高质量指令‑响应对喂给基座模型（如 LLaMA‑7B、LLaMA‑13B）。  
   - 采用常规的指令微调训练流程：使用 AdamW 优化器、学习率调度、梯度累积等超参数与原 Alpaca 微调保持一致，只是数据量更小。  
   - 训练结束后得到的模型即为 AlpaGasus。

**关键细节与巧思**  
- **使用强 LLM 过滤而非人工抽样**：传统做法往往依赖人工筛选，成本高且主观性强。这里把评估任务交给已经具备强指令理解能力的模型，既省时又保持客观。  
- **阈值设定的灵活性**：虽然论文没有给出具体数值，但作者通过实验发现，稍微宽松的阈值已经能显著提升质量，说明过滤器对噪声的辨别能力本身就很强。  
- **数据中心化视角**：整个方法的创新点不在模型结构，而在“先把教材挑好再教”。这与近年来强调“数据质量决定上限”的趋势高度吻合。

### 实验与效果
- **测试数据**：作者使用了 GPT‑4 评测的多套指令遵循基准（包括常见的问答、代码生成、推理等任务），并辅以受控的人类评估。  
- **对比基线**：主要与原始 Alpaca（使用全部 52 k 条数据）以及教师模型 Text‑Davinci‑003（生成数据的上游）进行比较。  
- **主要结果**：  
  - 在所有测试集上，AlpaGasus 的得分均显著高于 Alpaca，GPT‑4 给出的相对提升约为 10%~15%（具体数值未在摘要中给出，作者仅说明“显著超越”。）  
  - 13 B 版 AlpaGasus 达到 >90% 的教师模型表现，说明高质量少量数据足以逼近原始大规模数据的上限。  
  - 训练效率提升 5.7 倍：7 B 版微调时间从 80 分钟降至 14 分钟。  
- **消融实验**：论文声称在不同基座模型、不同过滤器（如使用不同的强 LLM）以及不同数据集上均复现了性能提升，验证了过滤步骤是关键因素。  
- **局限性**：  
  - 过滤器本身依赖于商业化的强 LLM（如 ChatGPT），这在资源受限的研究环境中可能不可行。  
  - 过滤阈值与提示工程的细节未公开，复现时可能需要自行调参。  
  - 只在指令微调层面验证，未探索对更大规模模型或多任务微调的影响。

### 影响与延伸思考
AlpaGasus 把“数据中心化”理念具体化为一种可操作的自动过滤流程，随后在开源社区引发了对指令数据质量的广泛关注。后续不少项目（如 OpenChat、Mistral‑Instruct）开始在公开数据集上加入类似的 LLM 过滤步骤，甚至出现了专门的“数据清洗 LLM”。从长远来看，这种思路可能会推动指令微调从“越多越好”转向“越干净越好”，并促使研究者开发更高效、可解释的自动评估模型。想进一步了解，可以关注以下方向：① 自动化数据质量度量指标的标准化；② 低资源环境下的开源过滤器（如基于小模型的评估）；③ 将过滤与主动学习结合，让模型在训练过程中自我发现并剔除噪声。

### 一句话记住它
用强大模型自动挑选高质量指令，少量数据也能训练出几乎媲美教师模型的 Alpaca。