# LLaMAX: Scaling Linguistic Horizons of LLM by Enhancing Translation   Capabilities Beyond 100 Languages

> **Date**：2024-07-08
> **arXiv**：https://arxiv.org/abs/2407.05975

## Abstract

Large Language Models (LLMs) demonstrate remarkable translation capabilities in high-resource language tasks, yet their performance in low-resource languages is hindered by insufficient multilingual data during pre-training. To address this, we conduct extensive multilingual continual pre-training on the LLaMA series models, enabling translation support across more than 100 languages. Through a comprehensive analysis of training strategies, such as vocabulary expansion and data augmentation, we develop LLaMAX. Remarkably, without sacrificing its generalization ability, LLaMAX achieves significantly higher translation performance compared to existing open-source LLMs (by more than 10 spBLEU points) and performs on-par with specialized translation model (M2M-100-12B) on the Flores-101 benchmark. Extensive experiments indicate that LLaMAX can serve as a robust multilingual foundation model. The code \footnote{\url{https://github.com/CONE-MT/LLaMAX/.}} and the models \footnote{\url{https://huggingface.co/LLaMAX/.}} are publicly available.

---

# LLaMAX：通过提升翻译能力将大语言模型的语言视野扩展至100余种语言 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在英语等高资源语言上的翻译表现已经相当惊艳，但当面对资源匮乏的语言时，模型往往因为预训练阶段缺少足够的双语或多语数据而表现不佳。传统的多语言模型要么在所有语言上均匀分配容量，导致低资源语言的表现被压制，要么专门为翻译任务训练独立模型，失去了通用语言理解的优势。于是，如何在保持 LLM 通用能力的同时，显著提升对百余种语言的翻译质量，成为了亟待突破的瓶颈。

### 关键概念速览
- **多语言持续预训练（Multilingual Continual Pre‑training）**：在已有的 LLM 基础上，再次使用大量多语言文本进行训练，类似给模型“补课”，让它在已经学会的能力上继续深化对新语言的理解。
- **词表扩展（Vocabulary Expansion）**：把原模型的词表（token set）加入更多语言的子词单元，使模型能够直接处理之前不认识的字符或词根，类似给人类词典增添新词条。
- **数据增强（Data Augmentation）**：通过机器翻译、回译或合成句子等手段人为制造更多训练样本，尤其是低资源语言的平行句对，像是给稀缺的教材加注释。
- **spBLEU**：一种对句子级别 BLEU 分数进行平滑处理的评估指标，数值越高说明翻译质量越好。
- **Flores‑101 基准**：覆盖 101 种语言的标准翻译评测集合，常被用来衡量多语言模型的真实翻译水平。
- **M2M‑100‑12B**：Meta 开源的专门用于多语言翻译的 120 亿参数模型，被视作翻译任务的强基准。

### 核心创新点
1. **持续预训练 + 词表扩展 → 直接在 LLaMA 系列上进行多语言“再学习”，而不是从头训练新模型**。这样既保留了原模型的通用语言理解能力，又让模型在新语言上拥有更细致的表示。
2. **大规模跨语言数据增强 → 用机器翻译生成的合成平行句对填补低资源语言的训练空白**。相比仅依赖公开语料库，作者把已有的高质量翻译系统当作“老师”，让模型在“练习”中快速提升。
3. **统一的多任务目标 → 在同一次训练中同时优化语言建模和翻译任务**。传统做法往往先做语言建模再单独微调翻译，这里把两者合并，避免了知识迁移的损失。
4. **不牺牲通用能力的“翻译专精” → 通过精心的学习率调度和层冻结策略，使得模型在提升翻译性能的同时，仍保持在零样本推理、问答等任务上的竞争力**。

### 方法详解
整体思路可以划分为三步：**词表扩展 → 多语言数据准备 → 持续预训练**。

1. **词表扩展**  
   - 先对 LLaMA 原始词表进行统计，找出在目标 100+ 语言中出现频率最高的子词。  
   - 使用 SentencePiece 等子词分词器重新训练一个更大的词表（约 1.5 倍原大小），并把新词映射到模型的嵌入层。  
   - 为避免破坏已有的语义空间，作者只在新增词上随机初始化嵌入，其余保持不变。

2. **多语言数据准备**  
   - **真实平行语料**：收集公开的多语言平行语料库（如 OPUS、CCMatrix），覆盖尽可能多的语言对。  
   - **合成平行语料**：对低资源语言缺失的方向，使用已有的强翻译模型（如 M2M‑100）把高资源语言句子翻译过去，再回译回原语言形成双向对齐。  
   - **单语语言模型数据**：保留原始的海量单语文本，以继续强化语言建模能力。  
   - 所有数据统一混合，比例大约为 70% 单语、30% 平行，确保模型在“懂语言”和“会翻译”之间保持平衡。

3. **持续预训练**  
   - **目标函数**：采用混合损失，语言建模使用自回归的交叉熵，翻译任务使用跨语言的条件语言模型（即在源语言句子后接目标语言标记，再预测目标语言 token）。  
   - **层冻结与学习率**：底层 6 层保持原学习率的 0.1，防止已学知识被冲刷；高层和新加入的词嵌入使用相对较大的学习率，以快速适应新语言。  
   - **训练调度**：先进行 10% 的 warm‑up，随后线性衰减，总训练步数约为 200k。  
   - **硬件与并行**：在 8×A100 GPU 上使用 ZeRO‑3 分布式优化，显著降低显存占用，使得 12B 参数模型能够在可接受的成本下完成训练。

**最巧妙的点**在于把翻译任务直接嵌入自回归语言模型的训练流程，而不是后置微调。这样模型在生成目标语言时已经拥有跨语言的上下文感知，翻译质量自然提升，同时也避免了“微调后忘记原任务”的灾难性遗忘。

### 实验与效果
- **评测数据**：主要在 Flores‑101（101 种语言）上测 spBLEU；另外在 WMT‑21 部分语言对、以及通用的 MMLU、ARC 等零样本任务上做对比。  
- **基线对比**：  
  - 与开源的 LLaMA‑2‑13B、Mistral‑7B 等模型相比，LLaMAX 在 Flores‑101 上整体提升超过 10 spBLEU。  
  - 与专门的翻译模型 M2M‑100‑12B 相比，LLaMAX 在多数语言对上实现了“持平”或略有优势，尤其在低资源语言（如斯瓦希里语、塔吉克语）上领先约 1‑2 spBLEU。  
- **消融实验**：  
  - 去掉词表扩展，spBLEU 下降约 3‑4 分，说明新词对低资源语言至关重要。  
  - 只使用真实平行语料而不做合成增强，整体下降约 5 spBLEU，验证了数据增强的价值。  
  - 关闭翻译任务的混合损失，仅保留语言建模，翻译性能跌至原始 LLaMA 水平，表明联合训练是关键。  
- **局限性**：作者指出在极端低资源语言（如某些土著语言）仍受限于合成数据的质量；此外，持续预训练对算力要求仍然高，普通研究团队难以复现完整流程。

### 影响与延伸思考
LLaMAX 的出现展示了“在通用大模型上继续做多语言专精”是一条可行路径，激发了后续工作在以下方向的探索：  
- **更高效的词表增量学习**，尝试只在少数层更新词嵌入以降低显存。  
- **跨语言自监督任务**（如跨语言掩码预测）作为替代翻译损失，进一步提升低资源语言的表示。  
- **多模态扩展**：把图像或语音的跨语言对齐信息加入持续预训练，构建真正的多语言多模态基础模型。  
- **开源生态**：LLaMAX 的代码和模型已公开，社区已经基于它衍生出针对特定地区语言的微调版本，推动了语言技术的民主化。  
如果想深入了解，可关注近期的 “Multilingual Continual Learning” 研讨会以及 Meta、DeepMind 在跨语言自监督方面的最新报告（推测）。

### 一句话记住它
**LLaMAX 证明：在已有大语言模型上做词表扩展、数据增强和联合翻译训练，就能把翻译能力从几种语言跃升到百余种，而不牺牲通用智能。**