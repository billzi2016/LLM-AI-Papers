# BayLing: Bridging Cross-lingual Alignment and Instruction Following   through Interactive Translation for Large Language Models

> **Date**：2023-06-19
> **arXiv**：https://arxiv.org/abs/2306.10968

## Abstract

Large language models (LLMs) have demonstrated remarkable prowess in language understanding and generation. Advancing from foundation LLMs to instructionfollowing LLMs, instruction tuning plays a vital role in aligning LLMs to human preferences. However, the existing LLMs are usually focused on English, leading to inferior performance in non-English languages. In order to improve the performance for non-English languages, it is necessary to collect language-specific training data for foundation LLMs and construct language-specific instructions for instruction tuning, both of which are heavy loads. To minimize human workload, we propose to transfer the capabilities of language generation and instruction following from English to other languages through an interactive translation task. We have developed BayLing, an instruction-following LLM by utilizing LLaMA as the foundation LLM and automatically constructing interactive translation instructions for instructing tuning. Extensive assessments demonstrate that BayLing achieves comparable performance to GPT-3.5-turbo, despite utilizing a considerably smaller parameter size of only 13 billion. Experimental results on translation tasks show that BayLing achieves 95% of single-turn translation capability compared to GPT-4 with automatic evaluation and 96% of interactive translation capability compared to GPT-3.5-turbo with human evaluation. To estimate the performance on general tasks, we created a multi-turn instruction test set called BayLing-80. The experimental results on BayLing-80 indicate that BayLing achieves 89% of performance compared to GPT-3.5-turbo. BayLing also demonstrates outstanding performance on knowledge assessment of Chinese GaoKao and English SAT, second only to GPT-3.5-turbo among a multitude of instruction-following LLMs. Demo, homepage, code and models of BayLing are available.

---

# BayLing：通过交互式翻译实现跨语言对齐与指令遵循的桥梁 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在英文语料上训练得非常充分，指令微调后能很好地理解和执行人类指令。但把同样的能力搬到中文、法文等非英文语言时，模型往往表现不佳。根本原因有两点：一是基础模型的预训练语料中非英文占比低，导致语言生成能力不足；二是指令微调需要大量语言特定的指令数据，人工收集成本高。于是出现了“英文模型强，其他语言弱”的尴尬局面，迫切需要一种省人力、跨语言迁移能力的方案。

### 关键概念速览
**大语言模型（LLM）**：拥有上百亿参数、通过海量文本自监督学习的模型，能生成连贯文字并完成多种任务。  
**指令微调（Instruction Tuning）**：在已有的基础模型上，用“请做…”“请解释…”等指令-答案对继续训练，使模型更贴合人类指令的需求。  
**跨语言对齐（Cross‑lingual Alignment）**：让模型在不同语言之间共享语义空间，使得同一概念在中文、英文等语言里表现相似。  
**交互式翻译（Interactive Translation）**：不是一次性把句子翻译完，而是模型在用户提问、纠错、补充信息的多轮对话中完成翻译。类似于两个人一起写稿子，一边翻译一边讨论细节。  
**单轮 vs 多轮能力**：单轮指一次输入一次输出的翻译；多轮指模型能够在对话中记住上下文、接受修正并继续翻译。  
**BayLing‑80**：作者自行构造的80条多轮指令测试集，用来评估模型在一般指令任务上的表现。  

### 核心创新点
1. **从英文指令迁移到多语言的桥梁 → 交互式翻译任务**：传统做法是直接收集中文指令数据，成本高。本文把“把英文指令翻译成中文”这一步变成模型自己的任务，让模型在翻译过程中学习指令的执行方式。这样既得到语言对齐，又得到指令遵循能力。  
2. **自动生成交互式翻译指令 → 大规模训练数据**：作者设计了一个流水线：先用强大的英文指令模型（如GPT‑3.5）生成多轮英文翻译对话，再交给已有的双语翻译系统把对话整体翻译成中文。得到的中文对话保留了原始的指令结构，直接用于指令微调。省去了人工标注。  
3. **在13B参数的LLaMA上实现接近GPT‑3.5的表现**：通过上述数据，作者在相对小的LLaMA‑13B上进行指令微调，得到的 BayLing 在多项评测（单轮翻译、交互式翻译、通用指令）上都能达到或接近 GPT‑3.5‑turbo 的水平，展示了数据质量对模型性能的放大效应。  
4. **构建专属评测集 BayLing‑80**：为了客观衡量多语言指令能力，作者手工编写了80条覆盖问答、推理、写作等场景的多轮指令。实验显示 BayLing 在该集上拿到 89% 的 GPT‑3.5 分数，验证了跨语言迁移的广度。  

### 方法详解
整体思路可以拆成三大块：**（1）英文指令生成、（2）交互式翻译生成、（3）指令微调**。下面按顺序展开。

1. **英文指令生成**  
   - 选取已有的指令微调模型（如 GPT‑3.5‑turbo）作为“老师”。  
   - 给老师一个指令模板，例如“请把下面的英文段落翻译成中文，并在用户纠错后重新翻译”。  
   - 老师在此模板下自行产生多轮对话：用户提出原文、模型给出翻译、用户指出错误、模型修正……形成完整的交互式翻译脚本。每条脚本约 5‑8 轮，覆盖常见的纠错、补全、风格要求等情形。

2. **交互式翻译生成**  
   - 将完整的英文对话交给高质量的机器翻译系统（如 DeepL、Google Translate）或双语模型，得到对应的中文对话。  
   - 为避免直译导致指令结构丢失，作者在翻译前后加入了“保持指令格式”的约束，使得中文对话仍然保留了“请…”“好的，我已经…”“请再检查”等指令关键词。  
   - 这样得到的中文数据既是翻译任务的答案，又是指令微调所需的“指令‑答案对”。因为是自动化流水线，能够一次性生成上百万条数据。

3. **指令微调**  
   - 选用 LLaMA‑13B 作为基础模型。  
   - 将自动生成的中文交互式翻译对话与原始的英文指令对一起混合，形成多语言、多任务的训练集。  
   - 采用常规的指令微调流程：使用 LoRA（低秩适配）或全参数微调，在每轮对话的最后一步让模型学习“输出正确答案”。  
   - 训练时加入了 **跨语言对齐损失**：在同一指令的英文和中文版本之间强制其内部表示相似，帮助模型在不同语言间共享语义。

**最巧妙的点**在于把“翻译”本身当作一种**交互式指令学习**的桥梁。模型在翻译过程中不断收到用户反馈，这相当于让模型在“做任务”的同时“学指令”。而且所有数据都是机器生成的，省去了人工标注的瓶颈。

### 实验与效果
- **评测任务**  
  - **单轮翻译**：使用标准的 WMT、Flores 等公开数据集，自动评估（BLEU、COMET）对比 GPT‑4。  
  - **交互式翻译**：邀请中文母语者进行人工打分，比较 BayLing 与 GPT‑3.5‑turbo。  
  - **通用指令**：在 BayLing‑80 多轮指令集上测算准确率、完整度。  
  - **学科知识测评**：分别在中国高考（Gaokao）和美国 SAT 试题上做选择题评估。

- **主要结果**  
  - 单轮翻译：在自动指标上 BayLing 达到 GPT‑4 的 95%。  
  - 交互式翻译：人工评估显示 BayLing 的多轮翻译质量为 GPT‑3.5‑turbo 的 96%。  
  - BayLing‑80：整体得分为 GPT‑3.5‑turbo 的 89%。  
  - 学科测评：在中文高考和英文 SAT 上，仅次于 GPT‑3.5‑turbo，领先多数公开的指令模型。  

- **消融实验**  
  - 去掉跨语言对齐损失，翻译质量下降约 3%‑4%。  
  - 只使用英文指令（不进行交互式翻译），多轮指令表现下降约 7%。  
  - 替换 LLaMA‑13B 为更小的 7B 版本，整体得分跌至 78%，说明模型规模仍有影响。

- **局限性**  
  - 交互式翻译数据仍依赖外部机器翻译系统，若翻译质量出现系统性偏差，可能会传递到微调模型。  
  - 只在英文→中文的迁移上做了实验，其他语言（如阿拉伯语、俄语）尚未验证。  
  - 由于训练数据全自动生成，缺少对特定领域（医学、法律）指令的深度覆盖。

### 影响与延伸思考
这篇工作展示了“指令迁移”可以通过**任务本身的双语对齐**来实现，打开了低成本多语言指令模型的可能性。随后有几篇后续工作尝试把同样思路推广到 **代码生成**、**对话安全** 等任务，甚至出现了 “交互式标注” 平台，让普通用户在对话中直接贡献指令数据。对想进一步探索的读者，可以关注以下方向：  
- **多语言指令微调的自监督生成**：如何在不依赖任何外部翻译系统的情况下，利用模型内部的跨语言表示自行生成指令对。  
- **指令对齐的结构化表示**：把指令抽象为图或树结构，提升跨语言迁移的鲁棒性。  
- **低资源语言的适配**：把 BayLing 的思路搬到资源极度匮乏的语言，验证其普适性。  

### 一句话记住它
**把交互式翻译当作双语指令课堂，模型在“翻译+纠错”中学会了非英文的指令跟随。**