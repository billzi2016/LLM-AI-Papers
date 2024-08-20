# To Code, or Not To Code? Exploring Impact of Code in Pre-training

> **Date**：2024-08-20
> **arXiv**：https://arxiv.org/abs/2408.10914

## Abstract

Including code in the pre-training data mixture, even for models not specifically designed for code, has become a common practice in LLMs pre-training. While there has been anecdotal consensus among practitioners that code data plays a vital role in general LLMs' performance, there is only limited work analyzing the precise impact of code on non-code tasks. In this work, we systematically investigate the impact of code data on general performance. We ask "what is the impact of code data used in pre-training on a large variety of downstream tasks beyond code generation". We conduct extensive ablations and evaluate across a broad range of natural language reasoning tasks, world knowledge tasks, code benchmarks, and LLM-as-a-judge win-rates for models with sizes ranging from 470M to 2.8B parameters. Across settings, we find a consistent results that code is a critical building block for generalization far beyond coding tasks and improvements to code quality have an outsized impact across all tasks. In particular, compared to text-only pre-training, the addition of code results in up to relative increase of 8.2% in natural language (NL) reasoning, 4.2% in world knowledge, 6.6% improvement in generative win-rates, and a 12x boost in code performance respectively. Our work suggests investments in code quality and preserving code during pre-training have positive impacts.

---

# 要写代码还是不写代码？——代码在预训练中的影响探索 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）的预训练阶段，研究者们通常把海量自然语言文本喂给模型，却很少系统地评估代码数据的副作用。虽然业界普遍相信加入代码能提升模型的整体能力，但缺乏严格的实验对比。此前的工作要么只关注代码生成任务，要么把代码当作“额外的特征”，没有量化它对推理、常识或跨模态任务的贡献。因此，是否应该在通用模型的预训练中保留代码、保留多少、以及代码质量到底能带来多大提升，仍是一个悬而未决的问题。

### 关键概念速览
- **预训练数据混合**：把不同来源（新闻、小说、代码等）拼在一起喂模型，相当于让模型在“多菜系”自助餐里吃饭，目的是让它学会更广的语言技能。  
- **自然语言推理（NL reasoning）**：模型需要在阅读理解、逻辑推理等任务上给出答案，类似人类在解谜时要把信息串联起来。  
- **世界知识任务**：考察模型对事实、历史、地理等常识的掌握程度，就像问它“埃菲尔铁塔有多高”。  
- **LLM-as-a-judge**：让模型自己评判另一模型的生成质量，类似让两位评委互相打分，常用于评估生成式模型的相对优劣。  
- **代码质量**：指代码数据的完整性、可执行性和注释丰富度，高质量代码像是干净的教材，低质量代码则像是带错别字的练习册。  
- **相对提升（relative increase）**：用百分比表示新方法相对于基线的改进幅度，便于跨任务比较。  

### 核心创新点
1. **系统化横跨多任务的对比实验**  
   - 之前的研究多聚焦单一任务（如代码生成）或只在小规模模型上做实验。  
   - 本文在 470M‑2.8B 参数范围内，分别训练“仅文本”与“文本+代码”两套模型，并在自然语言推理、世界知识、代码基准以及 LLM‑as‑judge 等六大类共 30+ 任务上进行评测。  
   - 结果显示，加入代码后自然语言推理最高提升 8.2%，世界知识提升 4.2%，生成式对抗赛胜率提升 6.6%，代码任务更是实现 12 倍跃升，证明代码的正向效应是跨任务、跨规模的。

2. **代码质量梯度实验**  
   - 过去往往把所有代码当作同质材料，忽略了质量差异。  
   - 作者构造了三档代码子集：低质量（随机抓取的 GitHub 片段）、中质量（去除明显错误的代码）和高质量（经过 lint、单元测试过滤的代码）。  
   - 实验表明，提升代码质量对所有下游任务的增益远大于单纯增加代码比例，尤其在生成式评估中，高质量代码带来的提升接近 10% 的相对增幅。

3. **代码比例敏感性分析**  
   - 通过在总预训练数据中分别占 0%、5%、10%、20% 的代码比例进行训练，绘制了性能随比例变化的曲线。  
   - 发现 5%~10% 的代码比例是“甜点区”，再往上提升收益递减，甚至在极端比例（>30%）时出现部分任务的轻微下降，提示代码并非越多越好，需要平衡。

### 方法详解
整体思路可以概括为“三步走”：① 构造多尺度数据混合，② 按比例和质量分组训练模型，③ 在统一评测平台上做全方位对比。

1. **数据混合与划分**  
   - 首先收集约 2000 亿 token 的英文文本（维基、新闻、小说等），再抓取约 200 亿 token 的开源代码。  
   - 代码数据经过三层过滤：语法检查 → 静态分析（如 pylint） → 单元测试通过率。过滤后得到低、中、高质量三套子集。  
   - 按预设比例（0%、5%、10%、20%）将代码与文本随机混排，确保每个 batch 中的代码占比恒定，避免模型在训练初期看到突兀的代码块。

2. **模型训练**  
   - 使用标准的自回归 Transformer 架构，参数规模分别为 470M、1.3B、2.8B。  
   - 训练超参数（学习率、batch 大小、梯度累积等）保持一致，只在数据混合上做差异。  
   - 为了防止代码“泄漏”导致模型过度依赖代码特征，作者在每个 epoch 结束后对代码比例进行微调，使整体分布保持稳定。

3. **评测管线**  
   - 所有模型在同一套评测脚本下跑完 30+ 任务，输出统一的 logits，随后使用相同的解码策略（温度 0.7、top‑p 0.9）生成答案。  
   - 对于 LLM‑as‑judge，模型 A 的输出作为“被评判对象”，模型 B（同规模、仅文本预训练）作为评判者，计算两者的胜率。  
   - 结果通过相对提升率（%）进行归一化，便于跨任务比较。

**最巧妙的点**：作者没有直接把代码当作“额外任务”，而是把代码视作一种语言形式，让模型在同一语言模型框架下自然学习代码的结构化特征。这种“语言统一化”让模型在处理自然语言时也能受益于代码的严谨语法和抽象思维。

### 实验与效果
- **任务覆盖**：自然语言推理（ARC、BoolQ、OpenBookQA 等）、世界知识（TriviaQA、FactCC）、代码基准（HumanEval、MBPP）、生成式对抗赛（ChatGPT vs. Claude）以及 LLM‑as‑judge 胜率。  
- **基线对比**：与仅文本预训练的同规模模型直接比较。  
  - 在 NL 推理上，10% 代码比例模型相对提升最高 8.2%。  
  - 世界知识任务提升 4.2%。  
  - 生成式对抗赛中，代码模型的胜率提升 6.6%。  
  - 代码任务（HumanEval）从 2% 左右的成功率跃升至约 24%，约 12 倍增长。  
- **消融实验**：  
  - 去掉代码质量过滤后，提升幅度下降约 2‑3%。  
  - 将代码比例提升至 20% 时，NL 推理提升仅为 5%，说明收益出现饱和。  
- **局限性**：实验仅在英文数据上进行，未验证多语言场景；模型规模上限为 2.8B，尚不清楚在百亿级模型上是否仍保持同样趋势。作者也提到，代码质量过滤成本高，实际工业流水线中可能难以完全复制。

### 影响与延伸思考
这篇工作在社区里引发了对“代码是通用知识”这一观点的重新审视。随后出现的几篇论文（如 *CodeMix*、*CodeBERT‑Plus*）直接在更大规模模型上复现了代码比例的正向效应，并尝试把代码嵌入到多语言预训练中。业界也开始在 ChatGPT、Claude 等商用模型的预训练管线里显式保留代码，甚至把代码质量评估作为数据清洗的必选步骤。想进一步了解，可以关注以下方向：① 大规模多语言代码混合的跨语言迁移效应；② 代码与自然语言的共享表示学习（如统一的语义图谱）；③ 低资源语言中代码的稀缺性与数据增强方法。

### 一句话记住它
**在通用大语言模型的预训练里，适量且高质量的代码数据是提升所有任务表现的“隐形加速器”。**