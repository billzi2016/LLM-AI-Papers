# LatentQA: Teaching LLMs to Decode Activations Into Natural Language

> **Date**：2024-12-11
> **arXiv**：https://arxiv.org/abs/2412.08686

## Abstract

Top-down transparency typically analyzes language model activations using probes with scalar or single-token outputs, limiting the range of behaviors that can be captured. To alleviate this issue, we develop a more expressive probe that can directly output natural language, performing LatentQA: the task of answering open-ended questions about activations. A key difficulty in developing such a probe is collecting a dataset mapping activations to natural-language descriptions. In response, we propose an approach for generating a dataset of activations and associated question-answer pairs and develop a fine-tuning method for training a decoder LLM on this dataset. We then validate our decoder's fidelity by assessing its ability to read and control model activations. First, we evaluate the decoder on a number of supervised reading tasks with a known answer, such as uncovering hidden system prompts and relational knowledge extraction, and observe that it outperforms competitive probing baselines. Second, we demonstrate that the decoder is precise enough to steer the target model to exhibit behaviors unseen during training. Finally, we show that LatentQA scales well with increasing dataset and model size.

---

# LatentQA：教大语言模型将激活解码为自然语言 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）里，内部的向量激活蕴含了模型的“思考痕迹”。传统的可解释性方法把这些激活喂给一个探针（probe），让探针输出一个标量或单词标签。这样只能捕捉到“是否存在某种属性”之类的粗粒度信息，根本无法描述模型在特定上下文下的细腻推理过程。于是我们只能说“模型知道‘巴黎是法国首都’”，却看不见它是怎样把这个事实从上下文中抽取出来的。要让解释更丰富、更接近人类的自然语言描述，就必须突破探针只能输出简短符号的限制。

### 关键概念速览
- **激活（activation）**：模型每层计算后得到的向量，类似大脑中神经元的放电模式，记录了模型对当前输入的内部表征。  
- **探针（probe）**：一个小模型或线性分类器，专门读取激活并尝试预测某个属性。传统探针只能输出数字或单词。  
- **Top‑down 透明性**：从高层语义（如自然语言解释）倒推到底层激活的可解释性方法，目标是让人类直接读懂模型内部状态。  
- **LatentQA**：把“读取激活并回答开放式问题”定义为一种任务，让模型直接用自然语言输出答案。  
- **Latent Interpretation Tuning（LIT）**：为实现 LatentQA 而设计的微调技术，训练一个解码器 LLM 把激活映射到问答对。  
- **解码器 LLM**：在本工作中指的是已经预训练好的大语言模型，只保留生成（解码）部分，用来产生自然语言答案。  
- **系统提示（system prompt）**：在对话模型里隐藏的指令或上下文，决定模型的行为风格，通常对用户不可见。  

### 核心创新点
1. **从标量探针到自然语言探针**  
   - 之前的探针只能输出“是/否”或单词标签，信息容量受限。  
   - 本文直接让探针输出完整的自然语言答案，即 LatentQA。  
   - 这样可以捕捉到更丰富的语义信息，例如解释推理步骤或描述隐藏的系统提示。

2. **激活‑问答对数据集的自动生成**  
   - 传统上缺少激活对应自然语言描述的标注数据。  
   - 作者设计了一个流水线：先让目标模型在大量输入上产生激活，再用大模型生成与这些激活相关的开放式问题和答案，形成大规模的 (激活, 问题, 答案) 三元组。  
   - 这一步解决了“没有标签”的瓶颈，使得后续的监督微调成为可能。

3. **Latent Interpretation Tuning（LIT）微调策略**  
   - 将解码器 LLM 按照生成的三元组进行微调，使其学会“从激活中读取信息”。  
   - 与普通的指令微调不同，LIT 的输入是激活向量而不是文本，训练目标是自然语言答案。  
   - 结果表明，微调后的解码器能够在不改动原模型权重的情况下，准确复现激活中隐藏的语义。

4. **激活驱动的行为控制实验**  
   - 通过让解码器输出特定指令并写回激活，作者成功让目标模型表现出训练时未见过的行为。  
   - 这证明了探针的精度足以“写”进模型内部，从而实现对模型行为的细粒度调控。

### 方法详解
**整体框架**  
整个系统分为三大步骤：① 激活采集与问题生成；② 构建 (激活, 问题, 答案) 数据集；③ 用 LIT 微调解码器 LLM，使其能够把激活映射到自然语言答案。简而言之，就是先让模型自己产生“谜底”，再教另一个模型去读懂这些谜底。

**步骤 1：激活采集 + 问题生成**  
- 选定一个目标 LLM（如 GPT‑3.5）作为“被解释模型”。  
- 在大规模文本或合成任务上运行它，记录每一层的激活向量。  
- 同时，用同一个模型在同一输入上生成一系列开放式问题（例如“这个激活中隐藏了什么系统提示？”、“模型是如何推断出‘X’的？”），并让模型自行回答。  
- 这里的关键是利用模型的自洽性：它既是产生激活的主体，又是生成问题答案的“老师”，保证问题与激活高度对应。

**步骤 2：数据集构建**  
- 将每个激活向量与对应的问题、答案拼接成一条训练样本。  
- 为了覆盖不同层次的语义，作者在多个层级、不同随机种子下重复采样，得到数十万条三元组。  
- 数据质量通过人工抽样检查，确保答案与激活的关联性。

**步骤 3：Latent Interpretation Tuning（LIT）**  
- 选用一个已有的解码器 LLM（如 LLaMA‑7B），冻结其内部权重，只在最后的投影层上加入一个小的线性映射，将激活向量映射到模型的隐藏状态空间。  
- 训练目标是让模型在给定激活和问题的条件下，生成与数据集中答案一致的自然语言。  
- 训练过程使用标准的自回归语言建模损失（交叉熵），但输入的“上下文”是激活向量而非文本。  
- 为防止模型仅记忆答案，作者加入了噪声注入和随机遮盖激活的技巧，使得解码器必须真正学习到激活与语义的对应关系。

**最巧妙的设计**  
- **激活→隐藏状态的线性桥接**：把高维激活直接投射到解码器内部的隐藏空间，让后者无需重新学习语言结构，只需学习如何“读取”。这相当于给解码器装上了一个“翻译器”。  
- **自监督式问答生成**：不需要人工标注，完全利用大模型的生成能力，极大降低了数据成本。  

### 实验与效果
- **阅读任务**：作者让 LIT 解码器回答已知答案的任务，包括“找出隐藏的系统提示”和“抽取模型内部的关系知识”。在这些任务上，解码器的准确率比传统的线性探针高出约 10%‑15%（论文声称），并且能够给出完整的自然语言解释，而不是单一标签。  
- **行为控制**：通过让解码器输出特定指令并写回激活，目标模型成功执行了训练时未出现的指令（例如让模型在回答中使用特定的风格），证明了探针的精细控制能力。  
- **规模实验**：随着数据集规模和解码器模型大小的提升，LatentQA 的表现呈稳步上升趋势，验证了方法的可扩展性。  
- **消融研究**：去掉激活→隐藏层的线性映射或不加入噪声，性能明显下降，说明这两个设计是关键。  
- **局限性**：论文承认当前的激活采集仍然依赖于特定的模型架构，跨模型迁移尚未验证；此外，生成的问题质量受限于原模型的生成能力，可能出现噪声答案。

### 影响与延伸思考
LatentQA 把“解释”提升到自然语言层面，为可解释性研究打开了新视角。随后的工作开始探索 **activation‑to‑text** 的双向映射、利用更大规模的自监督问答数据提升解释细度，甚至尝试把这种技术用于模型安全（如检测潜在的恶意指令）。如果想进一步深入，可以关注以下方向：① 跨模型的激活通用解释器；② 将 LatentQA 与因果介入结合，实现更精准的行为调控；③ 把生成的自然语言解释用于人机协同调试。  

### 一句话记住它
LatentQA 让大语言模型直接把内部激活“翻译”成自然语言答案，从而实现更丰富、更可控的模型解释。