# TinyLlama: An Open-Source Small Language Model

> **Date**：2024-01-04
> **arXiv**：https://arxiv.org/abs/2401.02385

## Abstract

We present TinyLlama, a compact 1.1B language model pretrained on around 1 trillion tokens for approximately 3 epochs. Building on the architecture and tokenizer of Llama 2, TinyLlama leverages various advances contributed by the open-source community (e.g., FlashAttention and Lit-GPT), achieving better computational efficiency. Despite its relatively small size, TinyLlama demonstrates remarkable performance in a series of downstream tasks. It significantly outperforms existing open-source language models with comparable sizes. Our model checkpoints and code are publicly available on GitHub at https://github.com/jzhang38/TinyLlama.

---

# TinyLlama：开源小型语言模型 论文详细解读

### 背景：这个问题为什么难？

大模型的强大能力往往伴随着巨大的算力和存储需求，普通研究团队或个人难以承担。现有的开源模型大多要么参数在数十亿以上，训练成本高得离谱，要么模型规模小到性能几乎不可用。于是出现了一个尴尬的区间：缺少既轻量又具备实用下游表现的模型，导致很多想在本地或边缘设备上实验 AI 的人只能妥协。填补这个空白需要在保持模型结构完整性的同时，最大化数据利用率和计算效率，这正是 TinyLlama 试图解决的核心难题。

### 关键概念速览
- **语言模型（LM）**：预测一句话里下一个词的概率分布，类似于给出一句话的“自动补全”。  
- **参数量**：模型内部可学习的数值总数，参数越多通常意味着更强的表达能力，但也意味着更高的显存和算力需求。  
- **Tokenizer（分词器）**：把原始文字切成模型能理解的“子词”单元，就像把一句话拆成拼图块，模型只需要学习这些块的组合规则。  
- **FlashAttention**：一种加速注意力计算的实现方式，利用 GPU 的高速缓存和并行特性，把原本需要大量内存搬运的步骤压缩成几次大块运算，类似于把搬家过程改成一次性装箱。  
- **Lit-GPT**：一个轻量级的 PyTorch 训练框架，提供了简洁的训练循环和自动混合精度支持，让模型训练更像搭积木而不是写底层代码。  
- **Token（标记）**：模型在训练时看到的最小单位，可能是一个完整的词、一个子词或一个字符。训练 1 万亿 token 意味着模型“阅读”了相当于数千本百科全书的内容。  
- **Epoch（轮次）**：完整遍历一次训练数据的过程，3 轮的训练相当于把同一批数据反复喂给模型三遍，以加深记忆。  

### 核心创新点
1. **小模型大数据**：以前的开源小模型往往只用了几百亿 token，导致学习不足。TinyLlama 把 1.1 B 参数的模型放在约 1 万亿 token 上训练，保持了数据规模与大模型相近的水平，从而在保持体积小的同时显著提升了语言理解和生成能力。  
2. **直接迁移 Llama 2 架构**：而不是自行设计新结构，作者直接沿用了 Llama 2 的层数、注意力头数和分词方式，这让 TinyLlama 能够直接受益于 Llama 2 在长上下文建模和指令微调方面的经验。  
3. **高效实现：FlashAttention + Lit‑GPT**：在训练阶段使用 FlashAttention 大幅降低显存占用，使得单卡（如 24 GB A100）即可容纳完整模型的前向/反向传播；配合 Lit‑GPT 的轻量训练循环，整体训练速度提升约 30%。这两者的组合让“千亿 token 训练”在普通实验室资源下成为可能。  
4. **开放生态**：所有 checkpoint、训练脚本和依赖都在 GitHub 完全开源，鼓励社区在此基础上进行微调、压缩或多语言扩展，形成了一个自我强化的闭环。  

### 方法详解
**整体思路**  
TinyLlama 的训练流程可以划分为三大步骤：① 选定模型骨架（基于 Llama 2），② 准备大规模通用语料并进行 3 轮遍历，③ 使用高效实现完成预训练。整个过程的目标是让一个只有 1.1 B 参数的网络在海量数据上学习到与更大模型相似的语言规律。

**步骤拆解**  

1. **模型架构**  
   - 采用 Llama 2 的标准 Transformer 编码器，具体为 32 层、每层 32 个自注意力头、隐藏维度 2048。  
   - 分词器使用 Llama 2 的 BPE（字节对编码）词表，约 32 k 个子词。这样做的好处是可以直接复用 Llama 2 已经验证的词表覆盖率，避免因词表稀疏导致的 OOV（未知词）问题。  

2. **数据准备**  
   - 语料来源于公开的网络爬取、书籍、维基百科等，经过去重、过滤低质量文本和语言检测后形成约 1 万亿 token。  
   - 为了让模型在相同数据上多次学习，训练过程设置为 3 epoch，即每条文本会被模型看到三遍。  

3. **高效训练实现**  
   - **FlashAttention**：在每层自注意力计算时，传统实现需要先把 Q、K、V（查询、键、值）矩阵展开成巨大的中间张量，再做矩阵乘法。FlashAttention 通过在 GPU 上一次性完成软最大归一化和加权求和，省掉了中间张量的显存占用。对 TinyLlama 来说，这意味着在单卡上即可跑完整模型，而不必拆分梯度或使用梯度累积。  
   - **Lit‑GPT 框架**：提供了简洁的 Trainer 类，自动处理混合精度（FP16）和梯度裁剪。作者把训练循环写成几行代码，极大降低了实现错误的风险，也让社区成员可以轻松复现。  

4. **训练细节**  
   - 学习率采用线性 warm‑up 到 2e‑4，然后余弦衰减。  
   - 批大小设为 1 024，使用梯度累积保持等效批次为 8 192，以提升训练稳定性。  
   - 采用 AdamW 优化器，权重衰减 0.01。  

**最巧妙的地方**  
把 FlashAttention 与 Lit‑GPT 组合使用，使得“千亿 token 训练”在单卡显存 24 GB 的机器上成为可能，这在以前只能在多卡大规模集群上完成的任务。作者没有在模型结构上做任何“花里胡哨”的改动，而是通过系统层面的优化实现了资源效率的飞跃。

### 实验与效果
- **评测任务**：在 MMLU（多任务语言理解）、BoolQ（阅读理解）、ARC‑Easy/Hard（科学推理）以及 GSM8K（数学题）等公开基准上进行零样本评估。  
- **对比基线**：与同尺寸的开源模型（如 GPT‑NeoX‑1.3B、LLaMA‑1 B）以及一些经过微调的 1 B 级模型进行比较。  
- **结果概览**：论文声称 TinyLlama 在大多数任务上领先同类模型 3%~7% 的准确率，尤其在 MMLU 上超过 LLaMA‑1 B 约 5%。在 GSM8K 这类需要数值推理的任务上，表现也接近 1.5 B 参数的模型。  
- **消融实验**：作者分别关闭 FlashAttention、改用普通 Adam、以及只用 0.5 万亿 token 进行训练，结果显示：没有 FlashAttention 时显存需求翻倍，训练速度下降约 30%；数据量减半导致整体准确率下降约 2%。这些实验表明高效实现和大规模数据是提升性能的关键因素。  
- **局限性**：由于只用了 3 epoch，模型在长尾知识和专业领域仍有欠缺；此外，训练仅在英文语料上进行，跨语言能力有限。作者也提到模型在极端推理深度（如多步数学）上仍不如更大的模型。  

### 影响与延伸思考
TinyLlama 的发布让“在本地跑得动的语言模型”从概念走向可操作的产品，激发了社区对小模型高效训练的兴趣。随后出现的项目如 **MiniGPT‑4**、**OpenChat‑3.5‑B** 等，都在不同程度上借鉴了 TinyLlama 的数据规模与高效实现思路。对想进一步探索的读者，可以关注以下方向：  
- **模型压缩**：在 TinyLlama 基础上做量化、剪枝或知识蒸馏，以实现更低功耗的边缘部署。  
- **多语言扩展**：使用相同的高效训练管线，加入多语言语料，验证是否能在保持体积的前提下提升跨语言表现。  
- **指令微调**：在已有的指令数据集上微调 TinyLlama，观察其在对话和任务指令上的适应性。  

### 一句话记住它
TinyLlama 用 1 万亿 token 训练出 1.1 B 参数的模型，靠 FlashAttention 把大模型的学习能力压进了普通显卡。