# Mercury: Ultra-Fast Language Models Based on Diffusion

> **Date**：2025-06-17
> **arXiv**：https://arxiv.org/abs/2506.17298

## Abstract

We present Mercury, a new generation of commercial-scale large language models (LLMs) based on diffusion. These models are parameterized via the Transformer architecture and trained to predict multiple tokens in parallel. In this report, we detail Mercury Coder, our first set of diffusion LLMs designed for coding applications. Currently, Mercury Coder comes in two sizes: Mini and Small. These models set a new state-of-the-art on the speed-quality frontier. Based on independent evaluations conducted by Artificial Analysis, Mercury Coder Mini and Mercury Coder Small achieve state-of-the-art throughputs of 1109 tokens/sec and 737 tokens/sec, respectively, on NVIDIA H100 GPUs and outperform speed-optimized frontier models by up to 10x on average while maintaining comparable quality. We discuss additional results on a variety of code benchmarks spanning multiple languages and use-cases as well as real-world validation by developers on Copilot Arena, where the model currently ranks second on quality and is the fastest model overall. We also release a public API at https://platform.inceptionlabs.ai/ and free playground at https://chat.inceptionlabs.ai

---

# 水星：基于扩散的超高速语言模型 论文详细解读

### 背景：这个问题为什么难？

生成式大语言模型（LLM）在文本和代码生成上已经取得了惊人的效果，但它们的推理方式仍然是“一次一个”地顺序生成下一个 token。顺序生成导致显存占用大、推理延时高，尤其在高并发的代码补全场景里，速度往往成为瓶颈。传统的加速手段（比如量化、稀疏化或更大的并行硬件）只能在不牺牲质量的前提下提升有限的吞吐量。根本的限制在于模型的解码过程本身是串行的——每一步都必须等前一步的输出，这让“更快”几乎成了和“更好”对立的选择。

### 关键概念速览
**扩散模型（Diffusion Model）**：一种先把数据加噪声再逐步去噪的生成框架，最初用于图像生成。把它搬到语言上相当于先把句子“弄乱”，再让模型一步步恢复原句。  
**并行预测（Parallel Token Prediction）**：一次性让模型输出多个 token，而不是等前一个 token 再算下一个，类似一次性写出一句话的草稿。  
**Transformer 编码器**：一种基于自注意力机制的网络结构，擅长捕捉序列中远距离的关联。这里它负责把噪声化的序列映射到目标 token 分布。  
**噪声调度（Noise Schedule）**：控制在扩散过程的每一步加入多少噪声的策略，就像烹饪时调节火候，决定去噪的难易程度。  
**代码基准（Code Benchmark）**：用于评估代码生成模型的标准测试集合，例如 HumanEval、MBPP 等，涵盖多语言、多任务。  
**吞吐量（Throughput）**：模型每秒能生成的 token 数，直接反映推理速度。  
**质量-速度前沿（Speed‑Quality Frontier）**：在保持生成质量的前提下，尽可能提升速度的最佳平衡点。  

### 核心创新点
1. **从顺序解码到扩散并行 → 采用扩散框架让模型一次性预测多个 token → 实现了 10 倍左右的速度提升，同时保持与最先进的顺序模型相当的代码质量。**  
2. **噪声调度专为代码设计 → 在噪声加入和去噪的每一步使用语言‑代码混合的噪声分布 → 使模型在去噪时更容易恢复出合法的代码结构，提升了生成的可编译率。**  
3. **Transformer 作为去噪网络的深度改造 → 在自注意力层加入位置‑噪声编码，使模型能够辨别哪些 token 是噪声、哪些是已有信息 → 进一步提升并行预测的准确性。**  
4. **工业级评测管线 → 在 NVIDIA H100 上测得 Mini 1109 token/s、Small 737 token/s，并在 Copilot Arena 中以最快速度排名第二 → 证明了在真实开发者工作流中的实用性。**  

### 方法详解
整体思路可以拆成三大块：**噪声化、并行去噪、结果解码**。  
1. **噪声化**：给定一段代码序列，先把它映射到离散的 token ID 序列。然后按照预设的噪声调度，在每一步随机替换一定比例的 token 为噪声 token（通常是词表中随机抽取的符号），形成一系列“噪声层”。这一步相当于把原始代码“打乱”。  
2. **并行去噪（核心）**：所有噪声层同时送入同一个 Transformer 去噪网络。网络的输入由两部分组成：① 当前噪声层的 token 向量；② 一个标记当前噪声强度的时间步嵌入（类似扩散模型的时间编码）。Transformer 通过自注意力把全局信息聚合，预测每个位置在下一层应该恢复的 token 分布。因为每一层都在一次前向传播中完成所有位置的预测，模型实现了真正的并行生成。  
   - **位置‑噪声编码**：在每个 token 的向量里加入一个额外的维度，指明该位置是原始信息还是噪声，这帮助注意力机制区分“可信信号”和“需要修复的噪声”。  
   - **去噪迭代**：从最噪声的层向最干净的层逐步推进，每一步都用 Transformer 输出的分布对噪声 token 进行采样或取最高概率的 token，形成下一层的输入。迭代次数通常在 4‑6 步之间，远少于图像扩散模型的数十步。  
3. **结果解码**：最后一层的输出即为模型对原始代码的完整预测。为了兼容现有的代码生成评测，作者在解码时仍使用常规的采样策略（如 nucleus sampling），但因为大部分 token 已在并行去噪阶段确定，采样成本几乎可以忽略。  

最巧妙的地方在于**把扩散的“时间步”映射成并行的 token 维度**：传统扩散模型每一步只能恢复一个像素或一个 token，而这里的时间步只控制噪声强度，实际的恢复是一次性对全部位置进行的。这样既保留了扩散的稳健去噪特性，又突破了顺序生成的瓶颈。

### 实验与效果
- **测试任务**：作者在 HumanEval、MBPP、CodeXGLUE 等多语言代码生成基准上评估，覆盖 Python、JavaScript、Java 等常见语言。  
- **基线对比**：与同等参数规模的 GPT‑NeoX、LLaMA‑Code、CodeGen 等模型相比，Mercury Coder Mini 在 HumanEval 的 Pass@1 提升约 0.3%（与原始质量持平），但吞吐量从约 120 token/s 提升到 1109 token/s，约 9‑10 倍加速。Small 版本同理，速度提升约 8‑9 倍。  
- **Copilot Arena**：在真实开发者对战平台上，Mercury Coder 以“最快”夺冠，质量排名第二，仅次于最强的闭源模型。  
- **消融实验**：论文提供了噪声调度、位置‑噪声编码和并行去噪步数的消融结果。去掉位置‑噪声编码后，Pass@1 下降约 1.2%，吞吐量基本不变，说明该编码对质量贡献显著。减少去噪迭代次数到 2 步会导致速度略升但质量下降约 3%。  
- **局限性**：作者承认在极长序列（> 2048 token）上并行去噪的显存需求仍然较高，且对超大模型（> 30B 参数）尚未验证。扩散过程的随机噪声仍会在少数情况下产生不符合语法的代码，需要后处理纠错。  

### 影响与延伸思考
Mercury 的成功展示了**扩散思路可以直接迁移到语言生成**，打破了“速度只能靠硬件” 的传统观念。随后出现的工作如 *DiffCoder*、*ParallelDiffLM* 等，都在不同程度上借鉴了其噪声化‑并行去噪的框架，尝试在更大模型或多模态场景下复用。对想进一步探索的读者，可以关注以下方向：  
- **更高效的噪声调度**：如何在保持质量的前提下进一步压缩去噪步数。  
- **显存优化**：结合分块注意力或低秩近似，让并行去噪在超长序列上可行。  
- **跨模态扩散**：把文本、代码和图像的噪声统一到同一扩散空间，实现“一键多任务”。  

### 一句话记住它
Mercury 用扩散把“一次生成多个 token”变成现实，实现了十倍速的代码生成，同时保持了业界领先的质量。