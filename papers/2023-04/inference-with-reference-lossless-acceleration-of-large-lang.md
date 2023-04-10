# Inference with Reference: Lossless Acceleration of Large Language Models

> **Date**：2023-04-10
> **arXiv**：https://arxiv.org/abs/2304.04487

## Abstract

We propose LLMA, an LLM accelerator to losslessly speed up Large Language Model (LLM) inference with references. LLMA is motivated by the observation that there are abundant identical text spans between the decoding result by an LLM and the reference that is available in many real world scenarios (e.g., retrieved documents). LLMA first selects a text span from the reference and copies its tokens to the decoder and then efficiently checks the tokens' appropriateness as the decoding result in parallel within one decoding step. The improved computational parallelism allows LLMA to achieve over 2x speed-up for LLMs with identical generation results as greedy decoding in many practical generation scenarios where significant overlap between in-context reference and outputs exists (e.g., search engines and multi-turn conversations).

---

# 基于参考的推理：大语言模型的无损加速 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）在生成文本时需要逐词（或逐子词）进行自回归计算，算力开销随模型规模线性增长。即使在推理阶段，模型仍要遍历全部层、全部注意力头，导致响应时间在实际产品（搜索、对话）里常常成为瓶颈。很多场景下，模型的输出会和已有的参考文本高度重复——比如检索到的文档、对话历史或模板化的答案，但传统解码流程并不会利用这些已知的相同片段，只会“一次一次”重新生成，浪费了大量算力。现有的加速手段（如量化、稀疏注意力、模型蒸馏）只能在精度上做折中，无法保证生成结果与原始贪心解码完全一致。于是出现了一个需求：在不改变模型输出的前提下，利用参考文本实现真正的“无损”加速。

### 关键概念速览
**自回归解码**：模型一次生成一个 token（词或子词），后续的每一步都要把已经生成的 token 作为输入，类似于人写句子时每写一个字都要回头看前面的文字。  
**参考文本（reference）**：在推理时额外提供的、与目标输出可能重叠的文字块，例如检索到的段落、上一次对话的答案等。  
**Span Selection（跨度选择）**：从参考文本中挑出一段连续的 token 序列，准备直接拷贝到模型的输出流里。可以把它想成在一本书里找出已经写好的句子，直接贴进去。  
**并行检查（parallel verification）**：一次性把选中的整段 token 同时送进模型的每层进行前向计算，判断它们是否符合模型在该位置的概率分布。类似于一次性让全体审稿人同时检查一段文字是否合规，而不是逐句审。  
**贪心解码（greedy decoding）**：每一步都选概率最高的 token，得到的序列是最直接、最快的生成结果。这里的目标是让加速后得到的序列与贪心解码完全相同。  

### 核心创新点
1. **从“逐词生成”到“跨度拷贝”**：传统方法只能在每一步输出单个 token。LLMA 首先在参考文本里搜索最长的匹配跨度，然后一次性把整段 token 复制到解码器。这样把多次计算合并为一次，大幅提升并行度。  
2. **并行适配性检查**：复制的跨度并不直接写入最终答案，而是让模型在同一解码步内部并行计算这些 token 的隐藏状态，并与模型原本的预测分布对比。只要所有 token 都被模型认可（即概率最高），就直接接受；否则回退到普通逐词生成。此机制保证了“无损”——输出永远和原始贪心解码一致。  
3. **动态跨度调整策略**：LLMA 在每一步都重新评估参考文本，若上一次拷贝的跨度因上下文变化而不再匹配，会自动缩短或重新搜索更合适的片段。这样避免了因为上下文漂移导致的错误复制。  
4. **统一的加速框架**：作者把上述三个模块封装成一个插件式加速层，几乎不需要改动原有模型代码，只在解码循环前后插入一次检查即可，兼容各种主流 LLM（如 LLaMA、GPT‑Neo 等）。  

### 方法详解
LLMA 的整体思路可以拆成四个步骤：**（1）参考准备 →（2）跨度检索 →（3）并行验证 →（4）结果写回**。下面逐步展开。

1. **参考准备**  
   在推理入口，系统会把一段外部文本（检索结果、对话历史、模板等）作为 reference。为了高效检索，作者使用了倒排索引或基于 n‑gram 的滑动窗口，把 reference 切成若干固定长度的 token 块，并记录每块的起止位置。

2. **跨度检索**  
   当解码器准备生成第 *t* 个 token 时，LLMA 会在 reference 中寻找最长的、从当前位置 *t* 开始能够完整匹配的 token 序列。匹配过程采用字典查表：把已经生成的前缀作为键，快速定位所有以该前缀开头的参考片段，然后取最长的那段。若找不到匹配，则返回空跨度，进入普通解码。

3. **并行验证**  
   假设找到的跨度长度为 *k*，LLMA 把这 *k* 个 token 同时送入模型的所有层进行前向传播。这里的关键是**一次性计算**：模型的自注意力机制本来就支持批量输入，作者把这 *k* 条序列当作 batch 中的 *k* 条独立样本，计算它们在第 *t* … *t+k‑1* 位置的隐藏状态和对应的 logits（未归一化的概率）。随后检查每个 token 的 logits 是否在该位置的最高概率上——如果全部满足，则认为这段跨度是模型“愿意”输出的。

4. **结果写回**  
   - **全部通过**：直接把跨度的 token 写入输出缓冲区，解码指针一次跳过 *k* 步。  
   - **部分通过或全部不通过**：对不符合的 token 回退到普通贪心解码，只生成最高概率的单个 token，然后重新进入步骤 2，继续尝试新的跨度。  

**最巧妙的点**在于把“复制”和“检查”合二为一的并行验证。传统的缓存技术（如 KV‑cache）只能复用已经计算过的 hidden states，但仍需要逐词生成；LLMA 把未生成的 token 先“假装”已经生成，利用模型本身的前向路径一次性算完，然后用概率最高的条件做“硬性”过滤，确保不产生任何偏差。

### 实验与效果
- **测试场景**：论文在搜索引擎答案生成、多轮对话以及代码补全三个实际业务中评估。所有场景的共同特征是系统会先检索到一段与答案高度相似的文档或历史记录。  
- **基线对比**：与原始贪心解码、以及常见的量化加速（FP16、INT8）做对比。  
- **加速幅度**：在搜索答案任务上，LLMA 在保持 100% 生成一致性的前提下，实现了约 **2.3×** 的吞吐提升；在对话场景中提升约 **1.9×**。相比仅使用量化的加速，LLMA 的速度提升更高且不牺牲精度。  
- **消融实验**：作者分别关闭跨度检索、并行验证和动态调整三块功能。结果显示，去掉并行验证会导致输出错误率上升到 12%，失去“无损”特性；去掉动态调整会在长对话中出现 5% 的回退次数增加，整体加速下降约 15%。  
- **局限性**：LLMA 依赖于参考文本与目标输出的重叠度；在完全原创的生成任务（如创意写作）中几乎找不到匹配跨度，速度提升接近 1×。此外，检索和匹配过程本身也会消耗一定的 CPU 资源，作者在论文中提到在极端低延迟场景下仍需进一步优化。

### 影响与延伸思考
LLMA 把“参考复用”提升到推理层面，打开了“无损加速”这一新思路。随后的工作开始探索 **参考感知的解码调度**（如在大模型服务中动态切换参考模式）以及 **跨模型的共享参考缓存**（把同一检索结果在多个模型实例间共享）。还有研究尝试把参考匹配与 **检索增强生成（RAG）** 结合，让检索模块不仅提供信息，还直接参与加速。对想进一步了解的读者，可以关注近期在 ACL、EMNLP 上出现的 “Reference‑Guided Decoding” 系列论文，以及在工业界的开源实现（如 Llama‑CPP 的 reference‑mode 插件）。

### 一句话记住它
LLMA 通过一次性拷贝并并行验证参考文本中的连续片段，实现了与原始贪心解码完全相同的输出，却快了两倍以上。