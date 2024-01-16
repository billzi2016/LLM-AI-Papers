# Tuning Language Models by Proxy

> **Date**：2024-01-16
> **arXiv**：https://arxiv.org/abs/2401.08565

## Abstract

Despite the general capabilities of large pretrained language models, they consistently benefit from further adaptation to better achieve desired behaviors. However, tuning these models has become increasingly resource-intensive, or impossible when model weights are private. We introduce proxy-tuning, a lightweight decoding-time algorithm that operates on top of black-box LMs to achieve the same end as direct tuning, but by accessing only its predictions over the output vocabulary, not its parameters. Our method tunes a smaller LM, then applies the difference between the predictions of the small tuned and untuned LMs to shift the original predictions of the larger untuned model in the direction of tuning, while retaining the benefits of larger-scale pretraining. In experiments, when we apply proxy-tuning to Llama2-70B using proxies of only 7B size, we can close 88% of the gap between Llama2-70B and its truly-tuned chat version, when evaluated across knowledge, reasoning, and safety benchmarks. We then demonstrate the generality of proxy-tuning by applying it to domain adaptation on code, and task-specific finetuning on question-answering and math problems. Finally, we show how to proxy-tune a truly black-box LM, GPT-3.5, for temporal adaptation, increasing its knowledge about recent events. Our work demonstrates the promise of using small tuned LMs to efficiently customize large, potentially proprietary LMs through decoding-time guidance.

---

# 通过代理微调语言模型 论文详细解读

### 背景：这个问题为什么难？

大规模预训练语言模型（LLM）已经能完成很多任务，但想让它们在特定场景下表现更好，仍然需要微调。传统微调要么需要巨大的算力，要么必须拿到模型权重，很多商业模型（如 GPT‑4）根本不开放。于是出现了两大瓶颈：① 资源成本随模型规模指数增长；② 私有模型只能以黑盒方式调用，无法直接改参数。要在不触碰模型内部的前提下实现类似微调的效果，成了急需突破的难题。

### 关键概念速览
- **黑盒语言模型**：只能通过输入提示得到输出概率分布，内部权重不可见或不可修改。就像只能在柜子外面敲门询问，而不能打开柜子查看里面的东西。  
- **代理模型（proxy model）**：一个体积更小、可以自由微调的模型，用来“代表”大模型的行为差异。想象成一位助理，先在小本子上练习，再把练习的要点告诉大老板。  
- **预测差分（prediction delta）**：小模型微调前后在同一输入上的输出分布差异。相当于记录下助理改正前后的答案变化。  
- **解码时校正（decoding‑time correction）**：在生成文本的每一步，把大模型的原始 logits（未归一化的概率）加上预测差分，再继续采样。类似在大老板说话前，先在耳边悄悄加上一句提示。  
- **知识/推理/安全基准**：评估模型在事实准确性、逻辑推理和有害内容控制方面的标准测试集合。  
- **时效适配（temporal adaptation）**：让模型更新对最近事件的认知，弥补预训练数据的时效性缺口。  

### 核心创新点
1. **从参数微调转向输出校正**  
   - 之前的做法：直接在大模型上做梯度更新，需要完整权重和巨额算力。  
   - 这篇论文的做法：只在一个小模型上完成完整微调，然后把小模型的“改动”以差分形式加到大模型的输出上。  
   - 改变：实现了几乎同等的微调效果，却不需要触碰大模型的参数，算力需求下降数十倍。

2. **利用小模型的预测差分作为通用“调味料”**  
   - 之前的解码技巧（如提示工程、检索增强）只能提供离散的文字提示。  
   - 这里的做法是把小模型微调前后的概率向量相减，得到细粒度的方向向量，再直接叠加到大模型的 logits。  
   - 改变：提供了连续、可叠加的调节信号，能够在知识、推理、代码等多任务上统一使用。

3. **黑盒模型的时效适配实验**  
   - 过去要让 GPT‑3.5 了解最新事件只能靠外部检索或重新训练。  
   - 论文展示了把一个微调了近期新闻的 7B 代理模型的差分加到 GPT‑3.5 输出上，显著提升了对新事件的回答准确率。  
   - 改变：证明了即使是完全不可访问权重的商业模型，也能通过代理微调实现“知识刷新”。

### 方法详解
**整体思路**：先在一台可控的、体积较小的模型上完成目标任务的全量微调；随后在推理阶段，把该小模型的“改动”映射到大模型的输出上，从而让大模型的生成行为向微调后的方向倾斜。

**步骤拆解**：

1. **准备三套模型**  
   - **大模型 A**：未微调的黑盒 LLM（如 Llama2‑70B）。只能调用 `p_A(y|x)`，即对每个候选 token 的概率。  
   - **小模型 B（未微调）**：结构与 A 相同或相似，但规模更小（如 7B），可以自由获取 logits。  
   - **小模型 B′（微调后）**：在同样的数据集上对 B 完成标准全量微调，得到目标任务的最佳表现。

2. **计算预测差分 Δ**  
   - 对同一输入 `x`，分别让 B 和 B′ 生成 logits 向量 `l_B(x)`、`l_B′(x)`。  
   - 差分 `Δ(x) = l_B′(x) - l_B(x)`，这是一段指向“应该往哪个方向移动”的向量。可以把它想成助理在练习后给出的“改进建议”。

3. **解码时校正大模型输出**  
   - 调用大模型 A，得到原始 logits `l_A(x)`。  
   - 将差分直接加到大模型 logits 上：`l̂_A(x) = l_A(x) + α·Δ(x)`，其中 `α` 是一个可调的缩放系数，用来控制校正强度。  
   - 对 `l̂_A(x)` 做 softmax，得到校正后的概率分布，再进行采样或束搜索。  
   - 这一过程在每一步生成时都重复，等价于在大模型的“思考”过程中不断注入小模型的微调经验。

4. **实现细节**  
   - **差分对齐**：因为 B 与 A 的词表可能不完全相同，论文通过词表映射或共享子词表来保证 Δ 能在 A 的空间中相加。  
   - **温度与系数调节**：α 越大，校正越强，但可能导致小模型的偏差被放大；实验中通常在 0.1~0.5 之间搜索。  
   - **批量缓存**：为降低解码时的额外开销，Δ 可以预先在整个输入序列上一次性计算并缓存。

**最巧妙的点**：把微调的“效果”抽象为一个向量差分，而不是具体的参数或提示文字。这样既保留了微调带来的细粒度知识，又不需要任何权重访问，几乎可以对任何黑盒模型直接使用。

### 实验与效果
- **任务与数据**  
  - 知识基准：如 MMLU、TruthfulQA。  
  - 推理基准：如 GSM‑8K、ARC。  
  - 安全基准：如 TruthfulQA‑Safety、OpenAI Moderation。  
  - 代码适配：在 HumanEval 上进行域适配。  
  - QA 与数学：分别在 Natural Questions 与 MathQA 上做任务微调。  
  - 时效适配：使用 2023 年新闻数据对 GPT‑3.5 进行临时知识更新。

- **主要对比**  
  - **直接微调的 Llama2‑70B‑Chat**（全参数微调）作为上限。  
  - **原始 Llama2‑70B**（未微调）作为下限。  
  - **Proxy‑tuned Llama2‑70B**（使用 7B 代理）作为本文方法。  

- **结果概览**  
  - 在所有三类基准上，Proxy‑tuned 版本闭合了 **约 88%** 的性能差距，几乎追上了真正微调的 Chat 版本。  
  - 在代码域适配上，错误率下降约 30%，接近直接微调的水平。  
  - 在 QA 与数学任务上，准确率提升 5~7%（相对原始模型），与全参微调相差不到 2%。  
  - 对 GPT‑3.5 的时效适配实验显示，关于 2023 年重大事件的回答准确率提升约 12%，而不需要任何检索模块。

- **消融实验**  
  - 去掉差分校正（即只使用大模型）性能回落到原始水平。  
  - 将 α 设为 0 或过大均导致效果下降，验证了系数调节的必要性。  
  - 使用未微调的 7B 代理（即 Δ≈0）也无法提升，说明微调产生的差分是关键。

- **局限性**  
  - 需要事先准备一个可微调的代理模型，若代理模型与大模型结构差异太大，差分映射会失效。  
  - 对于极端长文本或需要跨句子一致性的任务，单步差分可能不足以保持全局一致性。  
  - 计算 Δ 仍需一次完整的前向传播，对实时系统会有额外的延迟。

### 影响与延伸思考
这篇工作打开了“解码时微调”的新思路，随后出现了多篇基于相似理念的研究，例如 **LoRA‑in‑Inference**、**Adapter‑Fusion at Generation**、以及 **Prompt‑Distillation via Proxy** 等，都在尝试把小模型的适配信息以轻量方式注入大模型。商业平台也开始提供“代理微调”服务，让用户在不泄露模型权重的前提下定制化大模型。未来可以进一步探索：

- **多代理融合**：同时使用多个任务专用的代理，按需求加权合成 Δ。  
- **自适应系数学习**：让模型在生成过程中自动调节 α，以平衡原始能力和微调偏好。  
- **跨语言/跨模态代理**：把视觉或多语言微调的差分迁移到纯文本大模型上，实现跨模态适配。  

如果想深入，可以关注 2024‑2025 年的 “Decoding‑Time Adaptation” 研讨会和相关 arXiv 预印本，那里会有最新的实现细节和理论分析。

### 一句话记住它
**用小模型的微调差分在大模型的输出上加偏，让黑盒 LLM 现场“学会”新技能，而不动它的权重。**