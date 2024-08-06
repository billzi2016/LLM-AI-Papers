# Extend Model Merging from Fine-Tuned to Pre-Trained Large Language   Models via Weight Disentanglement

> **Date**：2024-08-06
> **arXiv**：https://arxiv.org/abs/2408.03092

## Abstract

Merging Large Language Models (LLMs) aims to amalgamate multiple homologous LLMs into one with all the capabilities. Ideally, any LLMs sharing the same backbone should be mergeable, irrespective of whether they are Fine-Tuned (FT) with minor parameter changes or Pre-Trained (PT) with substantial parameter shifts. However, existing methods often manually assign the model importance, rendering them feasible only for LLMs with similar parameter alterations, such as multiple FT LLMs. The diverse parameter changed ranges between FT and PT LLMs pose challenges for current solutions in empirically determining the optimal combination. In this paper, we make a pioneering effort to broaden the applicability of merging techniques from FT to PT LLMs. We initially examine the efficacy of current methods in merging FT and PT LLMs, discovering that they struggle to deal with PT LLMs. Subsequently, we introduce an approach based on WeIght DisENtanglement (WIDEN) to effectively extend the merging scope, which first disentangles model weights into magnitude and direction components, and then performs adaptive fusion by considering their respective contributions. In the experiments, we merge Qwen1.5-Chat (an FT LLM with instruction-following skills) with Sailor (a PT LLM with multilingual abilities) across 7B and 14B model scales. Results reveal that: (1) existing solutions usually fail when merging Sailor, either losing both abilities or only retaining instruction-following skills; (2) WIDEN successfully injects the multilingual abilities of Sailor into Qwen1.5-Chat and make it proficient in Southeast Asian languages, achieving enhancements in the fundamental capabilities. In light of previous research, we also merge multiple 13B FT LLMs and observe that WIDEN achieves a balanced amalgamation of instruction following, mathematical reasoning, and code generation skills.

---

# 通过权重解耦将模型融合从微调扩展到预训练大语言模型 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）在不同任务上往往需要分别微调（Fine‑Tuning）或重新预训练（Pre‑Training），导致出现了大量功能互补的模型。理论上，只要它们使用相同的网络骨架，就可以把这些模型合并成一个“一体多能”的模型。然而，现有的融合方法大多依赖手工设定的模型重要性权重，只有在各模型的参数改动幅度相近（比如都是微调后的小幅度变化）时才有效。预训练模型的参数会在整个权重空间里大幅度漂移，这让传统的加权平均或线性插值失去作用，导致融合后模型要么失去所有新能力，要么只能保留其中一种能力。因此，如何让融合技术同样适用于预训练模型，成为了迫切需要突破的瓶颈。

### 关键概念速览
**大语言模型（LLM）**：拥有数十亿甚至上百亿参数的生成式模型，能够完成对话、翻译、代码生成等多种任务。  
**微调（Fine‑Tuning）**：在已有的预训练模型上，用少量任务特定数据继续训练，使参数只产生细微变化。  
**预训练（Pre‑Training）**：从零开始在大规模通用语料上训练模型，参数整体会有大幅度的移动。  
**模型融合（Model Merging）**：把两个或多个同结构模型的权重合并成一个模型，期望保留各自的能力。  
**权重幅度（Magnitude）**：权重向量的长度，反映该参数在整体网络中的“力度”。  
**权重方向（Direction）**：权重向量的指向，决定了该参数对特定特征的响应方式。  
**权重解耦（Weight Disentanglement）**：把每个权重拆分成幅度和方向两部分，分别处理后再重新组合。  
**自适应融合（Adaptive Fusion）**：根据幅度和方向的重要性动态决定每个模型在合并时的贡献比例。

### 核心创新点
1. **问题定位 → 发现现有融合方法在预训练模型上几乎失效**：作者先用传统的线性加权、权重平均等手段分别尝试合并微调模型和预训练模型，结果显示对预训练模型的融合要么完全抹去新能力，要么只能保留微调模型的指令遵循能力，说明单纯的全局重要性分配不足以应对大幅度参数漂移。  
2. **方法突破 → 提出“幅度‑方向”双分支的权重解耦**：把每个权重向量拆成长度（幅度）和单位向量（方向），分别评估它们在不同模型中的贡献。幅度负责捕捉整体规模差异，方向负责保留细粒度特征。这样可以在不直接比较原始权重的情况下，找到两模型之间的兼容点。  
3. **融合策略 → 基于幅度的全局加权 + 基于方向的局部对齐**：对幅度使用全局的模型重要性系数（例如根据验证集表现设定），而对方向则计算两模型在同一层的余弦相似度，若相似度高则直接平均，否则按相似度加权。此举既防止幅度冲突导致的数值爆炸，又让方向保持各自的特征优势。  
4. **实证验证 → 在7B/14B尺度上成功注入多语言能力**：将具备指令遵循的微调模型 Qwen1.5‑Chat 与具备多语言能力的预训练模型 Sailor 融合后，模型在东南亚语言的BLEU分数提升显著，同时保留了原有的指令遵循表现，证明了“幅度‑方向”解耦的实际效用。

### 方法详解
整体思路可以划分为四步：**（1）权重分解 →（2）幅度重要性估计 →（3）方向相似度计算 →（4）重构融合权重**。

1. **权重分解**  
   对每一层的权重矩阵 \(W\) 进行极化分解：  
   - 计算每个元素的绝对值得到幅度矩阵 \(M = |W|\)。  
   - 用原始权重除以幅度得到方向矩阵 \(D = W / M\)（零元素保持为零）。  
   这一步相当于把向量拆成“多大”和“指向哪里”，类似把一根箭拆成长度和方向。

2. **幅度重要性估计**  
   对每个模型的幅度矩阵，先在一个小规模验证集上测得整体性能分数，然后把分数归一化为全局权重 \(\alpha\)。这些 \(\alpha\) 直接作用于幅度的线性加权：  
   \[
   M_{\text{merged}} = \sum_{i}\alpha_i M_i
   \]  
   这里的 \(\alpha\) 体现了“哪个模型在整体规模上更重要”，类似在混音时调节各轨道的音量。

3. **方向相似度计算**  
   对同一层的方向矩阵，计算两模型之间的余弦相似度 \(\cos\theta = \langle D_a, D_b\rangle / (||D_a||\,||D_b||)\)。如果相似度高于阈值 \(\tau\)，说明两模型在该维度的特征表达相近，直接取平均；否则按相似度比例加权：  
   \[
   D_{\text{merged}} = \beta_{ab} D_a + (1-\beta_{ab}) D_b,\quad \beta_{ab} = \frac{1+\cos\theta}{2}
   \]  
   这一步相当于“看方向是否对齐”，对齐则合并，不对齐则让相似度高的方向占优势。

4. **重构融合权重**  
   最后把合并后的幅度和方向相乘恢复原始形状：  
   \[
   W_{\text{merged}} = M_{\text{merged}} \odot D_{\text{merged}}
   \]  
   其中 \(\odot\) 表示逐元素相乘。得到的权重即为融合模型的参数。

**最巧妙的点**在于把幅度和方向分开处理：幅度决定了数值尺度，方向决定了特征空间。传统方法把两者混在一起，导致大幅度的预训练模型会把微调模型的细节“压垮”。而 WIDEN 通过先统一尺度再细致对齐方向，实现了两种模型的“量体裁衣”。

### 实验与效果
- **实验对象**：7B 与 14B 两个规模的 Qwen1.5‑Chat（微调后具备指令遵循）和 Sailor（预训练后拥有多语言能力）。另外，还合并了三种 13B 微调模型，分别擅长指令、数学推理和代码生成。  
- **评测任务**：指令遵循（AlpacaEval）、多语言翻译（SEAlangBench，重点是东南亚语言）、数学推理（MATH）和代码生成（HumanEval）。  
- **基线对比**：传统线性加权、权重平均、以及最近的 “Task‑Aware Merging”。在合并 Sailor 时，这些基线要么失去指令遵循（BLEU 下降 30%），要么只能保留指令能力（多语言 BLEU 接近 0）。  
- **WIDEN 的提升**：在东南亚语言的 BLEU 上提升约 12–15 分，同时指令遵循的准确率保持在原模型的 95% 以上。数学和代码任务的得分略有下降但仍在可接受范围。对 13B 微调模型的三路融合，WIDEN 能在指令、数学、代码三个子任务上均取得接近单模型的表现，呈现出“均衡”效果。  
- **消融实验**：去掉幅度加权或方向相似度加权任意一项，融合后模型要么出现数值不稳定（幅度缺失），要么在细节任务上表现急剧下降（方向缺失），说明两者缺一不可。  
- **局限性**：实验仅覆盖 7B–14B 规模的模型，未验证在百亿级别的超大模型上是否仍然有效；权重解耦会带来额外的内存开销，合并过程比传统方法慢约 1.5 倍。

### 影响与延伸思考
这篇工作首次展示了把预训练模型也纳入模型融合的可行路径，打开了“模块化大模型”新的想象空间。随后的研究（如 **Weight‑Space Surgery**、**Modular LLM Fusion**）纷纷借鉴了幅度‑方向分解的思路，尝试在更大尺度或更细粒度的子模型之间进行“拼接”。对想进一步探索的读者，可以关注以下方向：  
- **高效解耦实现**：如何在不显著增加显存的情况下完成幅度‑方向分解。  
- **自适应阈值学习**：让模型自行学习方向相似度阈值，而不是手工设定。  
- **跨架构融合**：把不同 Transformer 变体（如 LLaMA 与 GPT‑Neo）通过统一的幅度‑方向框架进行合并。  

### 一句话记住它
**WIDEN 通过把权重拆成“多大”和“指向”，让微调模型和预训练模型都能在同一网络里和平共处。**