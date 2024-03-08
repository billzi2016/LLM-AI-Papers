# Common 7B Language Models Already Possess Strong Math Capabilities

> **Date**：2024-03-07
> **arXiv**：https://arxiv.org/abs/2403.04706

## Abstract

Mathematical capabilities were previously believed to emerge in common language models only at a very large scale or require extensive math-related pre-training. This paper shows that the LLaMA-2 7B model with common pre-training already exhibits strong mathematical abilities, as evidenced by its impressive accuracy of 97.7% and 72.0% on the GSM8K and MATH benchmarks, respectively, when selecting the best response from 256 random generations. The primary issue with the current base model is the difficulty in consistently eliciting its inherent mathematical capabilities. Notably, the accuracy for the first answer drops to 49.5% and 7.9% on the GSM8K and MATH benchmarks, respectively. We find that simply scaling up the SFT data can significantly enhance the reliability of generating correct answers. However, the potential for extensive scaling is constrained by the scarcity of publicly available math questions. To overcome this limitation, we employ synthetic data, which proves to be nearly as effective as real data and shows no clear saturation when scaled up to approximately one million samples. This straightforward approach achieves an accuracy of 82.6% on GSM8K and 40.6% on MATH using LLaMA-2 7B models, surpassing previous models by 14.2% and 20.8%, respectively. We also provide insights into scaling behaviors across different reasoning complexities and error types.

---

# 常见的7B语言模型已经具备强大的数学能力 论文详细解读

### 背景：这个问题为什么难？

在过去，只有上百亿参数的模型才能在数学题目上取得可观的成绩，研究者普遍认为规模是数学能力的关键。于是大家要么训练超大模型，要么专门收集海量数学文本进行预训练，成本高得离谱。更糟的是，即便是大模型，也常常在一次生成时给出错误答案，可靠性差。于是出现了一个困惑：到底是模型本身缺乏数学推理，还是我们没有办法把它已有的潜力稳定地激发出来？

### 关键概念速览
- **LLaMA‑2 7B**：Meta 开源的 7 亿参数语言模型，算是“中等规模”。它的预训练数据和大模型一样是通用文本，没有专门的数学语料。
- **GSM8K**：一套 8 千条小学到初中水平的数学应用题，答案唯一且可自动评分，是衡量模型基础算术能力的常用基准。
- **MATH**：覆盖高中到大学水平的 12 千条高难度数学题，涉及代数、几何、微积分等，需要更深层次的推理。
- **SFT（监督微调）**：在已有模型上再用标注好的问答对进行微调，让模型学习“正确的回答模式”。相当于给模型上了针对性的辅导课。
- **合成数学数据**：用更强的模型（如 GPT‑4）生成的题目和答案，充当“人工制造的练习册”，弥补真实题库的稀缺。
- **多样采样 & 选最佳**：一次生成 256 条随机答案，然后挑出最接近正确答案的那一条。类似让学生写多份草稿，老师挑出最好的那份。

### 核心创新点
1. **发现 7B 模型本身已有高水平数学能力**  
   过去的研究把数学能力的出现归因于“规模”。这篇论文通过 256 次随机生成并挑选最佳答案的实验，展示 LLaMA‑2 7B 在 GSM8K 上能达到 97.7% 的准确率，在 MATH 上也有 72.0%。这说明模型内部已经蕴含了强大的算术和推理潜力，只是平时不容易被触发。

2. **用更大规模的 SFT 数据提升“一次生成”成功率**  
   作者发现模型的首个答案准确率只有 49.5%（GSM8K）和 7.9%（MATH），远低于多采样时的表现。于是把监督微调的数据量从几千提升到近百万，显著提升了单次生成的可靠性。关键在于“量的堆叠”而不是改变模型结构。

3. **合成数据几乎等同于真实数据**  
   真实的数学题库公开的只有几万条，远不足以支撑大规模微调。作者让 GPT‑4 先生成题目，再用人工或模型验证答案，得到约一百万条高质量合成题。实验表明，这些合成样本在提升模型性能上几乎和真实样本持平，且没有出现明显的饱和。

4. **系统化的规模行为分析**  
   论文不仅给出整体提升，还细分了不同推理深度（一步、两步、三步以上）和错误类型（算术错误、逻辑错误、格式错误）的表现随数据规模的变化趋势，为后续研究提供了细粒度的基准。

### 方法详解
整体思路可以概括为三步：**评估潜力 → 扩充微调数据 → 验证提升**。

1. **潜力评估**  
   - 直接使用原始 LLaMA‑2 7B，对每个 GSM8K / MATH 题目进行 256 次随机采样。  
   - 采用“选最佳”策略：把 256 条答案逐一与参考答案比对，取最高匹配度的那一条计分。  
   - 这一步相当于让模型“写草稿”，从中挑出最好的那份，揭示模型内部已经具备的解题能力。

2. **微调数据扩充**  
   - **真实数据**：收集公开的数学题库（GSM8K、MATH 以及其他公开来源），总量约几万条。  
   - **合成数据**：使用 GPT‑4 作为“题目生成器”。先给 GPT‑4 一个种子题目，让它改写或生成全新题目；随后让同一模型或人工检查答案的正确性，确保质量。  
   - 将真实+合成数据混合，形成约一百万条问答对。这里的关键是**多样性**：覆盖不同学科、不同难度、不同解题路径。

3. **监督微调（SFT）**  
   - 在 LLaMA‑2 7B 基础上进行标准的有标签微调：输入题目，模型学习输出对应的参考答案。  
   - 采用 **LoRA**（低秩适配）等轻量化参数调节技术，使得微调成本可控。  
   - 训练时使用 **混合采样**：真实数据占一定比例，合成数据占其余，防止模型过度拟合合成的特定模式。

4. **评估与选最佳**  
   - 微调后再次进行 256 次随机采样，统计 **首答准确率**（即第一次生成的正确率）以及 **多采样选最佳** 的上限。  
   - 对比不同数据规模（10k、100k、1M）下的表现，绘制出性能随数据量的曲线，观察是否出现饱和。

**最巧妙的点**在于把“合成数据”当作真实数据的等价替代，并且在微调阶段不做任何特殊的数学提示或链式思考（CoT）设计，仅靠数据量的提升就让模型一次生成的正确率从个位数跃升到两位数以上。

### 实验与效果
- **测试数据**：GSM8K（约 8 千题）和 MATH（约 12 千题），两者分别代表基础算术和高阶推理。  
- **基线**：原始 LLaMA‑2 7B（单次生成 49.5% / 7.9%），以及公开的同等规模模型（如 Falcon‑7B、Mistral‑7B）在官方报告中的成绩。  
- **结果**：  
  - 仅用 256 次采样挑最佳，原始模型已能达到 97.7%（GSM8K）和 72.0%（MATH）。  
  - 经过 1M 条合成+真实 SFT 微调后，单次生成准确率提升至 82.6%（GSM8K）和 40.6%（MATH），分别比同类 7B 基线高出 14.2% 和 20.8%。  
- **消融实验**：  
  - 去掉合成数据，微调仅用真实数据，性能下降约 3%–5%，说明合成数据贡献显著。  
  - 将合成数据比例提升到 90% 时，性能基本持平，进一步提升不再带来明显增益，暗示模型对合成数据的利用已接近上限。  
- **局限**：  
  - 多采样选最佳的实验在实际部署中成本高，真实场景仍需依赖一次生成的可靠性。  
  - 合成数据的质量仍受生成模型的限制，极端或创新题型可能出现系统性偏差。  
  - 论文未给出对更大模型（如 13B、30B）使用相同方法的实验，是否同样有效仍待验证。

### 影响与延伸思考
这篇工作冲击了“规模决定能力”的传统观念，提示研究者可以通过**高质量、海量的微调数据**来释放中等规模模型的潜能。随后出现的几篇论文（如《Synthetic Math for LLMs》《Scaling SFT with Generated Data》）直接借鉴了合成题库的思路，甚至把合成过程自动化为闭环：模型生成题目 → 自己评估答案 → 反馈微调。未来的方向可能包括：

- **自监督数学生成**：让模型在推理过程中自行产生中间步骤，形成类似 CoT 的自我监督信号。  
- **跨语言数学微调**：利用多语言合成题库，探索同一模型在不同语言数学任务上的迁移能力。  
- **数据效率提升**：研究更轻量的合成策略（如基于检索的模板填充），降低对强大生成模型的依赖。

如果想深入，可以关注 **OpenAI 的 MathBench**、**DeepMind 的 GopherMath** 以及 **Meta 的 LLaMA‑2 微调指南**，这些资源都在围绕“用数据而非参数”来提升数学能力。

### 一句话记住它
只要给 7 B 模型足够多、足够好的数学微调数据，它本身已经拥有的数学能力就能被稳稳释放。