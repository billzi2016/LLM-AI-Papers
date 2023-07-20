# Multi-Method Self-Training: Improving Code Generation With Text, And   Vice Versa

> **Date**：2023-07-20
> **arXiv**：https://arxiv.org/abs/2307.10633

## Abstract

Large Language Models have many methods for solving the same problem. This introduces novel strengths (different methods may work well for different problems) and weaknesses (it may be difficult for users to know which method to use). In this paper, we introduce Multi-Method Self-Training (MMST), where one method is trained on the filtered outputs of another, allowing us to augment the strengths and ameliorate the weaknesses of each method. Using a 176B parameter model trained on both language and code, we show that MMST can 1) improve the less performant method (up to 30%) making the model easier to use, 2) improve the more performant method (up to 32.2%) making the model more performant, and 3) improve the performance of related but distinct tasks (up to 10.3%) by improving the ability of the model to generate rationales. We then conduct ablation analyses to explore why MMST works. We show that MMST generates more data than traditional self-training, but the improvement in performance is driven by the use of multiple methods. We also analyze prompt-engineering and anti-correlated performance between methods as means of making MMST more effective. We hope the evidence from our paper motivates machine learning researchers to explore ways in which advances in language models allow for new forms of training.

---

# 多方法自监督训练：用文本提升代码生成，反之亦然 论文详细解读

### 背景：这个问题为什么难？

在大模型时代，同一个输入往往可以用多种方式来回答——比如把一道编程题直接给出代码，或者先写思路再写代码。过去的训练流程只让模型学会一种“套路”，导致两类任务（自然语言和代码）之间的知识难以互相渗透。于是出现了两个痛点：一是用户不知道该选哪种生成方式；二是每种方式的表现都有上限，单一方法难以突破。要想让模型在文本和代码之间相互借力，需要一种能够让不同生成方式相互学习的训练机制。

### 关键概念速览
- **自监督训练（Self‑Training）**：模型先用已有数据生成伪标签，再把这些伪标签当作新数据继续训练，类似学生先做练习再让老师批改再复习。  
- **多方法（Multi‑Method）**：指对同一输入使用不同的提示或解题路径，例如“直接生成代码” vs “先写思路再写代码”。  
- **过滤（Filtering）**：在自监督生成的伪数据中挑出质量较高的样本，像筛选出合格的作业再交给老师批改。  
- **Rationale（推理过程）**：模型在给出答案前写出的解释或思路，类似解题时的草稿。  
- **Prompt‑Engineering**：设计输入提示的技巧，决定模型走哪条思路，就像老师给学生不同的题目要求。  
- **反相关性能（Anti‑Correlated Performance）**：两种方法在同一实例上表现相反，一种好另一种差，类似不同解法在不同题目上的优势互补。  

### 核心创新点
1. **方法间的自监督互教**  
   - 之前的自监督只让模型用自己产生的答案继续训练。  
   - 这篇把一种方法（比如直接代码生成）的过滤后输出喂给另一种方法（先写思路再写代码）进行微调。  
   - 结果是弱方法的性能提升最高可达 30%，强方法也能再提升约 32%，实现了“弱者学强者、强者再强”的双向增益。  

2. **利用多方法生成更多伪数据**  
   - 传统自监督受限于单一生成路径，产出的伪样本数量有限。  
   - 多方法并行生成，使得可供过滤的候选样本数量大幅上升，进而提供更丰富的训练信号。  
   - 实验表明，性能提升主要来源于方法多样性，而非单纯数据量的增加。  

3. **通过 Prompt‑Engineering 与反相关性提升效果**  
   - 作者发现，精心设计的提示可以让两种方法在同一输入上表现出强烈的反相关性。  
   - 这种“互补”关系让过滤过程更容易挑出高质量样本，从而让自监督更有效。  
   - 通过调节提示，实验得到的提升幅度比随机提示高出约 5%。  

### 方法详解
整体思路可以拆成四步：

1. **准备两套生成方式**  
   - 方法 A：直接让模型输出代码（Code‑Only）。  
   - 方法 B：让模型先写出思路（Rationale），再给出代码（Rationale‑Then‑Code）。  
   两者的提示模板不同，分别对应“直接写代码”和“先解释后写代码”两种思维模式。

2. **自监督生成伪标签**  
   - 用已经微调好的大模型（176B 参数）分别在同一批未标注的编程题目上跑两套方式，得到两组输出。  
   - 每组输出都经过过滤：比如检查代码是否能通过单元测试、思路是否符合常规逻辑等。

3. **跨方法微调**  
   - 将方法 A 过滤后的高质量代码样本作为训练数据，喂给方法 B（即让 B 学习直接代码的模式）。  
   - 同理，把方法 B 过滤后的“思路+代码”样本喂给方法 A（让 A 学习带有推理的写法）。  
   - 这一步是核心的“互教”，通过微调让每个方法吸收对方的优势。

4. **迭代与收敛**  
   - 交叉微调完成后，再次用两种方式生成新伪标签，重复过滤与跨方法微调，直至验证集上的性能不再提升。  

**关键细节**  
- **过滤标准**：代码必须通过自动化测试，思路必须包含关键关键词（如“循环”“条件判断”），这保证了伪标签的质量。  
- **提示设计**：作者实验了多种提示模板，发现让两种方法在同一输入上产生“反相关”表现（比如一种常错，另一种常对）时，过滤效果最佳。  
- **数据量**：因为两种方法并行生成，伪标签的总量是单方法的约 2‑3 倍，这为模型提供了更丰富的学习信号。  
- **最巧妙的点**：跨方法微调本质上是一种“跨域迁移学习”，但不需要额外的标注数据，只靠模型自身的生成能力完成。  

### 实验与效果
- **实验平台**：使用 176B 参数的 BLOOM‑style 大模型，训练数据包括公开的自然语言描述与对应代码（如 HumanEval、MBPP）。  
- **任务**：主要评估代码生成的准确率（通过单元测试的比例）以及思路生成的质量（人工评估一致性）。  
- **Baseline**：单方法自监督（仅 Code‑Only 或仅 Rationale‑Then‑Code）以及原始未微调模型。  
- **结果**：  
  - 对弱方法（Rationale‑Then‑Code）提升最高约 30%。  
  - 对强方法（Code‑Only）提升约 32.2%。  
  - 在相关任务（如代码解释、错误定位）上也有约 10.3% 的提升，说明模型的推理能力整体增强。  
- **消融实验**：  
  - 去掉跨方法微调，只保留单方法自监督，提升幅度下降到 5% 左右，验证“多方法”是关键因素。  
  - 替换过滤标准为仅检查代码可运行，性能下降约 8%，说明思路过滤同样重要。  
- **局限性**：  
  - 需要对大模型进行额外的微调，计算成本不低。  
  - 实验仅在 BLOOM‑176B 上完成，尚未验证在更小模型或不同架构上的可迁移性。  
  - 过滤规则依赖于自动化测试，若测试覆盖不足，可能引入噪声。  

### 影响与延伸思考
这篇工作打开了“多模态自监督互教”的思路，后续有研究开始探索：  
- 在多语言翻译中让不同翻译策略相互学习（如直译 vs 意译）。  
- 在图文生成里让“先描述后绘图”和“直接绘图”互相强化。  
- 将 MMST 与指令微调（Instruction‑Tuning）结合，尝试让模型在更广泛的任务上自我提升。  
如果想进一步深入，可以关注 **跨任务自监督**、**提示工程的反相关性设计** 以及 **大模型高效微调** 这几个方向。  

### 一句话记住它
让模型用一种生成方式的高质量输出去教另一种方式，既能把弱者拉上来，也能让强者更强——这就是多方法自监督训练的核心魔法。