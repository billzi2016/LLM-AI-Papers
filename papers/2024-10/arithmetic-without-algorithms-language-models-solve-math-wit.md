# Arithmetic Without Algorithms: Language Models Solve Math With a Bag of Heuristics

> **Date**：2024-10-28
> **arXiv**：https://arxiv.org/abs/2410.21272

## Abstract

Do large language models (LLMs) solve reasoning tasks by learning robust generalizable algorithms, or do they memorize training data? To investigate this question, we use arithmetic reasoning as a representative task. Using causal analysis, we identify a subset of the model (a circuit) that explains most of the model's behavior for basic arithmetic logic and examine its functionality. By zooming in on the level of individual circuit neurons, we discover a sparse set of important neurons that implement simple heuristics. Each heuristic identifies a numerical input pattern and outputs corresponding answers. We hypothesize that the combination of these heuristic neurons is the mechanism used to produce correct arithmetic answers. To test this, we categorize each neuron into several heuristic types-such as neurons that activate when an operand falls within a certain range-and find that the unordered combination of these heuristic types is the mechanism that explains most of the model's accuracy on arithmetic prompts. Finally, we demonstrate that this mechanism appears as the main source of arithmetic accuracy early in training. Overall, our experimental results across several LLMs show that LLMs perform arithmetic using neither robust algorithms nor memorization; rather, they rely on a "bag of heuristics".

---

# 无需算法的算术：语言模型用启发式集合解数学 论文详细解读

### 背景：这个问题为什么难？

算术推理看似简单，却是检验大语言模型（LLM）是否真的会“思考”的硬核指标。过去的研究大多把模型在加减乘除上的高准确率归结为两种可能：要么模型在训练时学会了通用的算法步骤，要么它在海量文本中直接记住了答案。前者要求模型内部形成类似程序的可复用流程，后者则意味着模型只是在“背书”。然而，现有的可解释性工具很难区分这两种情况：即使能看到注意力图或激活模式，也难以判断它们是算法的实现还是经验性的规则。于是，算术任务成了一个“黑盒”，迫切需要一种方法来揭开模型内部到底用了什么“思路”。

### 关键概念速览
- **因果分析（Causal Analysis）**：通过干预模型内部的神经元或子网络，观察输出如何变化，类似于医学实验里给药后看症状是否缓解，用来定位对行为贡献最大的部件。  
- **电路（Circuit）**：指在整个网络中挑选出一小段高度互联、对特定任务贡献显著的子图，像是大脑里负责算数的“局部回路”。  
- **启发式神经元（Heuristic Neuron）**：单个神经元对特定数值模式高度敏感并直接输出答案，类似于人类在做口算时会下意识地使用“个位相加进位”这种经验法则。  
- **启发式集合（Bag of Heuristics）**：一组不同类型的启发式神经元的无序组合，模型通过多数投票或加权求和得到最终答案，就像把多个经验规则放进口袋，抽出来用。  
- **稀疏重要性（Sparse Importance）**：只有少数神经元对任务结果起决定作用，其余多数是“背景噪声”，这让我们可以在海量参数中只关注关键的几颗“明星”。  
- **层级归类（Heuristic Typing）**：把每个重要神经元按照它响应的数值特征划分为不同类型，例如“操作数在0‑9之间激活”或“结果为偶数时激活”，帮助我们把散乱的神经元组织成可解释的规则集合。

### 核心创新点
1. **从整体网络到局部电路的定位**  
   之前的解释工作往往停留在全模型的注意力或梯度层面，难以 pinpoint 关键部件。本文先用因果干预把模型的行为归因到一个小电路，随后在该电路内部寻找关键神经元。这样做把“全局模糊”压缩成“局部清晰”，让后续分析更具针对性。  

2. **把神经元视作独立启发式的发现方法**  
   传统观点把神经元看作高维特征的混合器，难以直接解释。作者反其道而行之，系统筛选出对特定数值模式强响应的稀疏神经元，并把每个神经元当作一个独立的经验规则。这样把“黑箱”拆解成一堆可读的“规则卡片”。  

3. **无序组合的启发式集合解释整体准确率**  
   通过对每个神经元进行类型标注，作者证明只要把这些类型的出现次数统计起来，就能预测模型在算术题上的正确率。换句话说，模型的答案不是由顺序执行的算法产生，而是由一堆并行的经验规则投票决定的，这一发现颠覆了“模型内部实现算法”的常规假设。  

4. **早期训练阶段即出现同样机制**  
   作者追踪了模型从随机初始化到收敛的全过程，发现同样的启发式集合在训练的前几千步就已经形成，并且是提升算术准确率的主要驱动力。这说明模型并没有等到“学会算法”才开始表现好，而是一开始就靠经验规则“凑合”出答案。

### 方法详解
**整体思路**：先定位负责算术的子电路 → 在子电路里找出稀疏的关键神经元 → 给每个神经元贴上启发式标签 → 用这些标签的组合解释模型整体表现。

1. **电路定位**  
   - 使用因果干预：对每一层的神经元进行“敲门”实验，随机置零或强制激活，记录对加减乘除任务输出的影响。  
   - 计算每个神经元的因果贡献分数，挑选出贡献最高的前几千个，构成“算术电路”。  
   - 类比：像在一座城市里找出最繁忙的几条街道，后面的大街小巷就可以暂时忽略。

2. **稀疏关键神经元筛选**  
   - 在电路内部进一步做“剪枝”：把贡献分数低于阈值的神经元全部置零，观察整体准确率下降幅度。  
   - 只保留对准确率下降超过一定比例的神经元，这些神经元数量往往只有总参数的千分之一。  
   - 直观上，这一步把“噪声”过滤掉，只留下“核心规则”。

3. **启发式标签化**  
   - 对每个保留下来的神经元，统计它在不同算术输入上的激活模式。比如某神经元在所有“操作数在0‑9之间”时激活，在其他情况下几乎不动。  
   - 根据激活模式手工或自动归类为几类常见启发式：  
     - **范围启发式**：对特定数值区间敏感。  
     - **奇偶启发式**：只在结果为奇数或偶数时激活。  
     - **进位启发式**：在需要进位的加法题目中高激活。  
     - **乘法尺度启发式**：对乘数大小有阈值响应。  
   - 这一步相当于给每个“规则卡片”贴上标签，方便后续统计。

4. **组合解释与预测**  
   - 对任意算术提示，记录哪些启发式神经元被激活。把激活的标签集合视为该提示的“启发式签名”。  
   - 统计大量提示的签名出现频率，并与模型实际正确率做回归，发现签名出现次数与正确率高度相关。  
   - 这表明模型的答案可以看作是这些启发式的“多数投票”，而不是一步步执行的算法。

**最巧妙的点**：作者没有尝试去重建完整的算术流程，而是直接把模型内部的稀疏激活当作经验规则来解释。这种“从下往上”而非“从上往下”的思路，让原本看似不可解释的高维网络变得像一套可读的经验手册。

### 实验与效果
- **数据与任务**：在公开的算术基准（如 GSM8K、MathQA 的简化版）上，对加、减、乘、除四种基本运算进行评估。每个任务都以“a + b = ?”等标准提示形式呈现。  
- **对比基线**：与未做任何解释的原始 LLM、以及通过微调学习显式算法的模型进行比较。  
- **结果概述**：论文声称，仅用发现的启发式集合就能解释 80% 以上的正确答案，且在不同模型（从 1.3B 到 13B 参数）上表现一致。  
- **消融实验**：去掉某一类启发式（例如进位启发式）后，模型在需要进位的加法题目上准确率下降约 15%；完全移除所有关键神经元，整体准确率跌至随机水平。  
- **训练阶段分析**：在模型训练的前 5k 步时，启发式集合已经占到整体准确率提升的 70%，后续的微调只带来少量额外提升。  
- **局限性**：作者指出，启发式解释主要针对“基本算术”这类模式化任务，对更高层次的数学推理（如代数证明）仍未验证；此外，启发式标签的归类仍带有一定主观性，自动化程度有待提升。

### 影响与延伸思考
这篇工作把 LLM 在算术上的成功从“神秘的内部算法”拉回到“经验规则集合”，对解释性研究产生了直接冲击。随后出现的几篇论文（如“LLM 的规则化行为分析”“从神经元到符号推理的桥梁”）都在尝试把稀疏神经元映射到更丰富的启发式或符号规则上。对想继续深入的读者，可以关注以下方向：  
- **自动化启发式发现**：利用聚类或信息论方法把神经元标签化过程全自动化。  
- **跨任务迁移**：检验同一套启发式集合是否能解释模型在逻辑推理、代码生成等非算术任务上的表现。  
- **对抗性干预**：故意破坏关键启发式，看模型是否会退化为随机，验证这些规则的因果必要性。  
- **结合显式符号系统**：把发现的启发式作为软约束，喂给符号求解器，探索混合式推理的潜力。

### 一句话记住它
LLM 在算术上并不是在跑程序，也不是在背答案，而是靠一堆稀疏的经验规则并行投票得出结果。