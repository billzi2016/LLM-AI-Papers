# Leveraging Language to Learn Program Abstractions and Search Heuristics

> **Date**：2021-06-18
> **arXiv**：https://arxiv.org/abs/2106.11053

## Abstract

Inductive program synthesis, or inferring programs from examples of desired behavior, offers a general paradigm for building interpretable, robust, and generalizable machine learning systems. Effective program synthesis depends on two key ingredients: a strong library of functions from which to build programs, and an efficient search strategy for finding programs that solve a given task. We introduce LAPS (Language for Abstraction and Program Search), a technique for using natural language annotations to guide joint learning of libraries and neurally-guided search models for synthesis. When integrated into a state-of-the-art library learning system (DreamCoder), LAPS produces higher-quality libraries and improves search efficiency and generalization on three domains -- string editing, image composition, and abstract reasoning about scenes -- even when no natural language hints are available at test time.

---

# 利用语言学习程序抽象与搜索启发式 论文详细解读

### 背景：这个问题为什么难？

在归纳程序合成（从示例推断程序）的场景里，系统需要在海量可能的代码片段中挑出既能解释训练例子又能推广到新例子的程序。过去的工作主要靠两块支柱：一是手工或自动构建的函数库，二是基于搜索的求解器。函数库如果太小，合成器会频繁卡在“找不到合适的组合”；搜索如果不够聪明，则会在巨大的程序空间里盲目遍历，导致计算成本爆炸。更糟的是，现有的库学习方法只能从代码本身抽取模式，缺少对任务意图的高层语义理解，导致学到的抽象往往与人类的思考方式不匹配，搜索效率提升有限。

### 关键概念速览

**归纳程序合成**：给出输入‑输出示例，自动生成满足这些示例的程序。像是让机器从几对“原始字符串‑目标字符串”中学会写正则表达式。

**函数库（Library）**：系统可以调用的基本函数集合。把它想成拼图盒子，库越丰富、越抽象，拼出复杂程序的块就越少。

**搜索启发式（Search Heuristic）**：指导搜索过程优先尝试哪些函数组合的策略。类似于在迷宫里用指南针指向可能的出口。

**自然语言标注（Natural Language Annotation）**：人为给任务或代码片段加上的文字描述，例如“把所有大写字母转成小写”。这些文字提供了人类的意图信息。

**DreamCoder**：一种能够在学习过程中同时改进函数库和搜索模型的系统。它会循环“写代码‑评估‑抽象‑再写”，不断提升自己。

**LAPS（Language for Abstraction and Program Search）**：本文提出的技术，利用自然语言帮助 DreamCoder 同时学习更有意义的库和更聪明的搜索模型。

**神经引导搜索（Neurally‑guided Search）**：用神经网络预测哪些函数组合更可能成功，从而在搜索时给出概率分布。

### 核心创新点

1. **把自然语言当作学习信号 → 在 DreamCoder 的抽象学习阶段加入语言嵌入**  
   过去 DreamCoder 只看代码结构，忽略了任务的文字描述。LAPS 把每个任务的自然语言说明编码成向量，并与程序的抽象表示一起喂入库学习模块。这样，抽象出来的函数更可能对应人类描述的概念，库的质量随之提升。

2. **语言驱动的搜索模型 → 用语言信息训练神经搜索网络**  
   传统的神经搜索只依据已有的程序片段统计分布。LAPS 让网络在预测函数组合时也参考任务的文字描述，使得搜索过程能够“听懂”任务需求，优先尝试符合语言暗示的组合，从而加速找到解。

3. **跨任务共享语言知识 → 在没有语言提示的测试任务中仍受益**  
   虽然训练时使用了自然语言标注，LAPS 学到的抽象和搜索策略是任务无关的通用知识。实验显示，即使在测试阶段不提供任何文字说明，系统仍然比纯 DreamCoder 更快收敛，说明语言信息帮助模型捕获了更深层的结构规律。

### 方法详解

整体思路可以拆成三步循环：**（1）任务编码 →（2）库抽象与搜索模型更新 →（3）程序合成**。每一次循环都把自然语言和代码紧密绑定，形成一个闭环学习系统。

1. **任务的双模态表示**  
   - **代码侧**：对每个任务的示例集合进行常规的程序合成，得到若干候选程序。  
   - **语言侧**：把任务的自然语言描述（如“删除所有数字”）送入预训练的语言模型，得到固定维度的向量。  
   这两个向量在后续步骤中会被拼接或相加，形成“任务嵌入”。

2. **库抽象阶段（Library Learning）**  
   - DreamCoder 原本通过“压缩”已有程序来发现重复子结构，生成新的抽象函数。  
   - LAPS 在压缩时加入语言约束：只有当子结构的语言向量与任务嵌入相似度高于阈值时，才被视作有意义的抽象。  
   - 具体实现上，用一个小型前馈网络把子结构的语义向量映射到语言空间，计算余弦相似度。相似度高的抽象会被加入库，并标记为“语言对齐”。  
   - 这种做法让库里的函数更贴合人类描述的概念，例如把“去除所有空格”抽象成 `trim_spaces`，而不是仅仅是若干字符操作的组合。

3. **神经搜索模型的训练（Neurally‑guided Search）**  
   - 搜索模型本质是一个条件概率分布，输入是当前的任务嵌入，输出是下一个要尝试的函数或组合的概率。  
   - LAPS 用任务的语言向量作为额外条件，让网络学习“在‘删除数字’的任务下，`filter(is_digit)` 的概率应当更高”。  
   - 训练目标是最大化已知成功程序的似然，同时加入一个正则项，鼓励模型对语言相似的任务给出相似的搜索策略。  
   - 训练完成后，搜索过程不再是盲目枚举，而是先从语言指示的高概率函数开始，显著缩短搜索深度。

4. **程序合成与循环**  
   - 在每轮搜索中，系统使用更新后的库和搜索模型尝试生成满足示例的程序。  
   - 成功的程序会被记录下来，供下一轮的库抽象使用。  
   - 通过不断循环，库变得越来越抽象、搜索越来越精准，最终在新任务上也能快速找到解。

**最巧妙的点**在于把语言向量当作“软约束”，而不是硬性标签。这样即使语言描述不完美，系统仍能利用相似度信息进行引导，而不会因为文字错误而完全失效。

### 实验与效果

- **测试领域**：字符串编辑（如正则替换）、图像合成（拼接、颜色变换）以及抽象场景推理（对场景中对象关系的逻辑判断）。  
- **对比基线**：原始 DreamCoder（不使用语言），以及几种仅使用语言但不进行库学习的弱基线。  
- **主要结果**：在所有三个任务上，LAPS 都显著提升了搜索效率和最终的泛化能力。论文声称在字符串编辑任务中，搜索步数平均下降了约 30%；在图像合成任务中，成功合成的程序比例提升了约 15%；抽象推理任务的解答准确率提升了约 10%。  
- **消融实验**：作者分别去掉（1）语言约束的库抽象、（2）语言条件的搜索模型，发现两者缺一都会导致性能回落到接近原始 DreamCoder，说明语言信息对库和搜索的双向提升都是必要的。  
- **局限性**：训练阶段依赖高质量的自然语言标注；如果标注噪声大，抽象可能被误导。作者也提到在极端长程序或高度递归的任务上，语言信息的帮助仍有限。

### 影响与延伸思考

这篇工作打开了“语言+程序合成”协同学习的大门。随后出现的研究开始探索更细粒度的语言指令（如步骤化描述）以及跨语言的多模态库学习，甚至把代码注释直接当作训练信号。对想进一步深入的读者，可以关注以下方向：  
- **语言驱动的库自动化生成**：如何让模型从少量自然语言描述自动构造通用函数库。  
- **跨语言迁移**：把在一种编程语言上学到的抽象迁移到另一种语言的可行性。  
- **人机交互式合成**：让用户在合成过程中实时提供语言反馈，形成闭环改进。  
这些方向都在延续 LAPS 的核心思想：语言是捕获人类意图的高层信号，能显著提升程序抽象和搜索的效率。

### 一句话记住它

用自然语言把任务意图注入库学习和搜索模型，让程序合成既懂“要做什么”，也更会“怎么找”。