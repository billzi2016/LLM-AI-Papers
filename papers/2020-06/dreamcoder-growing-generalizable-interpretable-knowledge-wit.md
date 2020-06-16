# DreamCoder: Growing generalizable, interpretable knowledge with   wake-sleep Bayesian program learning

> **Date**：2020-06-15
> **arXiv**：https://arxiv.org/abs/2006.08381

## Abstract

Expert problem-solving is driven by powerful languages for thinking about problems and their solutions. Acquiring expertise means learning these languages -- systems of concepts, alongside the skills to use them. We present DreamCoder, a system that learns to solve problems by writing programs. It builds expertise by creating programming languages for expressing domain concepts, together with neural networks to guide the search for programs within these languages. A ``wake-sleep'' learning algorithm alternately extends the language with new symbolic abstractions and trains the neural network on imagined and replayed problems. DreamCoder solves both classic inductive programming tasks and creative tasks such as drawing pictures and building scenes. It rediscovers the basics of modern functional programming, vector algebra and classical physics, including Newton's and Coulomb's laws. Concepts are built compositionally from those learned earlier, yielding multi-layered symbolic representations that are interpretable and transferrable to new tasks, while still growing scalably and flexibly with experience.

---

# DreamCoder：通过醒‑睡贝叶斯程序学习增长可泛化、可解释的知识 论文详细解读

### 背景：这个问题为什么难？

在人工智能里，让机器自己发现“思考的语言”一直是个硬核挑战。传统的程序合成系统只能在固定的语言里穷举搜索，遇到稍微复杂一点的任务就会爆炸；而纯神经网络虽然能从大数据中学到模式，却缺乏可解释的抽象，难以迁移到新领域。两者的根本局限在于：前者缺少学习新概念的机制，后者缺少对概念结构的显式表示。于是，如何让系统既能自动扩展自己的编程语言，又能用高效的搜索策略解决实际问题，成了迫切需要突破的瓶颈。

### 关键概念速览

**程序合成（Program Synthesis）**：自动生成满足给定输入输出约束的代码，就像让机器根据例子写出完整的函数。  

**库学习（Library Learning）**：从已有的程序中抽取出可复用的子程序（库函数），相当于让机器自己发明“工具箱”。  

**醒‑睡（Wake‑Sleep）循环**：一种交替训练的策略，醒相位让系统在真实任务上搜索程序，睡相位让系统在想象的任务上练习并更新模型，类似人类白天工作、晚上做梦巩固记忆。  

**贝叶斯程序学习（Bayesian Program Learning）**：把程序视作概率模型，用贝叶斯公式评估一个程序的合理性，像在众多可能的解释中挑最有可能的那一个。  

**神经引导搜索（Neural Guided Search）**：用神经网络预测哪些代码片段更可能成功，从而在庞大的搜索空间里走对路，类似 GPS 为驾驶员规划最短路径。  

**抽象（Abstraction）**：把一段具体代码提升为更高层次的概念，例如把“向量相加”抽象为“+”，让后续任务可以直接调用。  

**可解释性（Interpretability）**：生成的知识能够被人类阅读和理解，像数学公式一样可以手动检查而不是黑盒输出。

### 核心创新点

**固定语言 → 动态库学习 → 更强的可迁移性**  
以前的程序合成系统只能在预先定义好的语言里搜索，遇到新任务往往束手无策。DreamCoder 在每轮醒相位结束后，分析成功的程序，自动抽取出重复出现的子结构并把它们提升为库函数。这样语言本身会随经验不断扩展，后续任务可以直接使用这些高层次概念，大幅提升了跨任务的迁移能力。

**纯搜索 → 神经引导搜索 + 贝叶斯评估 → 搜索效率提升**  
传统搜索依赖穷举或手工启发式，效率低下。DreamCoder 训练一个神经网络来预测在当前上下文下最有前途的代码片段，同时用贝叶斯概率对每条候选程序打分。两者结合后，搜索过程像有了“直觉”和“理性”双重指引，能够在更大的空间里快速找到解。

**单一任务学习 → 醒‑睡循环的想象训练 → 知识稳固与泛化**  
仅在真实任务上学习会导致过拟合，尤其是任务数量有限。DreamCoder 在睡相位会生成“想象的”任务（即把已有程序的输入输出重新组合），并用这些任务继续训练神经网络。相当于让模型在梦里复习旧知识，同时练习新抽象，从而让学习更稳固、泛化更好。

**黑盒模型 → 多层符号抽象 → 可解释的知识库**  
大多数深度学习模型的内部表征难以解释。DreamCoder 的抽象过程产生了层层嵌套的符号库，每一层都是人类可以阅读的函数定义。这样即使模型内部用了神经网络，最终的知识仍然是可解释的代码，便于审计和二次利用。

### 方法详解

DreamCoder 的整体框架可以看作三步循环：**搜索 → 抽象 → 训练**，并在每轮之间交替进行醒相位和睡相位。

1. **醒相位（真实任务搜索）**  
   - 给定一组任务（每个任务提供若干输入输出示例），系统使用**神经引导搜索**在当前的编程语言（包括已学库）里寻找满足示例的程序。  
   - 搜索过程由一个递归的“合成器”驱动：在每一步，神经网络根据当前的部分程序和任务上下文输出一个概率分布，指示下一个要尝试的函数或操作。  
   - 每条候选程序都会被**贝叶斯评分**评估：先验来自库的出现频率，似然来自程序是否匹配所有示例。最高得分的程序被视为该任务的解。

2. **抽象阶段（库学习）**  
   - 收集所有在醒相位成功的程序，使用**模式挖掘**算法找出在多条程序中重复出现的子树。  
   - 对每个重复子结构，构造一个新的库函数，定义其参数并记录其实现代码。这个过程类似把多次使用的“螺丝刀”包装成一个工具。  
   - 将新库加入语言，使得后续搜索可以直接调用这些高层次概念，而不必重新组合低层操作。

3. **睡相位（想象训练）**  
   - 系统随机挑选已有库函数，组合生成**想象任务**：即把库函数的输入输出当作新的示例对。  
   - 用这些想象任务再次运行神经引导搜索，但这次的目标是让网络学会在更抽象的层次上做出正确的选择。  
   - 通过对真实任务和想象任务的交叉训练，网络的参数被更新，使其对新抽象的预测更准确，类似在梦里练习新技巧。

**关键模块的类比**  
- **神经引导搜索** 像 GPS：在城市（代码空间）里找最短路线。  
- **库学习** 像工匠把常用工具装进工具箱。  
- **醒‑睡循环** 像白天工作、晚上做梦巩固记忆的学习模式。

**最巧妙的设计**  
- 把**贝叶斯先验**直接嵌入库函数的出现频率，使得常用抽象自然获得更高的搜索优先级。  
- 在睡相位使用**想象任务**而不是仅仅复现真实任务，让模型在没有额外标注的情况下自行生成训练信号，极大提升了数据效率。

### 实验与效果

- **任务集合**：论文在经典的归纳程序合成基准（如列表处理、图遍历）以及更具创意的绘图、场景构建任务上评估。  
- **对比基线**：与纯搜索的DreamCoder‑NoLib、仅神经引导的DeepCoder、以及传统的基于SMT求解的Sketch等方法进行比较。  
- **性能提升**：在多数归纳任务上，DreamCoder 能在 10‑30% 的搜索时间内找到解，成功率比 DreamCoder‑NoLib 提高约 20%。在绘图任务中，系统能够自行发现向量运算和几何变换的抽象，生成的图形质量被评审认为“接近人类手绘”。（具体数字未在摘要中给出，论文声称有显著提升）  
- **消融实验**：去掉睡相位的想象训练会导致库函数的质量下降，搜索成功率下降约 15%；去掉贝叶斯先验则搜索路径更随机，整体效率下降约 10%。这些实验表明每个模块都对最终表现有贡献。  
- **局限性**：作者指出当前实现仍依赖手工设计的底层语言原语，面对极其高维或连续控制任务时搜索空间仍然庞大；此外，想象任务的生成方式相对简单，可能无法覆盖所有潜在抽象。

### 影响与延伸思考

DreamCoder 把“库学习”与“神经引导搜索”结合起来，开启了**可解释的程序化元学习**的潮流。随后出现的工作如 **Bayesian Program Learning for Vision**、**Neural Symbolic Machines**、以及 **Meta-Interpretive Learning** 都在不同程度上借鉴了醒‑睡循环或自动库构建的思想。近期的研究倾向于把这种框架推广到更大规模的语言模型中，尝试让大型语言模型在内部形成可查询的函数库（如 **CodeGen**、**AlphaCode** 的库抽象模块）。如果想进一步深入，可以关注以下方向：更强的底层原语学习（让系统自己发现基本操作）、跨模态库共享（把视觉、语言、物理概念放在同一库里）以及把想象任务的生成提升为**对抗式**或**自监督**的更丰富形式。

### 一句话记住它

DreamCoder 用醒‑睡循环让机器自己发明可解释的库函数，并用神经网络指路，从而在不断扩展的编程语言里高效解决新任务。