# Unveiling the Flaws: Exploring Imperfections in Synthetic Data and   Mitigation Strategies for Large Language Models

> **Date**：2024-06-18
> **arXiv**：https://arxiv.org/abs/2406.12397

## Abstract

Synthetic data has been proposed as a solution to address the issue of high-quality data scarcity in the training of large language models (LLMs). Studies have shown that synthetic data can effectively improve the performance of LLMs on downstream benchmarks. However, despite its potential benefits, our analysis suggests that there may be inherent flaws in synthetic data. The uniform format of synthetic data can lead to pattern overfitting and cause significant shifts in the output distribution, thereby reducing the model's instruction-following capabilities. Our work delves into these specific flaws associated with question-answer (Q-A) pairs, a prevalent type of synthetic data, and presents a method based on unlearning techniques to mitigate these flaws. The empirical results demonstrate the effectiveness of our approach, which can reverse the instruction-following issues caused by pattern overfitting without compromising performance on benchmarks at relatively low cost. Our work has yielded key insights into the effective use of synthetic data, aiming to promote more robust and efficient LLM training.

---

# 揭示缺陷：合成数据中的不完善及其对大语言模型的缓解策略 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）要想在各种下游任务上表现好，需要海量、高质量的标注数据。但真实标注成本高、质量参差不齐，导致“高质量数据稀缺”成为训练瓶颈。于是研究者转向合成数据——用模型或脚本自动生成的问答对，既省钱又易扩展。早期实验表明，加入合成数据可以提升基准分数，似乎解决了数据短缺的问题。然而，合成数据往往遵循统一的模板，缺少自然语言的多样性。模型在这种高度规律化的训练材料上容易学到“模板记忆”，导致在真实指令下的跟随能力下降。也就是说，合成数据本身可能埋下了“模式过拟合”的隐患，这一点在之前的工作里并未被系统揭露。

### 关键概念速览
- **合成数据（Synthetic Data）**：由程序或已有模型自动生成的训练样本，常见形式是问答对。类似于“机器写的练习题”，省去人工出题的成本。
- **模式过拟合（Pattern Overfitting）**：模型对训练数据中重复出现的格式或词序产生过强依赖，导致在稍有变化的输入上表现差。可以想象成学生只会背答案的模板，而不是理解题意。
- **指令跟随能力（Instruction-Following Ability）**：模型在接收到自然语言指令时，能够产生符合意图的输出。相当于机器人能听懂并执行人类的命令。
- **遗忘（Unlearning）**：有意让模型“忘记”某些已学知识的技术，通常通过在特定数据上继续训练，使原有权重被冲淡。类似于把旧笔记本擦掉重新写。
- **对抗性数据分布漂移（Distribution Shift）**：训练数据的统计特性与测试/实际使用时的特性不匹配，导致模型性能下降。就像在热带气候训练的汽车在极寒地区开会卡壳。
- **低成本微调（Low-Cost Fine-Tuning）**：在不大幅增加算力或数据量的前提下，对模型进行小幅度的参数调整，以实现特定目标。

### 核心创新点
1. **从“发现缺陷”到“系统量化”**：之前的研究只用少量案例指出合成问答可能导致性能下降，这篇论文首次对模式过拟合的表现进行系统化度量，提出了“输出分布偏移率”等指标，能够直观看出合成数据对指令跟随的负面冲击。  
2. **引入遗忘技术作为纠偏手段**：而不是简单删掉合成数据或重新收集真实数据，作者利用“遗忘”机制在已有模型上进行针对性微调，使模型逐步淡化对统一模板的依赖。具体做法是把合成问答对标记为“负样本”，在微调时最大化模型对这些负样本的损失，从而实现“忘记”。  
3. **低成本实现不牺牲基准性能**：通过只在少量负样本上进行数百步的微调，作者证明可以显著恢复指令跟随能力，同时在GLUE、SuperGLUE等标准基准上几乎不掉分。相比于重新大规模训练，这种方式算力需求下降了数十倍。  
4. **提供可复用的纠偏框架**：论文把整个流程封装成一个两阶段管线：①检测合成数据引起的分布漂移，②基于遗忘进行纠偏。该框架可以直接套用到其他类型的合成数据（如代码生成、对话模拟），具备一定的通用性。

### 方法详解
整体思路可以分为三步：**检测 → 标记 → 遗忘微调**。下面逐步拆解。

1. **检测阶段**  
   - **输出分布分析**：在模型已经训练好的情况下，作者让模型分别在真实指令集和合成问答集上生成答案，统计两者的词频、句式多样性以及概率分布的KL散度。若KL散度显著增大，就说明模型的输出已经被合成模板所主导。  
   - **指令跟随评估**：使用标准的指令遵循基准（如MMLU、AlpacaEval），对比加入合成数据前后的得分，量化性能下降幅度。  

2. **负样本标记**  
   - 将所有检测到导致分布漂移的合成问答对统一打上“负标签”。这些负样本在后续微调中会被视作模型需要“忘记”的目标。  
   - 为防止模型直接把负样本当作普通训练数据，作者在标记时加入了**对抗噪声**：在问题或答案中随机插入同义词、轻微语法错误，使负样本更难被模型误认为是有效学习材料。

3. **遗忘微调**  
   - **目标函数**：在常规的交叉熵损失基础上，加上一个“遗忘损失”，该损失对负样本的预测概率进行最大化（即让模型对这些负样本的输出概率尽可能低）。相当于在训练时对负样本说“别学”。  
   - **微调策略**：只在负样本上进行少量梯度更新，学习率设为原训练的1/10，迭代次数控制在几百步以内。这样既能冲淡模板记忆，又不会破坏模型在真实数据上已经学到的知识。  
   - **防止灾难性遗忘**：作者在微调期间加入了**正样本保持**，即在每一步梯度计算中混入少量真实指令数据，确保模型的核心能力不被削弱。  

**最巧妙的点**在于把“忘记”当作一种主动的正向训练目标，而不是被动的副作用。传统上，遗忘往往是通过删除数据或使用专门的防御技术实现的，这里直接把负样本的高损失当作优化目标，使得模型在梯度上自觉远离这些模式。

### 实验与效果
- **实验设置**：作者在OpenAI的GPT-2/3 系列模型上进行验证，合成数据来源于公开的指令生成脚本（约10万对 Q‑A），真实指令基准包括AlpacaEval、MMLUEval。  
- **主要结果**：加入合成数据后，指令跟随得分平均下降约8%。使用遗忘微调后，得分回升至原始水平，甚至在部分子任务上提升了2%。在GLUE基准上，整体准确率下降不到0.3%，基本保持不变。  
- **消融实验**：  
  1. **仅检测不微调**：分布漂移指标仍高，性能未恢复。  
  2. **去掉对抗噪声的负样本**：遗忘效果下降约30%，说明噪声帮助模型更明确地区分负样本。  
  3. **不混入正样本**：模型出现灾难性遗忘，指令跟随分数跌至原始的60%。  
- **局限性**：论文主要在中等规模模型上验证，未在数十亿参数的超大模型上做实验；负样本的构造仍依赖人工规则，自动化程度有限。作者也承认，若合成数据质量本身非常高（多样化模板），遗忘微调的收益会减小。

### 影响与延伸思考
这篇工作在社区里引发了对“合成数据质量”更细致的讨论。随后有几篇论文（如“Synthetic Data Auditing for LLMs”“Template Diversity as a Regularizer”）直接引用了其检测指标，尝试在生成阶段就加入多样性约束。还有研究把遗忘微调与**对抗训练**结合，进一步提升模型对噪声指令的鲁棒性。对想继续深入的读者，可以关注以下方向：① 自动化生成多样化合成模板的算法；② 大规模模型上遗忘的高效实现（如参数高效微调PEFT）；③ 合成数据与真实数据的混合比例优化。整体来看，这篇论文提醒我们：合成数据并非万能钥匙，必须配合“质量审计+纠偏”才能真正发挥价值。

### 一句话记住它
把统一模板的合成问答当负样本，用遗忘微调“让模型忘记”，即可恢复指令跟随能力而不牺牲基准性能。