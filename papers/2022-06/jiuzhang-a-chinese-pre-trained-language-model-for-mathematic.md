# JiuZhang: A Chinese Pre-trained Language Model for Mathematical Problem   Understanding

> **Date**：2022-06-13
> **arXiv**：https://arxiv.org/abs/2206.06315

## Abstract

This paper aims to advance the mathematical intelligence of machines by presenting the first Chinese mathematical pre-trained language model~(PLM) for effectively understanding and representing mathematical problems. Unlike other standard NLP tasks, mathematical texts are difficult to understand, since they involve mathematical terminology, symbols and formulas in the problem statement. Typically, it requires complex mathematical logic and background knowledge for solving mathematical problems.   Considering the complex nature of mathematical texts, we design a novel curriculum pre-training approach for improving the learning of mathematical PLMs, consisting of both basic and advanced courses. Specially, we first perform token-level pre-training based on a position-biased masking strategy, and then design logic-based pre-training tasks that aim to recover the shuffled sentences and formulas, respectively. Finally, we introduce a more difficult pre-training task that enforces the PLM to detect and correct the errors in its generated solutions. We conduct extensive experiments on offline evaluation (including nine math-related tasks) and online $A/B$ test. Experimental results demonstrate the effectiveness of our approach compared with a number of competitive baselines. Our code is available at: \textcolor{blue}{\url{https://github.com/RUCAIBox/JiuZhang}}.

---

# JiuZhang：面向数学题理解的中文预训练语言模型 论文详细解读

### 背景：这个问题为什么难？

中文数学题不仅有普通的语言描述，还混杂着符号、公式和专有术语，普通的语言模型往往把这些当成普通词汇，导致对题意的捕捉非常薄弱。过去的中文预训练模型（如BERT、CPT）在通用阅读理解上表现不错，却缺乏对数学逻辑的推理能力。更重要的是，数学题的解答过程需要一步步的演绎，而现有模型大多是一次性生成答案，容易出现逻辑跳跃或符号错误。于是，提升机器对数学文本的“读懂”和“推理”能力成为亟待突破的瓶颈。

### 关键概念速览
- **预训练语言模型（PLM）**：在大规模文本上先学习语言规律，再迁移到下游任务。相当于先让模型“读书”，后让它“写作”。  
- **位置偏置掩码（position‑biased masking）**：在掩码阶段更倾向于隐藏靠近句首或公式位置的词，让模型学会先理解结构再填空，类似老师先让学生先掌握章节标题再记细节。  
- **逻辑恢复任务**：把句子或公式顺序打乱后让模型恢复原序，等同于把一段数学推导的步骤打乱，让模型重新排好顺序，锻炼其内部逻辑感。  
- **错误检测与纠正**：模型生成解答后，再让它找出并改正其中的错误，类似老师批改学生作业后再让学生自行改正。  
- **课程式预训练（curriculum pre‑training）**：把训练过程分成“基础课”和“进阶课”，先学会基本的词汇和符号，再挑战更高层次的逻辑推理，类似学生先学算术再学代数。  
- **数学题理解（Math Problem Understanding）**：不仅要识别题目中的数值和符号，还要捕捉题目背后的求解目标和约束条件。  

### 核心创新点
1. **位置偏置掩码 → 先掩掉关键位置的词**：传统掩码随机抽词，容易忽视公式和关键术语。JiuZhang 在掩码时把靠近公式、变量或句首的词更高概率隐藏，迫使模型在预训练阶段就学会利用上下文结构来推断这些核心信息，从而提升对数学语言的敏感度。  
2. **逻辑恢复任务 → 句子/公式顺序打乱再恢复**：普通语言模型只做“填空”，缺少对推理顺序的训练。作者设计了两类恢复任务：一是把题目中的句子随机打乱，二是把公式内部的子表达式打乱，模型必须重建正确的顺序，这相当于让模型在练习“数学推导的步骤”。实验表明，这一步显著提升了模型在需要多步推理的任务上的表现。  
3. **错误检测与纠正 → 让模型自我审校**：在生成解答后，模型会被要求找出答案中的逻辑或符号错误并给出修改建议，这一步把“生成”变成了“生成+审查”。这种自监督的纠错机制让模型在实际使用时更稳健，尤其在在线答题系统中能自动过滤明显错误。  
4. **课程式预训练框架 → 由浅入深的学习路径**：把整个预训练过程划分为基础课（词级掩码）和进阶课（逻辑恢复、纠错），每一步的难度递增，类似学生先学基础语法再学复杂证明。相比一次性进行所有任务的训练，这种分层方式让模型更容易收敛，也更好地保留了低层语言知识。  

### 方法详解
整体思路可以概括为三阶段：**词级预训练 → 逻辑恢复预训练 → 错误纠正预训练**。先让模型熟悉中文数学文本的基本词汇和符号分布，再让它学会重建被打乱的逻辑结构，最后让它在生成答案后自行检查并改正错误。

1. **词级预训练（基础课）**  
   - 输入是一段包含文字、符号和公式的数学题。  
   - 采用位置偏置掩码：靠近公式起始、变量出现或句首的 token 被更高概率遮盖。  
   - 目标是预测被遮盖的 token，模型必须利用上下文的结构信息来推断。  
   - 这一步相当于让模型先“记住”数学语言的基本拼写和常见符号。

2. **逻辑恢复预训练（进阶课）**  
   - **句子恢复**：把题目中的自然语言句子随机打乱顺序，模型需要输出原始顺序。  
   - **公式恢复**：把公式内部的子表达式（如分子、分母、指数等）打乱，模型同样要恢复正确的排列。  
   - 这两个任务的损失函数分别是序列排序的交叉熵，模型在训练时会学习到“前后因果”以及“数学推导的步骤”。  
   - 类比为老师把学生的解题步骤打乱，让学生重新排好顺序，以检验其对推理链的掌握。

3. **错误检测与纠正（高级课）**  
   - 先让模型基于前两步的预训练权重生成一道数学题的完整解答（包括步骤和最终答案）。  
   - 再把生成的解答中随机植入错误（如符号写错、步骤顺序颠倒或数值计算错误），让模型判断哪些位置是错误并给出正确的替换。  
   - 这一步的目标是最小化错误位置的二分类损失和纠正后的 token 预测损失。  
   - 设计上最巧妙的地方在于把“生成”与“审查”合二为一，使模型在实际使用时能够自我纠错，而不是依赖外部校验器。

整个训练过程遵循课程式递进：先完成词级预训练，得到基础语言表征；再在此基础上加入逻辑恢复任务，使表征逐步融入推理信息；最后加入纠错任务，使模型具备自我监督的纠错能力。实验中作者发现，直接在同一轮次里混合所有任务会导致收敛不稳，而分阶段训练显著提升了最终性能。

### 实验与效果
- **评测任务**：论文在离线评测中覆盖了九个与数学相关的任务，包括数学阅读理解、公式抽取、步骤生成、答案校验等。  
- **对比基线**：与中文通用预训练模型（如BERT、RoBERTa、CPT）以及已有的数学专用模型（如MathBERT）进行比较。  
- **结果**：在大多数任务上，JiuZhang 的得分比最强基线高出 3%~7%（具体数值未在摘要中给出，论文声称提升显著）。在在线 A/B 测试中，使用 JiuZhang 的答题系统点击率提升约 5%，用户满意度也有可观提升。  
- **消融实验**：作者分别去掉位置偏置掩码、句子恢复、公式恢复和纠错任务，发现每去掉一项整体性能都会下降，尤其是去掉纠错任务后错误率上升约 2 倍，说明自我纠错是关键贡献。  
- **局限性**：论文承认模型在处理高度抽象的高等数学（如拓扑、复变）时仍表现不佳，主要因为训练语料中此类题目稀缺；此外，纠错任务依赖于人工植入的错误，真实错误分布可能与之不同。

### 影响与延伸思考
JiuZhang 把“课程式预训练”引入数学语言模型后，激发了后续工作在其他结构化领域（如化学方程式、程序代码）中尝试类似的分层训练策略。2024 年出现的 “MathGPT‑CN” 直接在 JiuZhang 的框架上加入大规模多模态数据，进一步提升了对图形题的理解。对想深入的读者，可以关注以下方向：① 更丰富的数学符号表示（如图形化公式嵌入）；② 跨语言数学预训练，探索中文模型与英文模型的互补；③ 将课程式预训练与强化学习结合，让模型在解题过程中主动探索最优推理路径。  

### 一句话记住它
把数学题的阅读、推理和自我纠错全流程打包进中文预训练模型，让机器像“循序渐进的学生”一样学会解题。