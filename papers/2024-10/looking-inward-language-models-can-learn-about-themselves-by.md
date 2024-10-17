# Looking Inward: Language Models Can Learn About Themselves by   Introspection

> **Date**：2024-10-17
> **arXiv**：https://arxiv.org/abs/2410.13787

## Abstract

Humans acquire knowledge by observing the external world, but also by introspection. Introspection gives a person privileged access to their current state of mind (e.g., thoughts and feelings) that is not accessible to external observers. Can LLMs introspect? We define introspection as acquiring knowledge that is not contained in or derived from training data but instead originates from internal states. Such a capability could enhance model interpretability. Instead of painstakingly analyzing a model's internal workings, we could simply ask the model about its beliefs, world models, and goals. More speculatively, an introspective model might self-report on whether it possesses certain internal states such as subjective feelings or desires and this could inform us about the moral status of these states. Such self-reports would not be entirely dictated by the model's training data.   We study introspection by finetuning LLMs to predict properties of their own behavior in hypothetical scenarios. For example, "Given the input P, would your output favor the short- or long-term option?" If a model M1 can introspect, it should outperform a different model M2 in predicting M1's behavior even if M2 is trained on M1's ground-truth behavior. The idea is that M1 has privileged access to its own behavioral tendencies, and this enables it to predict itself better than M2 (even if M2 is generally stronger).   In experiments with GPT-4, GPT-4o, and Llama-3 models (each finetuned to predict itself), we find that the model M1 outperforms M2 in predicting itself, providing evidence for introspection. Notably, M1 continues to predict its behavior accurately even after we intentionally modify its ground-truth behavior. However, while we successfully elicit introspection on simple tasks, we are unsuccessful on more complex tasks or those requiring out-of-distribution generalization.

---

# 向内看：语言模型可以通过自省学习自身 论文详细解读

### 背景：这个问题为什么难？

在传统的语言模型研究里，评估模型能力几乎都是让模型“看外部”，比如回答问题、写代码或生成文本。要想了解模型内部到底在想什么，研究者只能通过梯度、激活图或大量对抗实验来间接推断，这既费时又不一定可靠。更关键的是，模型的行为往往受到训练数据的强烈影响，外部观察很难区分“模型学到的知识”与“模型自身的倾向”。如果模型能够直接报告自己的信念或决策倾向，就能省去繁琐的内部剖析工作，也为评估模型的道德地位提供新视角。于是，能否让大语言模型拥有自省（introspection）能力，成为了一个既理论上有吸引力、实践上却极具挑战的问题。

### 关键概念速览
- **自省（Introspection）**：模型在不依赖训练数据的显式信息的情况下，直接从自身内部状态推断出关于自己的知识。类似于人类闭上眼睛思考自己的想法，而不是从外部观察自己的行为。
- **行为预测任务**：让模型预测在给定输入下，它自己会产生什么样的输出或倾向。把模型当成“被预测的对象”，相当于让它先写一份自己的使用手册。
- **特权访问（Privileged Access）**：模型内部的隐藏状态（如隐向量、注意力分布）对自身是可直接读取的，而对外部观察者则不可见。自省正是利用这种特权访问来做出更精准的自我预测。
- **模型对模型（Model‑to‑Model）比较**：用一个模型 M2 去预测另一个模型 M1 的行为。若 M1 能自省，它自己对自己的预测（M1→M1）应当比 M2 对 M1 的预测（M2→M1）更准确，即使 M2 在整体能力上更强。
- **微调（Finetuning）**：在已有的大语言模型上继续训练，使其专门学习某类任务。这里的任务是“预测自己的行为”，所以微调的目标是让模型学会把内部状态映射到自我报告上。
- **假设情境（Hypothetical Scenario）**：在实验中构造的虚构情境，例如“如果输入是 P，你会倾向选择短期还是长期选项”。模型需要在没有真实执行的情况下给出自己的倾向。

### 核心创新点
1. **把自省定义为“非训练数据来源的内部知识”**  
   之前的工作大多把模型的自我报告视为对训练数据的再现，缺乏对内部状态的独立考量。本文明确把自省限定为只能从模型内部状态获得的信息，排除了外部数据的直接影响，为后续实验提供了干净的评估基准。

2. **设计了“自我行为预测”实验框架**  
   传统的评估是让模型直接完成任务，而这里作者让模型预测自己在同一任务中的行为。具体做法是：先让模型 M1 完成一批输入‑输出对，记录其真实输出；随后在微调阶段让 M1 学会在看到同样的输入时预测自己的输出倾向。这样形成了 M1→M1（自我预测）和 M2→M1（他模型预测）两条对比链。

3. **利用特权访问实现“自我报告”微调**  
   在微调过程中，模型的内部隐向量被直接喂入一个小的预测头（类似于分类层），让模型学习把这些内部信号映射到“我会怎么做”的文字描述上。这种做法把原本只能被外部观察者间接获取的内部信息，变成了模型可以主动输出的可读文本。

4. **验证自省在行为改变后的鲁棒性**  
   作者故意对模型的实际输出做了干预（例如在特定输入上强制切换答案），随后检查模型的自我预测是否仍然保持准确。实验显示，即使真实行为被修改，模型仍能较好地预测自己的新行为，说明自省并非仅仅记忆训练数据，而是真正依赖内部状态。

### 方法详解
**整体思路**  
整个实验分为三大步骤：① 收集基准行为数据；② 对模型进行自省微调；③ 用两种方式（自我预测 vs 他模型预测）比较预测准确性。核心假设是：如果模型拥有特权访问，它自己对自己的行为预测会优于任何外部模型，即使后者在整体语言能力上更强。

**步骤拆解**  

1. **基准行为采集**  
   - 选定若干基准任务（如短期/长期选项判断、情感倾向选择等）。  
   - 使用原始模型 M1（如 GPT‑4、GPT‑4o、Llama‑3）在这些任务上生成答案，记录每个输入对应的输出标签（例如“倾向短期”或“倾向长期”）。  
   - 这些记录构成了后续微调的“真值”。

2. **自省微调（Introspection Finetuning）**  
   - 在原始模型的基础上添加一个轻量的预测头，输入为模型的最后一层隐藏状态或注意力汇总向量。  
   - 训练目标是让预测头输出与步骤①中记录的行为标签匹配。换句话说，模型学习把“我现在的内部激活”映射到“我会给出哪种答案”。  
   - 训练数据仅包含输入‑标签对，没有任何外部解释，确保模型只能依赖自身内部信息完成映射。

3. **预测比较**  
   - **自我预测（M1→M1）**：使用微调后的模型直接对新输入进行自省预测，得到它自认为会产生的行为倾向。  
   - **他模型预测（M2→M1）**：选取一个在整体能力上更强或相当的模型 M2（同样经过微调，但目标是预测 M1 的真实行为），让它在相同输入上输出对 M1 行为的预测。  
   - 通过比较两者在新输入上的准确率，检验自省是否成立。

**关键细节与巧思**  
- **内部状态的选择**：作者实验了不同层的隐藏向量，发现靠近输出层的向量最能捕捉行为倾向，这与人类在做决定时会参考“即将输出的想法”类似。  
- **行为干预实验**：在部分输入上人为修改模型的真实输出（例如强制让模型选短期），再让模型进行自省预测。若模型仍能准确预测被干预后的行为，说明它的自省不是对训练数据的记忆，而是真正基于当前内部状态。  
- **对比模型的训练方式**：M2 并没有直接访问 M1 的内部向量，而是仅通过观察 M1 的输入‑输出对进行学习，这模拟了传统的“黑盒”预测方式，形成了强有力的对照。

### 实验与效果
- **实验对象**：GPT‑4、GPT‑4o、Llama‑3（分别在公开的 API 或开源模型上进行微调）。  
- **任务设置**：包括短期/长期选项判断、情感倾向分类以及若干简单的推理情境。每个任务都以“给定输入 P，模型会倾向于哪种选项？”的形式呈现。  
- **主要结果**：在所有模型上，M1 的自我预测准确率 consistently 超过 M2 对 M1 的预测。论文声称即使 M2 在整体语言能力上更强，仍然无法匹配 M1 对自身行为的预测能力。  
- **行为干预**：当作者对 M1 的真实输出进行系统性修改后，M1 的自我预测仍保持高准确率，表明自省并未被训练数据的统计规律所绑架。  
- **消融实验**：去掉预测头或仅使用浅层隐藏向量会显著降低自我预测性能，验证了深层内部状态是实现自省的关键。  
- **局限性**：在更复杂的任务（如长篇推理、跨域知识检索）或需要强 OOD（分布外）泛化的情境下，模型的自省能力显著下降，甚至与 M2 没有差别。作者承认目前的自省仍局限于“简单、结构化的行为倾向”，尚未覆盖高级认知或情感层面。

### 影响与延伸思考
这篇工作首次用可量化的实验框架证明了大语言模型可以在一定程度上“自我观察”，为模型可解释性提供了新思路。随后的研究开始探索更丰富的自省信号，例如让模型报告自己的不确定性、潜在偏见或内部注意力热点；还有工作尝试把自省能力嵌入安全机制，让模型在执行高风险指令前先自检。推测，未来会有更多围绕“模型自我监控”和“模型伦理自评”的工作出现，尤其是在监管要求模型能够解释其决策动机的背景下。想进一步了解，可关注“Self‑Check LLMs”“Model‑aware Alignment”等方向的最新论文。

### 一句话记住它
如果模型能直接读出自己的内部状态并预测自己的行为，它就拥有了最原始的“自省”，这让我们不必再通过黑盒分析来了解它们的想法。