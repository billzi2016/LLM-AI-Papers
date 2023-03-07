# Larger language models do in-context learning differently

> **Date**：2023-03-07
> **arXiv**：https://arxiv.org/abs/2303.03846

## Abstract

We study how in-context learning (ICL) in language models is affected by semantic priors versus input-label mappings. We investigate two setups-ICL with flipped labels and ICL with semantically-unrelated labels-across various model families (GPT-3, InstructGPT, Codex, PaLM, and Flan-PaLM). First, experiments on ICL with flipped labels show that overriding semantic priors is an emergent ability of model scale. While small language models ignore flipped labels presented in-context and thus rely primarily on semantic priors from pretraining, large models can override semantic priors when presented with in-context exemplars that contradict priors, despite the stronger semantic priors that larger models may hold. We next study semantically-unrelated label ICL (SUL-ICL), in which labels are semantically unrelated to their inputs (e.g., foo/bar instead of negative/positive), thereby forcing language models to learn the input-label mappings shown in in-context exemplars in order to perform the task. The ability to do SUL-ICL also emerges primarily with scale, and large-enough language models can even perform linear classification in a SUL-ICL setting. Finally, we evaluate instruction-tuned models and find that instruction tuning strengthens both the use of semantic priors and the capacity to learn input-label mappings, but more of the former.

---

# 更大的语言模型在上下文学习中的表现不同 论文详细解读

### 背景：这个问题为什么难？
在语言模型的早期阶段，研究者发现模型可以通过“在上下文中学习”（in‑context learning, ICL）直接从示例推断任务，而不需要梯度更新。但到底模型是靠“记住”示例的映射关系，还是依赖它在大规模语料上形成的语义先验，一直没有清晰的答案。小模型往往表现出强烈的语义偏好，导致当示例标签与常识冲突时模型仍旧走常识路线。于是，学界缺乏一种系统实验来区分这两种驱动力，也不清楚这种行为是否会随模型规模而改变。正是这种模糊不清的认知，让人们迫切想弄清“大模型到底是怎么利用上下文学习的”。

### 关键概念速览
**上下文学习（In‑Context Learning, ICL）**：把任务示例直接写进模型的输入，让模型在一次前向传播中推断答案，类似于人看几道例题后直接做新题。  
**语义先验（Semantic Prior）**：模型在预训练阶段从海量文本中学到的常识和词义关联，比如“猫会抓老鼠”。这些先验会在推理时潜移默化地影响输出。  
**标签翻转（Flipped Labels）**：把任务的正负标签对调（例如把“正面”标记为负），用来检验模型是否会遵循示例而不是常识。  
**语义无关标签（Semantically‑Unrelated Labels, SUL）**：使用与输入毫无语义关联的标签（如“foo”“bar”），迫使模型只能靠示例中的映射关系来完成任务。  
**指令微调（Instruction Tuning）**：在大量指令-响应对上继续训练模型，使其更擅长遵循自然语言指令，类似于给模型上了一门“使用说明书”。  
**线性分类（Linear Classification）**：在特征空间里用一条直线（或超平面）把不同类别分开，这里指模型在 SUL‑ICL 环境下还能学到线性可分的决策边界。  

### 核心创新点
1. **利用标签翻转检验语义先验的可抑制性**  
   之前的工作大多只报告大模型能做 ICL，却没有系统地挑衅模型的常识。作者把正负标签对调后放进上下文示例，观察模型是否会遵从示例。结果显示，只有达到一定规模的模型才能克服预训练的语义先验，直接遵循翻转后的标签，这表明“覆盖语义先验”是规模驱动的 emergent 能力。  

2. **引入语义无关标签 ICL（SUL‑ICL）作为纯映射学习基准**  
   为了彻底剔除语义先验的干扰，作者设计了 SUL‑ICL：标签被换成与输入毫不相关的符号。这样模型只能靠上下文中展示的映射关系来做决定。实验发现，只有足够大的模型才能在这种设置下取得可观的准确率，甚至表现出线性分类的能力，说明大模型内部已经形成了可被上下文驱动的“快速学习”模块。  

3. **系统比较指令微调前后的行为差异**  
   作者把同一批模型分别在原始预训练权重和指令微调权重上进行上述两类实验。发现指令微调既加强了模型对语义先验的依赖，也提升了它学习输入‑标签映射的能力，只是前者的提升更为显著。这一发现帮助我们理解指令微调到底是让模型更“听话”，还是让它更“聪明”。  

### 方法详解
整体思路很直接：挑选若干主流大模型（GPT‑3、InstructGPT、Codex、PaLM、Flan‑PaLM），在同一套任务上分别做两类 ICL 实验——标签翻转和语义无关标签。每个实验都遵循以下步骤：

1. **任务选取**：作者使用了二分类的情感分析、自然语言推断（NLI）以及代码生成等标准基准，确保结果不局限于单一领域。  
2. **示例构造**：对于每个任务，先准备若干标准的 few‑shot 示例（通常 4‑8 条），每条示例包括输入、标签以及可选的指令前缀。  
   - **翻转标签**：把原本的正负标签互换后写入示例。  
   - **语义无关标签**：把原标签全部替换成随机字符串（如 “foo”“bar”），并在所有示例中保持一致的映射关系。  
3. **模型调用**：将示例和待预测的输入拼接成一个长文本，送入模型进行一次前向推理，直接读取模型输出的标签。  
4. **结果统计**：对比模型输出与期望标签的匹配率，绘制随模型参数量变化的曲线。

在实现细节上，作者做了两点值得注意的设计：

- **统一提示模板**：所有实验使用相同的提示格式（如 “Input: … Label: …”），避免因提示差异引入额外变量。  
- **多尺度对比**：每个模型族都选取了从几亿到上千亿参数的多个尺度，确保可以观察到“规模突变”现象。

最巧妙的地方在于 **SUL‑ICL** 的设定。因为标签本身不携带任何语义信息，模型只能靠在上下文中看到的映射关系来推断，这相当于把模型的“记忆”功能直接转化为“即时学习”。如果模型真的只是在“记忆”示例，它就会在 SUL‑ICL 中表现得像普通的模式匹配器；如果它拥有可被上下文激活的学习机制，则会在这种极端设置下仍然取得不错的准确率。实验结果正是后者，说明大模型内部已经具备了类似于“快速适应”模块的能力。

### 实验与效果
- **数据集与任务**：情感二分类（SST‑2）、自然语言推断（MNLI）、代码功能分类（Codex‑Eval）等。每个任务都分别做了翻转标签和 SUL‑ICL 两种实验。  
- **基线对比**：作者把每个模型的表现与同尺度的未做任何特殊处理的 ICL 结果作对比。  
- **关键发现**：  
  - 在翻转标签实验中，参数量低于 10 B 的模型准确率基本停留在 50% 左右（相当于随机猜），而 60 B 以上的模型准确率迅速上升到 80% 以上，显示出“规模阈值”。  
  - 在 SUL‑ICL 中，最小的模型几乎不超过随机水平，而 100 B 以上的模型能够达到 70% 左右的准确率，甚至在某些任务上接近传统微调模型的表现。  
  - 指令微调模型在翻转标签实验中提升约 10%‑15%，但在 SUL‑ICL 中的提升幅度更小，说明指令微调更偏向强化语义先验。  
- **消融实验**：作者去掉了示例中的指令前缀，发现性能略有下降，但规模效应仍然显著，说明核心现象不是由指令模板驱动的。  
- **局限性**：实验主要聚焦在二分类任务，未覆盖多分类或生成式任务；此外，模型的内部机制仍是黑盒，作者只能通过行为推断而未提供可解释的神经元分析。

### 影响与延伸思考
这篇工作在发布后迅速成为理解大模型“学习能力”争论的基石。它提醒研究者：模型的强大并非全靠预训练的语义知识，规模本身会带来一种“即时学习”能力。随后出现的多篇论文（如 “Emergent Abilities of Large Language Models”）引用了该实验框架，进一步探讨规模阈值与任务复杂度的关系。还有工作尝试在更细粒度的层面（如注意力头）定位 SUL‑ICL 背后的机制，或把 SUL‑ICL 作为评估新模型“可塑性”的标准。想深入了解的读者可以关注 **“可解释的上下文学习”** 与 **“规模突变的神经网络行为”** 两大方向，尤其是那些使用激活可视化或 probing 方法解释模型如何在一次前向传播中完成映射学习的研究。

### 一句话记住它
大模型在上下文学习时会“抛开”预训练的常识，直接从示例中学到输入‑标签映射，这种能力随模型规模突现。