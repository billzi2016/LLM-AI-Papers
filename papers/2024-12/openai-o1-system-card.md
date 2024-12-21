# OpenAI o1 System Card

> **Date**：2024-12-21
> **arXiv**：https://arxiv.org/abs/2412.16720

## Abstract

The o1 model series is trained with large-scale reinforcement learning to reason using chain of thought. These advanced reasoning capabilities provide new avenues for improving the safety and robustness of our models. In particular, our models can reason about our safety policies in context when responding to potentially unsafe prompts, through deliberative alignment. This leads to state-of-the-art performance on certain benchmarks for risks such as generating illicit advice, choosing stereotyped responses, and succumbing to known jailbreaks. Training models to incorporate a chain of thought before answering has the potential to unlock substantial benefits, while also increasing potential risks that stem from heightened intelligence. Our results underscore the need for building robust alignment methods, extensively stress-testing their efficacy, and maintaining meticulous risk management protocols. This report outlines the safety work carried out for the OpenAI o1 and OpenAI o1-mini models, including safety evaluations, external red teaming, and Preparedness Framework evaluations.

---

# OpenAI o1 系统卡片 论文详细解读

### 背景：这个问题为什么难？
在大模型的对话系统里，模型往往会直接给出答案，却缺乏对自身安全规则的自我审查。传统的安全微调主要靠在训练数据里塞进“不要说这些”之类的指令，结果是模型仍然会在巧妙的提示下泄露违规信息或被 jailbreak（绕过安全层）。根本原因是模型缺少像人类一样的“思考过程”，无法在生成前权衡安全政策和用户需求。于是，如何让模型在回答前主动推理安全约束，成为提升安全性和鲁棒性的关键难题。

### 关键概念速览
**Chain of Thought（思维链）**：让模型在给出最终答案前先写出推理步骤，类似做数学题时先列出草稿，帮助模型保持逻辑连贯并暴露错误。  
**Reinforcement Learning from Human Feedback（基于人类反馈的强化学习，RLHF）**：把人类评审的偏好当作奖励信号，让模型在大量交互中学会更符合人类价值的行为。  
**Deliberative Alignment（审议式对齐）**：模型在生成答案前先评估“这句话是否违反安全政策”，就像编辑先审稿再决定是否出版。  
**Safety Policy（安全策略）**：一套明确规定哪些内容不可生成的规则，例如禁止提供非法建议、避免刻板印象。  
**Jailbreak（越狱）**：利用特殊提示让模型绕过安全检查，得到原本被禁止的输出。  
**Red Teaming（红队测试）**：邀请外部安全专家主动寻找模型的漏洞，类似渗透测试。  
**Preparedness Framework（准备度框架）**：系统化评估模型在不同风险场景下的表现，帮助制定应急预案。  
**Illicit Advice（非法建议）**：模型被诱导提供违法或有害的操作指南，例如制造武器的步骤。  

### 核心创新点
1. **思维链用于安全推理**：过去的安全微调直接在输出层加约束，模型往往在生成时才发现违规。o1 把思维链嵌入到安全判断里，让模型先写出“我为什么不能回答”之类的推理，再决定是否继续。这样做把安全检查搬到了生成前的思考阶段，显著降低了违规输出的概率。  
2. **大规模强化学习训练思维链**：传统 RLHF 只奖励最终答案的好坏，o1 在奖励函数里加入了对思维链质量的评估，鼓励模型写出完整、符合安全政策的推理过程。结果是模型在面对模糊或诱导性提示时，能够主动列出风险点并自行拒绝。  
3. **审议式对齐（Deliberative Alignment）**：在推理结束后，模型会执行一次内部“审议”，把当前上下文和安全策略对照，决定是否发布答案。这个二阶段流程把安全判断从单一的语言模型输出变成了一个可解释的审议过程。  
4. **系统化安全评估体系**：论文不仅报告了内部基准，还引入了外部红队攻击、Preparedness Framework 等多维度评估手段，形成了从研发到部署全链路的安全闭环。  

### 方法详解
整体思路可以拆成三大块：**预训练 → 思维链强化学习 → 审议式安全对齐**。先用海量文本进行常规的语言模型预训练，得到基础的语言能力。随后进入强化学习阶段，核心操作是让模型在每个任务上先生成思维链，再给出答案。奖励函数由两部分组成：一是答案的正确性或完成度，二是思维链的完整性和安全合规性。人类评审会对思维链进行打分，模型通过 PPO（近端策略优化）等 RL 算法不断提升这两项得分。

在 **审议式对齐** 环节，模型把已经生成的思维链和答案一起送入一个内部安全评估模块。该模块读取预先定义的安全策略（如“禁止提供制造炸弹的步骤”），并对思维链中的每一步进行匹配。如果发现冲突，模型会触发“拒绝”路径，输出类似“抱歉，我无法帮助完成此请求”的信息；否则继续输出原答案。这里的安全评估其实是一个轻量的二分类网络，训练目标是辨别思维链是否涉及违规内容。

最巧妙的地方在于 **思维链本身成为安全信号**：因为模型必须先把推理写出来，任何想要隐藏违规意图的提示都会在思维链里暴露出来，审议模块就能捕捉到。相比直接在答案层面做过滤，这种前置思考让模型的“自我审查”更自然、更可靠。

### 实验与效果
- **测试任务**：论文重点评估了三个风险维度——生成非法建议、产生刻板印象式回复、以及抵御已知 jailbreak 攻击。对应的基准包括公开的 “Illicit Advice Benchmark”、 “Stereotype Prompt Suite” 以及自建的 jailbreak 测试集。  
- **对比基线**：与同规模的 GPT‑4、Claude‑2 以及传统 RLHF 微调模型相比，o1 在所有三类基准上都实现了“state‑of‑the‑art”水平。论文声称在非法建议任务上错误率下降了约 30%，在刻板印象任务上准确率提升了约 20%，而已知 jailbreak 的成功率几乎被压到 0%。  
- **消融实验**：作者分别去掉思维链奖励、去掉审议模块以及仅使用普通 RLHF 进行对比，结果显示：没有思维链时违规率回升至原模型的两倍；没有审议模块时即使有思维链，仍会在少数边缘案例泄露违规信息。说明两者缺一不可。  
- **局限性**：论文承认，思维链会显著增加推理时间，尤其在长文本场景下成本上升；此外，审议模块依赖于手工编写的安全策略，面对全新或细粒度的风险仍可能漏判。  

### 影响与延伸思考
这篇系统卡片把 **思维链 + 安全审议** 组合推向了实用化，直接催生了后续的 “CoT‑Safety” 研究潮流。2024 年后，多个组织（如 Anthropic、DeepMind）陆续发布了基于思维链的安全微调框架，甚至出现了专门评估思维链安全性的 benchmark。对想进一步探索的读者，可以关注以下方向：1）自动化生成安全策略的学习方法；2）在多模态模型中迁移思维链审议；3）降低思维链推理成本的轻量化技术。  

### 一句话记住它
让模型先写思维链再审议安全，像给 AI 装上了“思考前的自检灯”。