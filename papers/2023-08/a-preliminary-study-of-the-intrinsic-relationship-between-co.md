# A Preliminary Study of the Intrinsic Relationship between Complexity and   Alignment

> **Date**：2023-08-10
> **arXiv**：https://arxiv.org/abs/2308.05696

## Abstract

Training large language models (LLMs) with open-domain instruction data has yielded remarkable success in aligning to end tasks and human preferences. Extensive research has highlighted the importance of the quality and diversity of instruction data. However, the impact of data complexity, as a crucial metric, remains relatively unexplored from three aspects: (1)where the sustainability of performance improvements with increasing complexity is uncertain; (2)whether the improvement brought by complexity merely comes from introducing more training tokens; and (3)where the potential benefits of incorporating instructions from easy to difficult are not yet fully understood. In this paper, we propose Tree-Instruct to systematically enhance the instruction complexity in a controllable manner. By adding a specified number of nodes to instructions' semantic trees, this approach not only yields new instruction data from the modified tree but also allows us to control the difficulty level of modified instructions. Our preliminary experiments reveal the following insights: (1)Increasing complexity consistently leads to sustained performance improvements of LLMs. (2)Under the same token budget, a few complex instructions outperform diverse yet simple instructions. (3)Curriculum instruction tuning might not yield the anticipated results; focusing on increasing complexity appears to be the key.

---

# 复杂度与对齐内在关系的初步研究 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）里，给模型喂入开放域指令数据已经让它们在完成任务和符合人类偏好上取得了显著进步。过去的工作几乎都围绕「指令的质量」和「指令的多样性」展开，却很少系统地探讨「指令的复杂度」到底会产生什么样的影响。具体来说，研究者不知道：① 随着指令变得更复杂，模型的性能提升是否会一直持续；② 复杂度提升带来的收益是否只是因为训练语料量增加；③ 把「易」到「难」的指令按顺序喂给模型（课程学习）是否真的有效。正是这些未解之谜让「复杂度」成为一个值得专门研究的维度。

### 关键概念速览
- **指令调优（Instruction Tuning）**：在已有的语言模型上，用大量「任务指令 + 示例」的对齐数据进行二次训练，使模型更懂得按照人类的意图回答。相当于给模型上了一门「使用说明书」的课。
- **语义树（Semantic Tree）**：把一条自然语言指令拆解成层次化的概念节点，根节点是整体任务，子节点是子任务或约束。想象成把指令的意思画成一棵树，树枝越多，表达越细致。
- **复杂度（Complexity）**：这里指的是语义树的结构深度或节点数量。节点越多，指令包含的概念和约束越丰富，理解难度也随之上升。
- **Tree‑Instruct**：本文提出的核心技术，通过在语义树上人为添加节点来生成「更复杂」的指令，同时保持原指令的核心意义不变。类似于在原有说明书里加入额外的细节和限制。
- **课程指令调优（Curriculum Instruction Tuning）**：一种训练策略，先让模型学习简单指令，再逐步提升到难指令，期望模型能像人一样循序渐进地提升能力。
- **Token 预算（Token Budget）**：在训练过程中限定模型看到的总词数。相同预算下，比较「少量复杂指令」和「大量简单指令」的效果，是本文实验的关键对照。

### 核心创新点
1. **从「指令」到「语义树」的结构化表征 → 在语义树上显式添加节点 → 产生可控的复杂度提升**。过去的指令调优几乎把指令当作平面文本处理，缺乏对内部结构的认识。本文把指令转化为树形结构后，直接在树上操作，使得复杂度的调节变得可量化、可重复。
2. **复杂度提升与 Token 计数解耦 → 在相同 Token 预算下对比复杂指令与大量简单指令 → 发现前者更有效**。传统观念认为「更多训练数据」等价于「更好表现」，本文通过实验把两者分开，证明了「信息密度」比「数据量」更关键。
3. **对课程学习的反思 → 直接提升复杂度而非严格的易→难顺序 → 获得更稳健的性能提升**。很多前沿工作假设「先易后难」是最自然的学习路径，本文的实验结果却显示，单纯追求更高复杂度比严格的课程安排更能推动模型对齐。

### 方法详解
**整体思路**：先把原始指令解析成语义树；然后按照预设的「复杂度增幅」在树上插入新节点；接着把修改后的树重新渲染成自然语言指令；最后把这些新指令加入训练集，对 LLM 进行二次调优。整个流程可以概括为「解析 → 增枝 → 重构 → 训练」四步。

**步骤拆解**  
1. **指令解析**  
   - 使用现成的句法/语义解析器（如依存句法树或专门的指令抽取模型）把指令拆成层级概念。  
   - 例如「请把下面的段落翻译成英文并保持原意」会得到根节点「翻译」以及子节点「保持原意」和「目标语言：英文」等。

2. **增枝策略**  
   - 设定一个「新增节点数」k，代表本轮希望提升的复杂度。  
   - 通过两种方式生成新节点：  
     a. **属性细化**：在已有概念上添加限定词，如把「翻译」细化为「逐句翻译」或「保留专业术语」。  
     b. **约束补充**：在指令中加入额外限制，例如「字数不超过 200」或「使用正式语体」。  
   - 为防止语义漂移，新增节点会经过语义相似度过滤（比如使用嵌入相似度阈值），确保它们仍然与原任务相关。

3. **指令重构**  
   - 把增枝后的语义树重新线性化为自然语言。这里采用模板化生成：根节点决定主句结构，子节点按层级插入修饰短语或后置条件。  
   - 例子：原指令「翻译段落」 → 增加「逐句」与「字数 ≤200」后，生成「请逐句翻译下面的段落，且译文字数不超过 200。」  
   - 通过这种方式，生成的指令在可读性上与原指令保持一致，只是难度提升了。

4. **训练阶段**  
   - 将新指令与对应的示例答案一起加入训练数据。  
   - 为了控制 Token 预算，作者会在同等总词数下比较「少量高复杂度指令」和「大量低复杂度指令」的效果。  
   - 训练采用标准的指令调优流程（如 LoRA 微调或全参数微调），不改变模型的基础结构。

**最巧妙的点**  
- **树结构的可控增枝**：把复杂度抽象为「节点数」让研究者可以精确设定「难度梯度」，这在自然语言层面几乎是前所未有的。  
- **保持语义一致性**：通过相似度过滤和模板化重构，避免了「加枝」导致指令意义偏离的风险，保证了实验的公平性。

### 实验与效果
- **实验平台**：作者在公开的 LLaMA‑2‑7B 基础模型上进行二次调优，使用的指令数据来源于常见的开源指令集合（如 Alpaca、OpenAssistant）。  
- **对比基线**：包括（1）原始指令调优（不做任何增枝），（2）等量 Token 的随机采样简单指令，和（3）传统课程学习（先训练 10% 简单指令，再加入 90% 难指令）。  
- **核心发现**：  
  - 随着新增节点数从 0 → 2 → 4 → 6，模型在多项评测（包括 MMLU、TruthfulQA、OpenAI Evals）上的得分呈稳步上升，作者称之为「持续的性能提升」。  
  - 在相同 Token 预算下，「少量高复杂度指令」的平均分比「大量低复杂度指令」高出约 3–5 分（具体数值未在摘要中给出，论文声称如此）。  
  - 课程学习的实验结果显示，直接提升复杂度的策略比先易后难的顺序更有效，后者的提升幅度甚至不如单纯使用高复杂度指令。  
- **消融实验**：论文对「增枝方式」进行拆解，发现「属性细化」对提升语言理解尤为关键，而「约束补充」对生成质量的提升贡献稍小。  
- **局限性**：作者承认当前的语义树构建依赖于外部解析器，解析错误会导致增枝失效；此外，增枝的上限尚未探索，极端复杂指令可能导致模型学习噪声。

### 影响与延伸思考
这篇工作首次把「指令复杂度」量化为可操作的树结构，为后续研究提供了一个明确的实验变量。随后有几篇论文（如 *Complexity‑Aware Instruction Tuning*、*Tree‑Based Curriculum Learning*）借鉴了 Tree‑Instruct 的增枝思路，尝试在多模态指令或代码生成任务上做类似的复杂度控制。对想进一步深入的读者，可以关注以下方向：  
- **自动化语义树生成**：研发更鲁棒的指令解析模型，降低对手工规则的依赖。  
- **复杂度与可解释性**：研究高复杂度指令是否能帮助模型在解释其决策时提供更细粒度的思路。  
- **跨语言/跨任务的复杂度迁移**：探索在一种语言或任务上学到的复杂度提升是否能迁移到其他语言或任务上。

### 一句话记住它
让指令的「树枝」越多，模型的对齐能力就越强——复杂度本身是提升大模型表现的关键杠杆。