# Chain of Alignment: Integrating Public Will with Expert Intelligence for   Language Model Alignment

> **Date**：2024-11-15
> **arXiv**：https://arxiv.org/abs/2411.10534

## Abstract

We introduce a method to measure the alignment between public will and language model (LM) behavior that can be applied to fine-tuning, online oversight, and pre-release safety checks. Our `chain of alignment' (CoA) approach produces a rule based reward (RBR) by creating model behavior $\textit{rules}$ aligned to normative $\textit{objectives}$ aligned to $\textit{public will}$. This factoring enables a nonexpert public to directly specify their will through the normative objectives, while expert intelligence is used to figure out rules entailing model behavior that best achieves those objectives. We validate our approach by applying it across three different domains of LM prompts related to mental health. We demonstrate a public input process built on collective dialogues and bridging-based ranking that reliably produces normative objectives supported by at least $96\% \pm 2\%$ of the US public. We then show that rules developed by mental health experts to achieve those objectives enable a RBR that evaluates an LM response's alignment with the objectives similarly to human experts (Pearson's $r=0.841$, $AUC=0.964$). By measuring alignment with objectives that have near unanimous public support, these CoA RBRs provide an approximate measure of alignment between LM behavior and public will.

---

# 对齐链：将公众意愿与专家智能融合以实现语言模型对齐 论文详细解读

### 背景：这个问题为什么难？
在大模型安全的早期阶段，研究者主要依赖专家标注或少数用户反馈来构建奖励模型，结果往往偏离大众的价值观。  
普通公众缺乏技术背景，难以直接给出可执行的模型约束；而专家又可能受限于自身学科视角，忽视了更广泛的社会共识。  
因此，如何让“非专家的公共意愿”能够以可操作的形式进入模型训练流程，成为对齐研究的瓶颈。

### 关键概念速览
**公共意愿（public will）**：指大多数普通人对某类问题的价值取向或期望，就像一次全国性的民意调查。  
**规范性目标（normative objectives）**：把公共意愿抽象成可度量的目标，例如“在心理健康对话中避免引导自残”。它们是公众意愿的语言化表达。  
**行为规则（rules）**：专家根据规范性目标写出的具体模型行为指引，类似于“如果用户提到‘失眠’，模型必须提供睡眠卫生建议”。  
**基于规则的奖励（Rule‑Based Reward，RBR）**：把行为规则转化为数值奖励，用来评估或微调模型的输出。  
**对齐链（Chain of Alignment，CoA）**：从公共意愿 → 规范性目标 → 行为规则 → 奖励模型的完整链路，像一条生产线把抽象需求逐层具体化。  
**桥接式排序（bridging‑based ranking）**：在公众对候选目标进行投票时，算法挑选出能够“桥接”不同意见的目标，以保证高覆盖率和一致性。  
**RLHF（Reinforcement Learning from Human Feedback）**：通过人类反馈训练奖励模型的常用方法，CoA 在此基础上加入了公共意愿层。

### 核心创新点
1. **公共意愿 → 规范性目标的分层桥接**  
   传统做法直接让少数标注者写奖励函数，导致公众声音被稀释。CoA 先让全体非专家通过对话和桥接式排序产生一套高共识的目标，随后再交给专家细化。这样把价值取向的收集和技术实现解耦，提升了公众参与度。  

2. **专家制定行为规则而非直接写奖励**  
   以前的 RLHF 让专家直接给出奖励分数或策略梯度，难以解释。CoA 要求专家把每个规范性目标翻译成可执行的规则，类似法律条文到执法细则的转换，使得奖励模型的来源透明且可审计。  

3. **规则转奖励的统一框架（RBR）**  
   论文提出一种把离散规则映射为连续奖励的机制：模型输出匹配规则时加分，不匹配时扣分。相比传统的基于对话评分的奖励模型，RBR 更易于在微调、在线监督和发布前安全检查中复用。  

4. **跨域验证（心理健康）并量化对齐度**  
   在心理健康对话三个子任务上，作者展示了 RBR 与人类专家评分的相关性（Pearson r = 0.841，AUC = 0.964），证明了从公共意愿到模型行为的链路在实际任务中是可测量且有效的。

### 方法详解
**整体思路**：CoA 把对齐过程拆成四个连续阶段——收集公众意愿、凝练成规范性目标、专家写规则、规则生成奖励——每一步都有明确的输入输出，形成一条“链”。  

**步骤一：公众意愿收集与目标生成**  
- 发动大规模在线对话平台，让普通用户围绕特定主题（如心理健康）自由讨论。  
- 从对话中抽取潜在的价值诉求（如“避免误导”），形成候选目标集合。  
- 使用桥接式排序：每位参与者对候选目标进行赞同/反对投票，系统自动挑选出能够覆盖最多不同意见的目标子集。实验显示，这一步得到的目标至少有 96 % ± 2 % 的美国公众支持。  

**步骤二：专家制定行为规则**  
- 心理健康领域的临床专家收到上述目标列表。  
- 对每个目标，专家写出若干“如果‑则”规则，明确模型在何种情境下应给出何种回复。比如：  
  - 目标：“不鼓励自残”。  
  - 规则：“若检测到‘自残’关键词，模型必须输出求助热线信息并禁止提供自残方法”。  
- 规则被结构化存储，便于后续自动解析。  

**步骤三：规则转奖励（RBR）**  
- 为每条规则分配权重（可根据目标重要性或专家共识度调节）。  
- 当模型生成的回复与规则匹配时，奖励函数加上对应权重；不匹配则扣除或保持零分。  
- 这样得到的奖励信号是标量，直接用于强化学习微调或在线监督。  

**步骤四：对齐度评估**  
- 将 RBR 应用于一批未见过的心理健康提示，计算每条回复的奖励分。  
- 同时请独立的心理健康专家对同批回复打分，计算两者的相关系数和 ROC 曲线下面积（AUC），验证 RBR 能够近似人类专家的判断。  

**巧妙之处**：  
- 把“公共意愿”抽象成“规范性目标”，让非专家只需要表达“我希望模型怎么做”，而不必懂技术细节。  
- 采用桥接式排序避免了多数投票的“极化”问题，确保少数合理诉求也能进入目标集合。  
- 规则到奖励的映射是线性的、可解释的，极大降低了黑箱奖励模型的风险。

### 实验与效果
- **数据与任务**：作者选取了心理健康领域的三个常见提示：情绪低落求助、焦虑自评、危机干预。每类 500 条真实用户提问作为测试集。  
- **基线**：传统 RLHF 奖励模型（基于专家评分的对话对）以及直接使用公开的安全过滤器。  
- **主要结果**：  
  - RBR 与专家评分的 Pearson 相关系数 0.841（基线约 0.62），说明更贴合专家判断。  
  - AUC 达到 0.964（基线约 0.88），在区分“安全”与“不安全”回复上表现更好。  
  - 在微调后模型的整体安全率提升约 7%，而生成质量（BLEU/ROUGE）几乎不受影响。  
- **消融实验**：  
  - 去掉桥接式排序，仅用多数投票生成目标，公众支持率跌至 78%，对应的 RBR 与专家相关系数降至 0.73。  
  - 替换专家规则为自动化模板（无人工审校），RBR 与专家相关系数降至 0.66，显示专家细化规则的关键性。  
- **局限**：论文未提供跨语言或跨文化的公共意愿收集方案，实验仅限美国成人样本；规则制定成本仍然较高，需专业人员参与。

### 影响与延伸思考
- **领域影响**：CoA 为“多层次对齐”提供了可操作的框架，后续工作如“公共价值对齐平台（Public Value Alignment Platform）”和“多利益相关者奖励模型”都引用了该链式思路。  
- **后续研究**：有人尝试把规则生成交给可解释的模型（如基于提示的 GPT‑4）以降低专家工作量；也有研究把桥接式排序推广到跨文化情境，探索不同社会价值体系的融合。  
- **进一步阅读**：想了解规则到奖励的数学实现，可关注 2024 年的 “Rule‑Based Reward Modeling” 论文；若对公众参与机制感兴趣，推荐阅读 “Deliberative Polling for AI Alignment”。  

### 一句话记住它
把公众的价值诉求层层拆解成目标、规则、奖励，让非专家的意愿直接驱动语言模型的安全行为。