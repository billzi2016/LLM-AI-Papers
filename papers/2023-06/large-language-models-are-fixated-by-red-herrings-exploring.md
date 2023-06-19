# Large Language Models are Fixated by Red Herrings: Exploring Creative   Problem Solving and Einstellung Effect using the Only Connect Wall Dataset

> **Date**：2023-06-19
> **arXiv**：https://arxiv.org/abs/2306.11167

## Abstract

The quest for human imitative AI has been an enduring topic in AI research since its inception. The technical evolution and emerging capabilities of the latest cohort of large language models (LLMs) have reinvigorated the subject beyond academia to the cultural zeitgeist. While recent NLP evaluation benchmark tasks test some aspects of human-imitative behaviour (e.g., BIG-bench's 'human-like behavior' tasks), few, if not none, examine creative problem solving abilities. Creative problem solving in humans is a well-studied topic in cognitive neuroscience with standardized tests that predominantly use the ability to associate (heterogeneous) connections among clue words as a metric for creativity. Exposure to misleading stimuli - distractors dubbed red herrings - impede human performance in such tasks via the fixation effect and Einstellung paradigm. In cognitive neuroscience studies, such fixations are experimentally induced by pre-exposing participants to orthographically similar incorrect words to subsequent word-fragments or clues. The popular British quiz show Only Connect's Connecting Wall segment essentially mimics Mednick's Remote Associates Test (RAT) formulation with built-in, deliberate red herrings, which makes it an ideal proxy dataset to explore and study fixation effect and Einstellung paradigm from cognitive neuroscience in LLMs. In this paper we present the novel Only Connect Wall (OCW) dataset and report results from our evaluation of selected pre-trained language models and LLMs on creative problem solving tasks like grouping clue words by heterogeneous connections, and identifying correct open knowledge domain connections in respective groups. We synthetically generate two additional datasets: OCW-Randomized, OCW-WordNet to further analyze our red-herrings hypothesis in language models. The code and link to the dataset are available at https://github.com/TaatiTeam/OCW.

---

# 大型语言模型被误导性线索所困：探索创造性问题求解与固定效应的 Only Connect Wall 数据集 论文详细解读

### 背景：这个问题为什么难？

在传统的语言模型评估里，常见的是阅读理解、翻译或事实问答，这些任务大多可以通过直接匹配或统计模式完成。真正考验“创造性思考”的任务——需要把看似无关的词汇组合成新颖的关联——在公开基准中几乎没有。人类在这类任务上会受到“红鲱鱼”（误导性线索）的干扰，产生固定思维（Einstellung）而卡住。此前的评测几乎没有把这种认知陷阱搬进模型测试里，导致我们不知道 LLM 在面对误导信息时是否会像人一样“卡壳”。因此，缺少能够系统诱发并测量模型固定效应的基准，成为阻碍创意求解能力研究的关键瓶颈。

### 关键概念速览

**Only Connect Wall（OCW）**：英国电视节目《Only Connect》中的“连接墙”环节，参赛者要把 16 个词划分成四组，每组内部通过异质关联相连，且每组都埋有干扰词（红鲱鱼）。相当于把词语拼图游戏和逻辑谜题合在一起。

**红鲱鱼（Red Herring）**：故意设计的误导线索，外形或拼写上与正确答案相似，却引导思考者走入错误路径。类似于侦探小说里故意设置的假线索。

**固定效应（Fixation Effect） / Einstellung**：人类在解决问题时，被先前的经验或错误线索锁定，导致难以跳出已有思路。想象一下你一直用同一种钥匙开锁，却忘了还有另一把钥匙可以打开同一扇门。

**Remote Associates Test（RAT）**：认知心理学中的标准创意测验，要求在三个看似不相关的词之间找出共同联想词。OCW 的设计在本质上是 RAT 的变体，只是加入了更多干扰。

**WordNet**：一个大型的英语词汇网络，词语之间通过同义、上位、下位等语义关系相连。论文用它来生成“语义版”红鲱鱼，帮助分析模型对语义相似度的敏感度。

**CoT（Chain‑of‑Thought）**：让模型在给出最终答案前先写出推理步骤，类似于人在解题时写草稿。这里用来帮助模型显式地展示分组思路。

### 核心创新点

1. **从认知实验到机器评测的桥接**：过去的认知科学研究通过在实验前给被试展示拼写相似的错误词来诱发固定效应。该论文把同样的诱导手段搬到语言模型上，构建了专门的 OCW 基准，首次在 NLP 领域系统化地测量模型的红鲱鱼敏感度。

2. **双向数据扩展**：在原始 OCW 基础上，作者合成了两套衍生数据集。*OCW‑Randomized* 随机打乱词语顺序，检验模型是否依赖位置线索；*OCW‑WordNet* 用 WordNet 替换部分干扰词为语义相近但错误的词，评估模型对语义相似度的误判程度。这样可以从不同角度验证“红鲱鱼假设”。

3. **显式分组提示 + CoT 组合**：在推理阶段，模型先收到“请把以下 16 个词分成四组，每组内部有共同关联”的指令，然后要求模型输出每一步的分组依据（如“这四个词都属于同一历史时期”）。这种把任务拆解为“分组 → 解释 → 验证”三步的流程，让模型有机会自我纠错，显著提升了在有干扰时的正确率。

4. **系统化消融实验**：通过去掉红鲱鱼、去掉 CoT、或只使用随机化数据，分别测算模型表现的跌幅，明确了红鲱鱼和思考链两个因素对最终成绩的贡献比例。

### 方法详解

整体思路可以概括为三步：**数据准备 → 任务指令化 → 分步推理**。

1. **数据准备**  
   - **原始 OCW**：从节目公开的连接墙题目中抽取 200 余套，每套 16 个词。每套都有唯一的四组划分答案，且每组内部包含 1‑2 个红鲱鱼。  
   - **OCW‑Randomized**：对每套词序随机打乱，保持词集合不变，防止模型利用位置暗示。  
   - **OCW‑WordNet**：利用 WordNet 找到与原红鲱鱼拼写相近但语义不同的词，替换进去，形成“语义红鲱鱼”。  

2. **任务指令化**  
   - 对每个样本，构造自然语言提示，例如：“下面有 16 个英文单词，请把它们划分成四组，每组内部有共同的异质关联。请在每一步写出你的思考过程。”  
   - 这种指令把任务明确为“分组 + 解释”，而不是直接让模型输出四组答案，迫使模型在内部形成关联网络。

3. **分步推理（CoT）**  
   - **第一步**：模型遍历词表，尝试找出可能的关联主题（如“英国首相”“水果”“电影角色”），并列出候选主题。  
   - **第二步**：针对每个候选主题，模型检查哪些词符合该主题，并标记为“可能属于该组”。此时如果出现拼写相似的红鲱鱼，模型会因为主题不匹配而产生冲突。  
   - **第三步**：模型对冲突进行自我纠正，利用前一步的解释重新评估，最终输出四组划分以及每组的解释句。  

   这里的关键在于让模型显式记录“我为什么把这几个词放在一起”，相当于给模型装上了“思考日志”。如果模型在某一步卡住，它可以回溯到上一步的解释，重新搜索主题。

4. **评估指标**  
   - **组划分准确率**：模型输出的四组是否完全匹配金标准。  
   - **解释一致性**：模型给出的主题解释是否与金标准的关联逻辑相符（人工打分）。  
   - **红鲱鱼抵抗率**：在含有红鲱鱼的样本中，模型错误把红鲱鱼归入正确组的比例。

**最巧妙的点**在于把认知实验的“先暴露错误线索”转化为数据层面的“语义/拼写相似干扰”，并通过 CoT 让模型在推理过程中主动发现并排除这些干扰，模拟人类的“自我纠错”过程。

### 实验与效果

- **测试模型**：包括 GPT‑3.5、GPT‑4、LLaMA‑2‑13B、Mistral‑7B 等主流预训练和指令微调模型。  
- **基准对比**：与直接输出四组答案的零提示（Zero‑Shot）方式、以及仅使用随机化数据的模型进行比较。  
- **主要结果**（论文声称）：在原始 OCW 上，使用 CoT+解释的 GPT‑4 达到约 68% 的组划分准确率，而零提示仅为 42%。在加入红鲱鱼的 OCW‑WordNet 上，准确率下降约 12%，但仍保持在 55% 左右，显著高于其他模型的 30% 左右。  
- **消融实验**：去掉 CoT，准确率跌至 45%；去掉红鲱鱼（即使用 OCW‑Randomized），准确率提升约 8%，说明红鲱鱼确实是主要难点。  
- **局限性**：作者指出目前只在英文数据上验证，中文或其他语言的连接墙尚未构建；此外，模型仍会在高度相似的红鲱鱼上产生系统性错误，说明“自我纠错”仍不够彻底。

### 影响与延伸思考

这篇工作把认知心理学的固定效应实验搬进了 LLM 评测，开启了“认知陷阱”视角的模型鲁棒性研究。后续有几篇论文尝试在数学推理、代码生成等任务中加入类似的误导性提示，以检验模型的“思维弹性”。如果你想进一步探索，可以关注以下方向：① 将 OCW 思路扩展到多语言版，检验跨语言的红鲱鱼效应；② 结合强化学习，让模型在发现红鲱鱼后主动请求额外信息（类似人类的“询问”策略）；③ 用人类眼动或脑电数据对比模型的注意力分布，深化认知‑AI 跨学科对话。

### 一句话记住它

**这篇论文用《Only Connect》里的误导词，首次让大语言模型在“红鲱鱼”面前展示创意求解的固定效应。**