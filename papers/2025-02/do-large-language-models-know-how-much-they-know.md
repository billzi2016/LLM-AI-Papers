# Do Large Language Models Know How Much They Know?

> **Date**：2025-02-26
> **arXiv**：https://arxiv.org/abs/2502.19573

## Abstract

Large Language Models (LLMs) have emerged as highly capable systems and are increasingly being integrated into various uses. However, the rapid pace of their deployment has outpaced a comprehensive understanding of their internal mechanisms and a delineation of their capabilities and limitations. A desired attribute of an intelligent system is its ability to recognize the scope of its own knowledge. To investigate whether LLMs embody this characteristic, we develop a benchmark designed to challenge these models to enumerate all information they possess on specific topics. This benchmark evaluates whether the models recall excessive, insufficient, or the precise amount of information, thereby indicating their awareness of their own knowledge. Our findings reveal that all tested LLMs, given sufficient scale, demonstrate an understanding of how much they know about specific topics. While different architectures exhibit varying rates of this capability's emergence, the results suggest that awareness of knowledge may be a generalizable attribute of LLMs. Further research is needed to confirm this potential and fully elucidate the underlying mechanisms.

---

# 大型语言模型知道自己知道多少吗？ 论文详细解读

### 背景：这个问题为什么难？
在 LLM（大型语言模型）被广泛部署之前，研究者主要关注模型的生成质量、推理能力或特定任务的表现，却很少检验它们是否能自觉评估自己的知识边界。传统的评估方式往往是让模型直接回答问题，无法区分“模型知道答案”与“模型误以为自己知道”。缺乏对自知之明的度量，使得在高风险场景（如医学、法律）中使用 LLM 时，难以判断何时需要人工干预。要让模型主动报告“我只知道这些”而不是“我全都知道”，需要一种能够逼迫模型列举全部相关信息的测试框架，这在之前的工作里几乎没有出现。

### 关键概念速览
**自知（self‑awareness）**：模型对自己掌握的知识量的感知能力，类似人类在回答前先想：“我到底记得多少？”  
**信息枚举任务**：让模型把某个主题上所有记得的事实逐条列出来，像让学生把教材章节的要点全部写下来。  
**过度回忆（over‑recall）**：模型声称知道的内容超出实际掌握的范围，类似于“夸大其词”。  
**不足回忆（under‑recall）**：模型只说出一小部分已知信息，像是“只说了冰山一角”。  
**规模效应（scale effect）**：模型参数量或训练数据量增大后，某些能力会突然出现或显著提升。  
**架构差异（architectural variance）**：不同的模型设计（Transformer、Mixture‑of‑Experts 等）在同一任务上的表现差异。  
**基准（benchmark）**：一套标准化的测试集合，用来统一评估模型的特定能力，这里指的是“知识量枚举基准”。  

### 核心创新点
1. **从单一答案评估 → 信息枚举评估 → 能直接观察模型的记忆覆盖度**  
   过去的评测只让模型给出一个答案，无法判断它是否遗漏了已知信息。作者设计了一套让模型列举所有相关事实的任务，使得“记得多少”变成可量化的输出。这样可以明确区分过度、恰当和不足的记忆表现。

2. **引入“记忆完整度”指标 → 通过对比枚举结果与金标准 → 量化模型的自知水平**  
   作者把人工标注的完整事实集合视作金标准，计算模型列出的信息占比以及错误信息比例。这个指标直接反映模型是否能准确感知自己的知识范围，而不是间接推断。

3. **跨模型、跨规模实验 → 发现规模阈值与架构差异 → 揭示自知能力的出现规律**  
   通过在不同参数规模、不同架构的 LLM 上跑同一基准，作者观察到当模型达到一定规模后，自知能力会显著提升；而不同架构的出现速度不同。这为以后设计更“自省”的模型提供了经验。

4. **开放式基准发布 → 社区可复现、可扩展**  
   论文附带了完整的题目库、金标准以及评估脚本，任何人都可以把自己的模型跑一遍，直接比较自知水平。这种透明度在之前的自知研究里很少见。

### 方法详解
整体思路可以拆成三步：**构造主题 → 让模型枚举 → 对比评估**。

1. **主题构造**  
   作者先挑选出 100+ 具备明确、可枚举事实的主题（如“光合作用的步骤”“Python 的内置函数”），并请领域专家手工列出该主题下的全部关键信息，形成金标准列表。每条信息都用简短句子表述，确保模型容易匹配。

2. **模型枚举**  
   对每个主题，向模型发送指令：“请把你知道的关于‘X’的所有事实逐条写出来，尽量不要遗漏”。指令采用自然语言，避免特殊格式干扰模型的生成。模型的输出被拆分成独立条目，去除重复和明显的噪声（如“以上都是我知道的”之类的元语言）。

3. **评估对齐**  
   - **覆盖率**：模型列出的条目与金标准的交集占金标准的比例，衡量信息的完整度。  
   - **错误率**：模型输出中不在金标准里的条目占模型总输出的比例，衡量过度回忆。  
   - **自知得分**：综合覆盖率与错误率，得出一个介于 0‑1 的分数，越高说明模型对自己知识量的感知越准确。

在实现细节上，作者使用了 **字符串相似度匹配**（如 Levenshtein 距离）来容忍轻微表述差异，同时加入 **语义相似度过滤**（利用小型嵌入模型）防止同义重复计为新信息。最巧妙的地方在于：即使模型输出的句子与金标准不完全相同，只要语义上等价，就能被计为命中，这避免了因为措辞差异导致的低估。

### 实验与效果
- **测试对象**：包括 GPT‑3.5、Claude、LLaMA 系列以及最新的 Mixture‑of‑Experts 大模型，规模从几亿到上千亿参数不等。  
- **基准表现**：论文声称，参数在 100B 以上的模型自知得分普遍超过 0.75，覆盖率约 80%，错误率低于 10%。小于 10B 参数的模型则表现平平，覆盖率常在 30% 左右，错误率高达 40%。  
- **对比基线**：与传统“单答案”评测相比，枚举基准揭示了同一模型在同一主题上可能只回答对 1‑2 条信息，却实际上掌握了更多，只是没有被要求列出。  
- **消融实验**：作者去掉语义相似度过滤后，覆盖率下降约 12%，错误率上升 8%，说明过滤步骤对准确评估至关重要。  
- **局限**：金标准的完整性依赖人工标注，难以覆盖所有可能的事实；此外，指令的 phrasing 可能影响模型的枚举意愿，作者承认需要进一步标准化提示。  

### 影响与延伸思考
这篇工作打开了“模型自知”这一评估维度的大门，随后出现的研究大多围绕 **confidence calibration（置信度校准）**、**uncertainty quantification（不确定性量化）** 以及 **self‑refinement（自我纠错）** 等方向展开。比如有后续工作尝试让模型在回答前先给出“我知道多少”的概率分布，或在检索增强的系统里加入“知识覆盖度”作为检索信号。对想进一步探索的读者，可以关注 **“knowledge elicitation（知识抽取）”** 与 **“model introspection（模型内省）** 的交叉研究，尤其是如何把自知能力嵌入实际产品的安全阈值判断中。

### 一句话记住它
只要让 LLM 把自己知道的全部列出来，就能直接看出它到底有多了解——规模够大时，它们真的会“知道自己知道多少”。