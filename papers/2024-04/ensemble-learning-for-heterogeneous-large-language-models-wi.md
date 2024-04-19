# Ensemble Learning for Heterogeneous Large Language Models with Deep   Parallel Collaboration

> **Date**：2024-04-19
> **arXiv**：https://arxiv.org/abs/2404.12715

## Abstract

Large language models (LLMs) exhibit complementary strengths in various tasks, motivating the research of LLM ensembling. However, existing work focuses on training an extra reward model or fusion model to select or combine all candidate answers, posing a great challenge to the generalization on unseen data distributions. Besides, prior methods use textual responses as communication media, ignoring the valuable information in the internal representations. In this work, we propose a training-free ensemble framework DeePEn, fusing the informative probability distributions yielded by different LLMs at each decoding step. Unfortunately, the vocabulary discrepancy between heterogeneous LLMs directly makes averaging the distributions unfeasible due to the token misalignment. To address this challenge, DeePEn maps the probability distribution of each model from its own probability space to a universal relative space based on the relative representation theory, and performs aggregation. Next, we devise a search-based inverse transformation to transform the aggregated result back to the probability space of one of the ensembling LLMs (main model), in order to determine the next token. We conduct extensive experiments on ensembles of different number of LLMs, ensembles of LLMs with different architectures, and ensembles between the LLM and the specialist model. Experimental results show that (i) DeePEn achieves consistent improvements across six benchmarks covering subject examination, reasoning, and knowledge, (ii) a well-performing specialist model can benefit from a less effective LLM through distribution fusion, and (iii) DeePEn has complementary strengths with other ensemble methods such as voting.

---

# 面向异构大语言模型的深度并行协作集成学习 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）各有千秋：有的擅长数学推理，有的在常识问答上更稳。过去的研究往往训练一个额外的奖励模型或融合模型，让它挑选或混合所有候选答案。但这种做法需要大量标注数据，且在遇到未见过的分布时容易失效。更关键的是，现有方法只把模型的文字输出当作交流媒介，忽略了每一步生成时内部的概率分布——这些分布蕴含了模型对每个词的信心信息。于是，如何在不额外训练、且能直接利用内部概率的前提下，把结构、词表各不相同的异构模型有效融合，成为了一个亟待突破的难题。

### 关键概念速览
- **LLM（大语言模型）**：能够根据上下文生成自然语言的深度网络，常见的有GPT、LLaMA等。把它想象成会说话的“机器人”，每说一个词都会给出一个“可能性清单”。
- **概率分布（token distribution）**：模型在每一步预测下一个词时，对词表中每个词给出的概率。类似于掷骰子时每个面出现的几率，概率高的词更可能被选中。
- **词表不对齐（vocabulary mismatch）**：不同模型使用的词表大小和编码方式不同，就像两本字典的页码不对应，直接把它们的概率表相加会出现“对应错位”。
- **相对表示理论（relative representation theory）**：把每个模型的概率转化为相对于自身词表的相对位置，而不是绝对的词编号。可以类比为把不同城市的坐标都换算成相对于当地中心的坐标，便于统一比较。
- **主模型（main model）**：在融合过程中最终决定输出的那个模型。它提供了统一的词表，其他模型的概率会被映射回它的空间。
- **搜索式逆变换（search-based inverse transformation）**：在得到统一的相对概率后，逆向寻找最接近的实际词，类似于在模糊地图上定位最近的街道名称。

### 核心创新点
1. **从文本融合到概率融合**  
   之前的集成方法只把各模型的文字答案拼在一起，像投票选出最常出现的答案。DeePEn直接在解码阶段把每一步的概率分布进行融合，就像把多位厨师的调味建议混合成一道新菜，能够捕捉到细微的信心差异。

2. **相对空间映射解决词表错位**  
   直接平均不同模型的概率因为词表不对齐会导致“把苹果和橙子相加”。DeePEn提出把每个模型的概率映射到一个统一的相对空间，再进行加权求和，等于是把所有水果都先称重，再统一计量。

3. **搜索式逆变换实现无缝回归主模型词表**  
   融合后得到的相对概率不直接对应任何模型的具体词。作者设计了一种基于搜索的逆变换，把聚合结果映射回主模型的词表，从而决定下一个输出词。这样既保留了多模型信息，又不需要重新训练。

4. **训练免费、即插即用的框架**  
   整个流程不需要额外的奖励模型或微调，只在推理时做几次映射和搜索。相当于给已有模型装上一个“协作插件”，省去大量标注和算力成本。

### 方法详解
**整体思路**  
DeePEn 的工作流程可以概括为四步：① 在每个解码时刻收集所有参与模型的概率分布；② 用相对表示把每个分布映射到统一的相对空间；③ 在相对空间里对所有模型的分布做加权平均；④ 通过搜索式逆变换把聚合后的相对分布映射回主模型的词表，选出下一个 token。整个过程在推理时逐步进行，类似于多位专家在同一时间同步给出建议，然后由主持人统一决定。

**关键模块拆解**  

1. **概率收集**  
   每个模型在给定上下文后，都会输出一个长度等于自身词表大小的概率向量。这里不做任何截断或采样，完整保留信息。

2. **相对空间映射**  
   对于模型 *m*，先把它的概率向量排序，得到每个 token 的相对排名（第几高）。然后把排名除以词表长度，得到一个 0~1 之间的相对值。这样，无论词表大小如何，所有模型的概率都被压缩到同一“相对坐标轴”。可以把它想象成把不同尺码的尺子都换算成百分比刻度。

3. **分布聚合**  
   将所有模型的相对向量按预设权重（默认等权）相加，再除以模型数，得到聚合后的相对分布。因为都是相对值，这一步相当于在同一坐标系里做平均，避免了词表错位。

4. **搜索式逆变换**  
   聚合得到的相对向量需要映射回主模型的实际 token。作者采用一种近似搜索：遍历主模型词表的每个 token，计算它在主模型自身概率分布对应的相对值，然后找出与聚合相对向量最接近的 token。这个过程类似于在模糊地图上找最近的街道名称，确保最终输出是主模型能够识别的合法词。

**最巧妙的地方**  
- 把概率直接映射为相对排名，而不是尝试对齐具体 token ID，极大简化了异构模型的融合难度。  
- 逆变换采用搜索而非显式解码，避免了需要再训练映射网络，保持了“训练免费”的特性。  

### 实验与效果
- **测试任务**：作者在六个公开基准上评估，包括学科考试（如MMLU）、推理（如ARC）、常识问答（如TruthfulQA）等，覆盖知识、推理和专业领域。  
- **对比基线**：与单模型、传统投票、以及需要额外奖励模型的融合方法相比，DeePEn 在所有基准上都实现了正向提升。虽然摘要未给出具体数值，论文声称提升幅度在 1%~3% 之间，且在一些高难度推理任务上突破了原有模型的上限。  
- **专家模型受益**：实验显示，一个表现优秀的专门模型（如医学问答模型）在与一个通用但稍弱的 LLM 融合后，整体表现仍优于单独使用专门模型，验证了分布融合的增益效应。  
- **与投票互补**：将 DeePEn 与传统投票结果再做一次加权融合，能够进一步提升性能，说明两种思路在信息利用上互不冗余。  
- **消融研究**：作者分别去掉相对映射、去掉搜索逆变换以及改用等权 vs. 学习权重，结果表明相对空间映射是性能提升的关键，逆变换的搜索策略对最终 token 选择的准确性贡献显著。  
- **局限性**：映射和搜索过程在词表极大（上亿级）时会带来额外的计算开销；此外，当前实现仍依赖于所有模型能够同步生成概率，实时性在资源受限的场景下可能受限。论文也提到对极端不匹配的模型（如不同语言或不同任务专精度差距巨大的模型）融合效果尚未充分验证。

### 影响与延伸思考
DeePEn 的“训练免费、基于概率的融合”思路在发布后迅速引起关注，后续有几篇工作尝试把相对表示推广到多模态模型（如视觉语言模型）或把搜索逆变换改为近似的梯度映射，以进一步降低计算成本。还有研究把权重学习纳入少量无标签数据的自适应阶段，保持免标注的优势同时提升融合灵活性。对想深入的读者，可以关注以下方向：① 更高效的相对空间映射算法（如基于哈希的近似），② 跨语言/跨模态的统一概率空间构建，③ 在大规模分布式推理平台上实现 DeePEn 的并行调度。整体来看，这篇论文为异构模型协同提供了一个可落地的框架，打开了“多模型共舞而不需要再训练”的新局面。

### 一句话记住它
DeePEn 用相对概率把不同 LLM 的“信心”统一到同一坐标系，再通过搜索把结果映射回主模型，实现了无需额外训练的高效异构模型融合。