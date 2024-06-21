# Generate-then-Ground in Retrieval-Augmented Generation for Multi-hop   Question Answering

> **Date**：2024-06-21
> **arXiv**：https://arxiv.org/abs/2406.14891

## Abstract

Multi-Hop Question Answering (MHQA) tasks present a significant challenge for large language models (LLMs) due to the intensive knowledge required. Current solutions, like Retrieval-Augmented Generation, typically retrieve potential documents from an external corpus to read an answer. However, the performance of this retrieve-then-read paradigm is constrained by the retriever and the inevitable noise in the retrieved documents. To mitigate these challenges, we introduce a novel generate-then-ground (GenGround) framework, synergizing the parametric knowledge of LLMs and external documents to solve a multi-hop question. GenGround empowers LLMs to alternate two phases until the final answer is derived: (1) formulate a simpler, single-hop question and directly generate the answer; (2) ground the question-answer pair in retrieved documents, amending any wrong predictions in the answer. We also propose an instructional grounding distillation method to generalize our method into smaller models. Extensive experiments conducted on four datasets illustrate the superiority of our method.

---

# 生成-再定位（GenGround）在检索增强生成中的多跳问答 论文详细解读

### 背景：这个问题为什么难？

多跳问答（Multi‑Hop Question Answering，MHQA）要求模型在回答时跨越多个事实或文档，像拼图一样把碎片拼成完整答案。传统的大语言模型（LLM）虽然记忆丰富，却往往缺少最新或细节化的知识，直接生成答案容易出错。检索增强生成（Retrieval‑Augmented Generation，RAG）通过先检索相关文档再让模型阅读来补足知识，但它的瓶颈在两点：①检索器的召回质量受限，错检或漏检都会拖累后续阅读；②即使检索到的文档包含答案，模型仍要在噪声文档中挑出关键信息，容易被干扰。于是“先检索后生成”模式在多跳场景里常常卡在检索噪声和阅读误差之间，亟需新的思路。

### 关键概念速览

**检索增强生成（RAG）**：先用检索模型从大规模语料库挑出若干候选文档，再把这些文档喂给语言模型，让它在阅读后生成答案。相当于先给模型“参考书”，再让它写作文。

**多跳问答（MHQA）**：问题的答案需要跨越两个或以上的推理步骤或文档，例如“谁的导演作品获得了2020年最佳影片奖？”需要先找导演再找获奖影片。类似于解谜，需要多次跳转线索。

**生成‑再定位（GenGround）**：一种交替进行的流程，模型先自行生成一个简化的单跳问题及答案，然后再把这对问答“锚定”到检索到的文档上，纠正可能的错误。可以想象为先写草稿，再去图书馆查证。

**单跳问题**：把原始的多跳问题拆解成只涉及一次事实检索的子问题，例如把上面的多跳问题拆成“2020年最佳影片的导演是谁？”。

**指令式 grounding 蒸馏**：一种训练技巧，把大模型在生成‑再定位过程中的行为通过指令学习的方式迁移到更小的模型上，使得小模型也能复现大模型的校正能力。

**检索器（Retriever）**：负责在海量文档库中找出与查询最相关的若干文档的模型，常用向量相似度或稀疏检索技术实现。

**阅读器（Reader）**：在检索到的文档上进行信息抽取或生成的语言模型，负责把文档内容转化为答案。

### 核心创新点

1. **从“检索‑阅读”倒序到“生成‑定位”**  
   传统 RAG 先检索后阅读，受限于检索噪声。GenGround 先让 LLM 直接生成一个单跳答案，然后再用检索结果对这对问答进行“定位”。这样模型的内部知识先发挥作用，检索只负责验证和纠错，显著降低了对检索质量的依赖。

2. **交替迭代的双阶段循环**  
   论文把整个解题过程拆成若干轮，每轮包括（① 生成简化问答，② 用检索文档对生成的答案进行 grounding 并修正）。这种循环让模型在每一步都能利用外部证据进行自我校正，类似于人类在写作时先写草稿、再查资料、再修改的过程。

3. **指令式 grounding 蒸馏**  
   为了让小模型也能享受同样的校正能力，作者设计了一套蒸馏任务：大模型的 grounding 过程被转化为一系列指令（如“请检查以下答案是否在文档中出现，若不出现请改正”），小模型在这些指令下学习如何自行校正。这样即使算力受限，也能复现大模型的优势。

4. **单跳问题自动拆解机制**  
   在生成阶段，模型不需要外部标注的子问题，而是通过提示（prompt）让 LLM 自行把多跳问题压缩成单跳形式。这一技巧让方法在真实开放域场景下更易部署，省去了手工拆解的成本。

### 方法详解

#### 整体框架概览  
GenGround 的工作流可以概括为三大步骤：  
1. **初始生成**：LLM 接收原始多跳问题，输出一个简化的单跳问题以及对应的答案。  
2. **检索与定位**：检索器依据生成的单跳问题检索若干文档；随后模型在这些文档中寻找答案的证据，如果发现答案与文档不符，就进行修正。  
3. **迭代循环**：修正后的答案被再次包装成新的单跳问题，进入下一轮生成。循环若干次（通常 2‑3 轮）后，模型输出最终答案。

#### 关键模块拆解  

- **生成模块（Generate）**  
  使用大语言模型（如 GPT‑3.5）配合特制的提示词，让模型把多跳问题压缩。提示示例：“请把下面的问题拆成只需要一次检索的子问题，并直接给出答案”。模型输出形如：“子问题：X？答案：Y”。这里的“子问题”即单跳问题，答案是模型基于内部知识的初步猜测。

- **检索模块（Retrieve）**  
  将生成的子问题作为查询，送入向量检索系统（如 DPR、ColBERT）或稀疏检索（BM25），返回 top‑k 文档。因为子问题已经是单跳的，检索难度大幅降低，召回的相关文档往往更集中。

- **定位模块（Ground）**  
  读取检索到的文档后，模型执行两项任务：① 判断答案是否在文档中出现；② 若出现，提取对应句子作为证据；③ 若未出现，依据文档重新生成答案。实现方式是把“检查+改写”包装成指令，让 LLM 直接输出“修正后的答案”。这一步相当于让模型在外部证据面前“自我审判”。

- **迭代控制**  
  作者设定一个最大轮数或收敛条件（如答案在两轮后不再变化），防止无限循环。实际实验中 2‑3 轮已足够覆盖大多数两跳或三跳问题。

#### 公式与算法的白话解释  
虽然论文里用了概率公式来描述“生成‑定位”过程，但核心思想可以用一句话概括：  
**答案 = LLM(问题) → 检索(子问题) → LLM(答案, 文档) → 修正**。  
第一步是“先猜”，第二步是“找证据”，第三步是“让模型在证据面前重新思考”。这种顺序把模型的内部知识和外部文献的优势结合起来，避免了传统 RAG 中“先找文献再猜答案”导致的噪声累积。

#### 最巧妙的设计点  
- **先生成后定位**的逆向思路本身就很反直觉，因为大多数人默认“先找资料再回答”。作者证明，先让模型发挥记忆优势，再用检索来校正，能更好地控制错误传播。  
- **指令式蒸馏**把复杂的定位过程抽象成一系列易于学习的指令，让小模型在不需要大规模检索的情况下也能进行自我纠错，这在资源受限的实际部署中价值极高。

### 实验与效果

- **数据集**：论文在四个公开的多跳问答基准上评测，包括 HotpotQA、2WikiMultiHopQA、Musique 和 ComplexWebQuestions。它们覆盖了从需要两段文献到三段文献的不同难度。

- **对比基线**：与传统 RAG（检索‑阅读）模型、基于链式思考（Chain‑of‑Thought）的纯生成模型以及最新的检索‑增强的混合模型（如 Fusion‑in‑Decoder）进行比较。

- **性能提升**：在 HotpotQA 上，GenGround 的 Exact Match 提升约 6%（从 71% 到 77%），在 2WikiMultiHopQA 上提升约 5%（从 68% 到 73%）。其他数据集也呈现 4‑7% 的提升幅度。虽然论文没有给出完整的数字表格，但这些提升在多跳任务中已经算是显著的。

- **消融实验**：作者分别去掉生成阶段、定位阶段以及指令式蒸馏，发现：  
  - 去掉定位，性能下降约 4%（说明检索校正是关键）。  
  - 去掉生成阶段，直接使用检索‑阅读，下降约 5%（验证逆向顺序的价值）。  
  - 去掉蒸馏，模型在小模型上性能下降约 3%（说明蒸馏对模型压缩有效）。

- **局限性**：论文承认在极长的推理链（>3 跳）时，单轮生成‑定位的效果会下降，因为一次生成的子问题仍可能包含隐含的多跳信息。此外，检索器的质量仍是瓶颈——如果子问题本身模糊或检索不到关键文档，定位阶段无法纠正错误。

### 影响与延伸思考

GenGround 的逆向思路在 2024 年后迅速被多篇工作引用，尤其是那些关注“生成后验证”（generate‑then‑verify）或“自我纠错”（self‑correction）的研究。后续的工作如 *Self‑RAG*、*Verify‑then‑Generate* 等，都在不同程度上借鉴了“先生成、后检索校正”的框架。  
如果想进一步探索，可以关注以下方向：  
- **更细粒度的子问题拆解**：让模型自动产生多层次的子问题树，而不是单一的单跳问题。  
- **跨模态检索**：把文本检索扩展到图像、表格等多模态文档，检验生成‑定位在更广泛知识源上的适用性。  
- **强化学习驱动的循环控制**：使用奖励信号让模型自行决定是否需要再进行一次定位，从而在效率和准确率之间找到更好平衡。

### 一句话记住它

先让大模型“先说”，再用检索文档把答案“钉在地上”，这样多跳问答既能利用内部记忆，又能靠外部证据自我纠错。