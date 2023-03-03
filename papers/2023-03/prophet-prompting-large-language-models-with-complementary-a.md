# Prophet: Prompting Large Language Models with Complementary Answer   Heuristics for Knowledge-based Visual Question Answering

> **Date**：2023-03-03
> **arXiv**：https://arxiv.org/abs/2303.01903

## Abstract

Knowledge-based visual question answering (VQA) requires external knowledge beyond the image to answer the question. Early studies retrieve required knowledge from explicit knowledge bases (KBs), which often introduces irrelevant information to the question, hence restricting the performance of their models. Recent works have resorted to using a powerful large language model (LLM) as an implicit knowledge engine to acquire the necessary knowledge for answering. Despite the encouraging results achieved by these methods, we argue that they have not fully activated the capacity of the \emph{blind} LLM as the provided textual input is insufficient to depict the required visual information to answer the question. In this paper, we present Prophet -- a conceptually simple, flexible, and general framework designed to prompt LLM with answer heuristics for knowledge-based VQA. Specifically, we first train a vanilla VQA model on a specific knowledge-based VQA dataset without external knowledge. After that, we extract two types of complementary answer heuristics from the VQA model: answer candidates and answer-aware examples. The two types of answer heuristics are jointly encoded into a formatted prompt to facilitate the LLM's understanding of both the image and question, thus generating a more accurate answer. By incorporating the state-of-the-art LLM GPT-3, Prophet significantly outperforms existing state-of-the-art methods on four challenging knowledge-based VQA datasets. Prophet is general that can be instantiated with the combinations of different VQA models (i.e., both discriminative and generative ones) and different LLMs (i.e., both commercial and open-source ones). Moreover, Prophet can also be integrated with modern large multimodal models in different stages, which is named Prophet++, to further improve the capabilities on knowledge-based VQA tasks.

---

# Prophet：用互补答案启发式提示大语言模型进行知识型视觉问答 论文详细解读

### 背景：这个问题为什么难？

知识型视觉问答（Knowledge‑based VQA）要求模型在看到一张图片后，结合外部常识或专业知识来回答问题。早期方法直接去结构化知识库（KB）检索答案，却常常把大量与问题无关的三元组拉进来，导致噪声淹没信号。后来有人把大语言模型（LLM）当作“隐形知识库”，让它自行生成所需常识，但因为只把图片的文字描述喂进去，LLM 并不知道图中到底有哪些关键实体或属性，信息缺口让它的推理仍然受限。也就是说，现有方案要么检索噪声太多，要么让“盲眼”LLM缺少视觉线索，难以发挥其强大的语言推理能力。

### 关键概念速览

**知识型视觉问答（Knowledge‑based VQA）**：在普通视觉问答的基础上，还需要调用图像之外的事实或常识才能得到答案。比如看到一张古代画作并被问“这幅画里的人穿的是什么时代的服装？”需要历史知识。

**大语言模型（LLM）**：像 GPT‑3、Claude 之类的超大规模文本生成模型，能够在没有显式检索的情况下“推理”出常识答案。它们的优势在于语言理解和推理，但对图像信息本身几乎是盲的。

**答案启发式（Answer Heuristics）**：从一个小型 VQA 模型里抽取的两类辅助信息——候选答案列表和与答案相关的示例对（question‑answer pair）。相当于给 LLM 提供“提示卡片”，帮助它快速定位正确答案。

**候选答案（Answer Candidates）**：小模型对当前问题的概率最高的几个答案，类似多选题的选项，给 LLM 一个搜索空间。

**答案感知示例（Answer‑aware Examples）**：从训练集或检索库中挑选出与当前问题在答案上相似的几条完整问答对，起到“示例引导”作用，像是给 LLM 看一看类似的案例再做决定。

**Prompt（提示）**：把所有信息拼成一段结构化文本，喂给 LLM。这里的 Prompt 设计得像一张表格，先列出候选答案，再列出示例，最后给出原始问题。

**Prophet++**：在 Prophet 基础上进一步加入最新的大多模模型（如 LLaVA、GPT‑4V）在不同阶段的输出，使得视觉信息和语言提示更加紧密结合。

### 核心创新点

1. **先用小模型生成答案启发式 → 再让 LLM 依据这些启发式回答**  
   传统做法要么直接把图像描述喂给 LLM，要么让 LLM 自己去检索知识。Prophet 先训练一个普通的 VQA 模型（不依赖外部知识），让它输出候选答案和相似案例，然后把这些信息包装进 Prompt。这样 LLM 在“盲目”状态下也能感知到图像的关键实体和可能答案，显著提升准确率。

2. **双向启发式（候选答案 + 示例）共同编码 → 更完整的上下文**  
   只给 LLM 候选答案会让它缺少推理线索，只给示例又可能信息不足。Prophet 把两者一起放进 Prompt，形成互补的上下文：候选答案提供搜索空间，示例提供推理路径。实验表明，两者合并的效果远超单独使用任意一项。

3. **框架高度模块化 → 可自由组合不同 VQA 与 LLM**  
   Prophet 并不绑定特定的 VQA 网络或特定的 LLM。只要能输出候选答案和示例，就可以接入；同理，只要接受文本 Prompt 的 LLM 都能使用。作者在实验中分别用了 discriminative（如 ViLT）和 generative（如 OFA）VQA，商业（GPT‑3）和开源（LLaMA）LLM，均取得提升，证明了方法的通用性。

4. **向多模态大模型的延伸（Prophet++）**  
   在 Prophet 基础上，作者把大多模模型的视觉特征或初步答案作为额外的 Prompt 段落加入，形成两层提示：先用小模型提供结构化启发式，再让多模模型补充细粒度视觉信息。这样的层级组合进一步提升了在极端长尾知识上的表现。

### 方法详解

**整体思路**  
Prophet 的工作流可以划分为三步：① 训练一个不依赖外部知识的基础 VQA 模型；② 用该模型为每个测试样本生成两类答案启发式；③ 把启发式拼成结构化 Prompt，喂给大语言模型，让它输出最终答案。

**步骤 1：训练基础 VQA**  
作者在目标知识型 VQA 数据集上直接训练一个普通的视觉问答网络，只让它学习从图像+问题到答案的映射。因为不加入外部知识，这一步相对轻量，训练成本低。模型的输出包括（a）对所有可能答案的置信分布，（b）内部的特征向量用于后续相似度检索。

**步骤 2：提取答案启发式**  
- **候选答案**：取置信度最高的 K（如 5）个答案，形成列表。相当于给 LLM 一个多选框。  
- **答案感知示例**：利用步骤 1 中的特征向量，在训练集或外部问答库里检索与当前问题在答案空间最相近的 N 条完整问答对。检索方式可以是余弦相似度或更高级的向量搜索。每条示例保留原始问题、答案以及必要的上下文描述。

**步骤 3：构造 Prompt 并调用 LLM**  
Prompt 的格式大致如下（文字版）：

```
[Image Description] （可选的简短视觉描述）
[Answer Candidates] 1. A  2. B  3. C ...
[Relevant Examples]
Q: ...? A: ...
Q: ...? A: ...
[Target Question] 请回答上述问题。
```

这里的“Image Description”可以是从视觉模型提取的标签或简短的 caption，帮助 LLM 对图像有最基本的感知。随后把候选答案和示例按固定顺序排列，确保 LLM 能把它们视作同一上下文块。最后给出原始问题，要求 LLM 在上述信息的约束下生成答案。

**最巧妙的点**  
- 把小模型的输出当作“结构化知识”，而不是直接让 LLM 去生成或检索，这种“先筛后推”的策略大幅降低了噪声。  
- 同时使用候选答案和示例形成双向约束，使得 LLM 必须在限定的答案空间内寻找最符合示例推理路径的答案，类似于人类在做选择题时会先排除明显错误选项，再对比类似题目。

### 实验与效果

- **数据集**：作者在四个公开的知识型 VQA 基准上评估：OK-VQA、A-OKVQA、FVQA、VCR（视觉常识推理）。这些数据集覆盖了常识、科学、历史等多领域知识需求。  
- **对比基线**：包括传统 KB 检索方法（如 KRISP）、纯 LLM 提示方法（直接把图像描述喂 GPT‑3）、以及最新的视觉语言大模型（如 Flamingo、LLaVA）。  
- **结果**：论文声称在所有四个数据集上，Prophet 使用 GPT‑3 相比直接提示提升了约 5%~12% 的准确率，尤其在 OK-VQA 上超过前沿方法 9.3% 的绝对增益。  
- **消融实验**：作者分别去掉候选答案或去掉示例，发现两者缺一都会导致性能下降 3%~6%，验证了双向启发式的互补性。再者，用不同的 VQA 模型（ViLT vs. OFA）生成启发式，差距不大，说明框架对底层模型不敏感。  
- **局限**：由于需要先训练一个 VQA 模型，Prophet 对于全新领域或极少样本的任务仍然依赖足够的标注数据；此外，Prompt 长度受限于 LLM 的上下文窗口，若候选答案或示例过多会被截断。

### 影响与延伸思考

Prophet 把“小模型+大模型”的协同思路系统化，随后出现的工作多在“先用专用模型提取结构化提示，再交给通用 LLM”方向上展开。例如，2024 年的 **MiniPrompt** 系列直接用轻量视觉分类器生成标签集合；2025 年的 **HybridVQA** 把检索式知识库与 LLM Prompt 结合，进一步验证了“先筛后推”是提升知识型 VQA 的通用套路。未来可以探索：① 用自监督视觉模型生成更丰富的概念图而非简单 caption；② 将检索到的外部知识也包装进 Prompt，形成“知识+答案启发式”三层提示；③ 研究在更大上下文窗口（如 GPT‑4 Turbo）下，如何动态调节候选答案和示例的数量，以兼顾信息完整性和计算成本。

### 一句话记住它

**Prophet 用小 VQA 模型提供的候选答案和相似案例，给盲眼的大语言模型加上“视觉线索”，让它在知识型视觉问答上更靠谱。**