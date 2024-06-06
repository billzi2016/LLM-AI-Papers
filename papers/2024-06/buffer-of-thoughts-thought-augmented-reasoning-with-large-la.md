# Buffer of Thoughts: Thought-Augmented Reasoning with Large Language   Models

> **Date**：2024-06-06
> **arXiv**：https://arxiv.org/abs/2406.04271

## Abstract

We introduce Buffer of Thoughts (BoT), a novel and versatile thought-augmented reasoning approach for enhancing accuracy, efficiency and robustness of large language models (LLMs). Specifically, we propose meta-buffer to store a series of informative high-level thoughts, namely thought-template, distilled from the problem-solving processes across various tasks. Then for each problem, we retrieve a relevant thought-template and adaptively instantiate it with specific reasoning structures to conduct efficient reasoning. To guarantee the scalability and stability, we further propose buffer-manager to dynamically update the meta-buffer, thus enhancing the capacity of meta-buffer as more tasks are solved. We conduct extensive experiments on 10 challenging reasoning-intensive tasks, and achieve significant performance improvements over previous SOTA methods: 11% on Game of 24, 20% on Geometric Shapes and 51% on Checkmate-in-One. Further analysis demonstrate the superior generalization ability and model robustness of our BoT, while requiring only 12% of the cost of multi-query prompting methods (e.g., tree/graph of thoughts) on average. Notably, we find that our Llama3-8B+BoT has the potential to surpass Llama3-70B model. Our project is available at: https://github.com/YangLing0818/buffer-of-thought-llm

---

# 思维缓冲区：基于大语言模型的思维增强推理 论文详细解读

### 背景：这个问题为什么难？

在使用大语言模型（LLM）解决需要多步推理的任务时，模型往往会出现“思路漂移”或“中途卡壳”。传统的 Chain‑of‑Thought（思维链）让模型把每一步写出来，但每一次推理都要从零开始，缺少对过去任务的经验复用。更进一步的 Tree‑of‑Thought、Graph‑of‑Thought 通过多次查询探索搜索空间，虽然提升了准确率，却把计算成本推到数倍甚至十倍。于是，研究者面临两个矛盾：**如何让模型在保持高准确率的同时，显著降低推理开销**，以及**如何让模型的推理过程能够随任务累积而逐渐变得更聪明**。这正是 BoB（Buffer of Thoughts）要破解的难点。

### 关键概念速览
- **思维模板（thought‑template）**：从大量解题过程里抽象出的高层次推理结构，就像把“解几何题的常用步骤”浓缩成一张流程图，供后续问题直接套用。
- **元缓冲区（meta‑buffer）**：存放所有思维模板的仓库，类似于人类的大脑记忆库，随着解决的任务增多会不断扩充。
- **缓冲区管理器（buffer‑manager）**：负责从元缓冲区挑选最匹配的模板、在新任务上微调并把新产生的有效模板写回去的模块，像是图书馆的管理员，既要检索也要更新藏书。
- **思维增强推理（thought‑augmented reasoning）**：在实际求解时，先把选中的模板实例化为具体的推理步骤，再交给 LLM 完成细节，类似于先给学生一个解题框架，再让他填空。
- **多查询提示（multi‑query prompting）**：指在同一个问题上向模型发起多次不同的查询以探索解空间，成本高但有时能提升答案质量。
- **鲁棒性（robustness）**：模型在面对噪声、表述变化或少量错误提示时仍能保持正确推理的能力。

### 核心创新点
1. **从“每题一次”到“模板复用”**  
   传统方法每遇到新问题都让模型从头写思维链，导致重复劳动。BoB 通过在元缓冲区中保存高层次思维模板，并在新任务上检索最相似的模板进行实例化，实现了经验的跨任务复用。这样模型不必重新发现推理结构，直接在已有框架上填充细节，显著提升了推理效率。

2. **动态更新的缓冲区管理机制**  
   以前的模板库往往是离线构建，缺乏自适应能力。BoB 引入缓冲区管理器，在每次解题后评估生成的推理是否比已有模板更优，如果是，就把新模板写回元缓冲区。相当于让模型在使用过程中“自学”，缓冲区容量随任务数量线性增长，保持可扩展性。

3. **高效的检索‑实例化流程**  
   为了避免全库线性搜索，作者设计了基于任务描述的向量检索 + 轻量级相似度过滤的两阶段检索。检索到的模板随后通过占位符替换、结构微调等步骤实例化为具体的思维链，整个过程只需要一次模型调用，成本仅为多查询方法的约 12%。

4. **跨模型的性能迁移**  
   实验显示，使用 BoB 的 Llama‑3‑8B 能在多数基准上逼近甚至超越未使用 BoB 的 Llama‑3‑70B。核心原因在于模板提供了强先验，使得小模型能够“站在巨人的肩膀上”。这表明思维模板的价值不局限于特定模型规模。

### 方法详解
**整体框架**  
BoB 的推理过程可以划分为四步：① **模板收集**（离线阶段），② **元缓冲区构建**，③ **任务检索‑实例化**（在线推理），④ **缓冲区更新**。整体思路是把“思考的套路”先存起来，遇到新问题时快速调出并微调，然后把新学到的套路再放回去。

**1. 模板收集**  
研究者先让基线 LLM 在一批标注好的任务上使用标准的 Chain‑of‑Thought 提示，得到完整的推理过程。随后通过人工或自动化的模式抽取，把这些过程压缩成高层次的思维模板。例如，在“24 点游戏”中，模板可能是“先找两数相乘得到中间结果，再与第三数相加”。每个模板都带有占位符（如 `<NUM1>`、`<OP>`），便于后续填充。

**2. 元缓冲区构建**  
所有模板以向量形式存入元缓冲区，向量由模板的文字描述和结构特征共同编码。这样可以利用向量相似度快速检索。元缓冲区本身是一个可写的键值库，键是模板的语义摘要，值是模板本体及其统计信息（如使用频率、成功率）。

**3. 任务检索‑实例化**  
当模型收到新问题时，首先把问题描述编码成向量，然后在元缓冲区进行近似最近邻搜索，返回 top‑k（通常为 1‑3）最相似的模板。随后进入实例化阶段：  
- 用问题中的具体数值或概念替换模板占位符。  
- 根据模板的结构（如“先比较后递归”）生成对应的提示文本。  
- 将实例化后的思维链一次性喂给 LLM，让模型只完成细节填充和最终答案输出。  
这一步只需要一次模型调用，避免了多轮查询的累积成本。

**4. 缓冲区更新**  
推理结束后，系统会检查生成的答案是否正确以及思维链的质量（比如是否出现逻辑冲突）。如果表现优于当前模板对应的历史记录，管理器会把这条新的思维链抽象成模板并写回元缓冲区。更新策略采用阈值过滤和置信度加权，确保噪声不会污染库。

**最巧妙的设计**  
- **模板的层次抽象**：不是把完整的推理步骤硬塞进去，而是保留“结构骨架”，让模型在实例化时仍有自由度，这兼顾了可复用性和灵活性。  
- **自适应阈值**：缓冲区管理器会根据任务难度动态调节更新阈值，难题更容易保留新模板，简单任务则倾向于使用已有模板，避免库膨胀。

### 实验与效果
- **测试任务**：论文在 10 项推理密集型基准上评估，包括 Game of 24、Geometric Shapes、Checkmate‑in‑One、数独、逻辑谜题等。  
- **对比基线**：主要与标准 Chain‑of‑Thought、Tree‑of‑Thought、Graph‑of‑Thought 以及最新的 Self‑Consistency 方法比较。  
- **核心数字**：在 Game of 24 上提升 11%，Geometric Shapes 提升 20%，Checkmate‑in‑One 提升 51%。整体平均提升约 18%。  
- **成本对比**：BoB 只需要约 12% 的模型调用次数即可达到或超过多查询方法的效果，推理时间和算力开销大幅下降。  
- **消融实验**：去掉缓冲区管理器后，性能下降约 6%；仅使用单一模板（不检索）则下降约 9%；不进行模板更新则在后期任务上出现显著退化，说明动态更新是关键。  
- **局限性**：论文承认在高度开放式的自然语言问答上，模板检索的效果仍有限，因为模板的结构化程度难以覆盖所有语言变体。还有模板抽取过程仍依赖人工规则，自动化程度有待提升。

### 影响与延伸思考
BoB 把“经验复用”引入 LLM 推理，开启了 **思维库**（thought‑library）概念的探索。随后的工作如 *ThoughtBank*、*PromptCache* 等，都在尝试把历史推理过程以更高效的方式存取。一个值得关注的方向是 **跨模态模板**：把视觉、代码等任务的推理结构也统一进元缓冲区，实现多模态的思维共享。另一个潜在路径是 **自监督模板生成**，让模型在无监督数据上自行抽象出思维模板，进一步降低人工成本。对想深入的读者，可以关注近期在 ACL、NeurIPS 上出现的 “Prompt Retrieval” 与 “Meta‑Prompting” 系列论文，它们在思路上与 BoB 有明显的血缘关系。

### 一句话记住它
**BoB 把高层次的推理模板存进“记忆库”，让每一次解题都能直接套用经验，从而用更少的调用次数获得更高的准确率。**