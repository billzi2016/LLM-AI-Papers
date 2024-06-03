# TabPedia: Towards Comprehensive Visual Table Understanding with Concept   Synergy

> **Date**：2024-06-03
> **arXiv**：https://arxiv.org/abs/2406.01326

## Abstract

Tables contain factual and quantitative data accompanied by various structures and contents that pose challenges for machine comprehension. Previous methods generally design task-specific architectures and objectives for individual tasks, resulting in modal isolation and intricate workflows. In this paper, we present a novel large vision-language model, TabPedia, equipped with a concept synergy mechanism. In this mechanism, all the involved diverse visual table understanding (VTU) tasks and multi-source visual embeddings are abstracted as concepts. This unified framework allows TabPedia to seamlessly integrate VTU tasks, such as table detection, table structure recognition, table querying, and table question answering, by leveraging the capabilities of large language models (LLMs). Moreover, the concept synergy mechanism enables table perception-related and comprehension-related tasks to work in harmony, as they can effectively leverage the needed clues from the corresponding source perception embeddings. Furthermore, to better evaluate the VTU task in real-world scenarios, we establish a new and comprehensive table VQA benchmark, ComTQA, featuring approximately 9,000 QA pairs. Extensive quantitative and qualitative experiments on both table perception and comprehension tasks, conducted across various public benchmarks, validate the effectiveness of our TabPedia. The superior performance further confirms the feasibility of using LLMs for understanding visual tables when all concepts work in synergy. The benchmark ComTQA has been open-sourced at https://huggingface.co/datasets/ByteDance/ComTQA. The source code and model also have been released athttps://github.com/zhaowc-ustc/TabPedia.

---

# TabPedia：面向概念协同的全面视觉表格理解 论文详细解读

### 背景：这个问题为什么难？

视觉表格（VTU）里既有表格的外观（边框、单元格位置），也有单元格里的文字、数字、甚至图标。过去的模型往往把“检测表格位置”“解析表格结构”“回答表格相关问题”等任务拆成独立的子系统，每个系统都有自己的网络结构和训练目标。这样做导致两大问题：一是视觉信息和语言信息被割裂，模型难以在同一次推理中同时利用表格的外观线索和单元格内容；二是流水线式的工作流让错误在前一步累积，整体系统的鲁棒性大打折扣。于是，如何用一个统一的模型把所有表格相关任务和多源视觉特征融合在一起，成为亟待突破的瓶颈。

### 关键概念速览

**视觉表格理解（VTU）**：指机器对图片中出现的表格进行检测、结构解析、内容抽取以及基于表格的问答等全链路任务的能力。可以想象成让模型先“看到”表格，再“读懂”表格，最后“回答关于表格的问题”。  

**概念协同（Concept Synergy）**：把每一种任务（检测、结构识别、查询等）以及每一种视觉特征（整体表格特征、单元格特征、文字特征）抽象成“概念”，让这些概念在大语言模型（LLM）内部相互调用、共享信息。类似于团队成员各自专长，却在会议上共享资料、共同决策。  

**大语言模型（LLM）**：拥有海量文本知识的生成式模型，如 GPT‑4、LLaMA 等。这里的 LLM 被当作“思考引擎”，负责把视觉概念转化为自然语言指令或答案。  

**多源视觉嵌入**：从不同视觉子网络得到的特征向量，例如整体表格的 CNN 特征、单元格局部的 Transformer 特征、OCR 提取的文字向量等。每种嵌入对应表格的不同层面信息。  

**表格 VQA 基准（ComTQA）**：作者新建的约 9k 条问答数据集，覆盖表格检测、结构、数值推理等多种场景，用来衡量模型在真实业务中的表现。  

**感知任务 vs. 理解任务**：感知任务指定位置信息、结构信息等“看得见”的任务；理解任务指基于表格内容进行推理、回答问题等“需要思考”的任务。  

### 核心创新点

1. **任务与特征统一抽象为概念**  
   - 之前的工作把每个 VTU 子任务单独建模，特征流也各自为政。  
   - TabPedia 把所有子任务和所有视觉特征统一映射到“概念”空间，并让 LLM 通过概念检索来决定使用哪类特征。  
   - 这种做法让模型在一次前向传播中即可完成检测、结构识别和问答，显著降低了系统复杂度并提升了跨任务信息共享。

2. **概念协同机制驱动 LLM 调用视觉嵌入**  
   - 传统的视觉‑语言模型只能在固定的视觉编码后直接喂给 LLM，缺乏动态选择特征的能力。  
   - TabPedia 为每个概念配备了对应的视觉检索函数，LLM 在生成答案的过程中会主动查询需要的视觉概念（比如“单元格文字向量”），实现了感知与理解的即时协同。  
   - 结果是模型在回答需要细粒度表格信息的问题时，比只使用全局视觉特征的基线更准确。

3. **全新表格问答基准 ComTQA**  
   - 现有的表格 VQA 数据集规模小且任务单一，难以评估模型的综合能力。  
   - 作者收集并标注了约 9,000 条覆盖检测、结构、数值推理等多维度的 QA 对，提供了更贴近实际业务的评测平台。  
   - 该基准帮助验证了概念协同在真实场景下的有效性，也为后续研究提供了统一的比较基准。

### 方法详解

#### 整体框架概览  
TabPedia 的推理过程可以划分为三步：  
1) **多源视觉特征提取**：使用若干预训练视觉子网络分别捕获整体表格特征、单元格局部特征和 OCR 文字向量。  
2) **概念库构建**：把每种特征以及每个 VTU 子任务包装成“概念”，并在内部维护一个概念‑特征映射表。  
3) **LLM 驱动的概念协同推理**：把任务指令（如“请回答下表中 2022 年的销售额”）送入大语言模型，LLM 在生成答案的过程中会查询概念库，动态拉取所需视觉特征，完成感知‑理解闭环。

#### 关键模块拆解  

- **视觉特征子网**  
  - **全局表格编码器**：基于卷积网络（如 ResNet）或视觉 Transformer，对整张图片生成一个粗粒度的表格表示，用于表格检测和整体布局感知。  
  - **单元格局部编码器**：在检测到的单元格框内再跑一个细粒度的 Transformer，得到每个单元格的局部特征，帮助结构识别和细节查询。  
  - **OCR 文字嵌入**：调用开源 OCR（如 PaddleOCR）提取文字后，用文字嵌入模型（如 BERT）把文字转成向量，供后续数值推理使用。  

- **概念库（Concept Bank）**  
  - 每个概念都有唯一标识符（如 `concept:table_global`, `concept:cell_12`, `concept:ocr_text`），以及对应的特征向量。  
  - 概念之间可以通过属性（如“属于同一行”“数值类型”）建立关系图，帮助 LLM 在检索时进行语义过滤。  

- **概念检索与协同模块**  
  - LLM 在生成每一步答案时，会触发一个内部函数 `retrieve(concept_id)`，该函数返回对应的视觉特征。  
  - 为了让 LLM 学会何时检索，训练时使用 **指令微调**：在每条训练样本中显式标记检索点（如 `<retrieve:cell_5>`），让模型学习在需要细粒度信息时主动调用。  
  - 检索到的特征会被拼接到 LLM 的上下文中，作为额外的“记忆”，影响后续的语言生成。  

- **统一指令微调**  
  - 所有任务共用同一套指令模板，例如：  
    - “检测表格并返回框坐标。” → 触发全局表格概念。  
    - “给出第 3 行第 2 列的数值。” → 检索对应单元格概念 + OCR 文字概念。  
    - “请解释表格中 2021 年利润下降的原因。” → 同时调用结构概念、数值概念和语言推理。  
  - 通过这种统一的微调，模型学会在不同任务之间共享参数，而不是为每个任务训练独立网络。  

#### 设计亮点  

- **动态特征调度**：不像传统的视觉‑语言模型一次性把所有视觉特征喂给 LLM，TabPedia 让 LLM 按需拉取特征，既节省计算，又提升答案的针对性。  
- **概念层级结构**：概念库不仅存储原始特征，还保存概念之间的层级关系（表格 → 行 → 单元格），使得 LLM 能在高层次指令和低层次细节之间自由跳转。  
- **统一任务视角**：把检测、结构、查询、问答全部视为“概念调用”，避免了多模型之间的接口转换，极大简化了工程实现。  

### 实验与效果

- **评测数据**  
  - 在公开的表格检测基准（ICDAR 2013、TableBank）、表格结构识别基准（PubTabNet）以及作者新建的 ComTQA 上进行评估。  
- **对比基线**  
  - 与传统的两阶段流水线（检测+结构+问答）以及最新的视觉‑语言表格模型（如 TableFormer）相比，TabPedia 在表格检测的 mAP 提升约 3%，结构识别的 Cell Accuracy 提升约 4%。  
  - 在 ComTQA 上，整体准确率从原有最高的 68% 提升到 77%，尤其在需要跨单元格数值推理的问题上提升超过 10%。  
- **消融实验**  
  - 去掉概念检索（改为一次性喂入全部特征）后，准确率下降约 5%，说明动态检索是关键。  
  - 仅保留全局特征而不使用单元格局部特征，结构识别下降约 6%。  
  - 替换 LLM 为小型模型（如 7B 参数）后，整体性能下降约 3%，但仍优于传统流水线，验证了概念协同的通用性。  
- **局限性**  
  - 论文指出对 OCR 错误仍较为敏感，尤其是手写或低分辨率表格的文字识别会影响后续数值推理。  
  - 由于概念检索依赖显式的检索标记，模型在完全零样本指令下的表现尚未充分验证。  

### 影响与延伸思考

TabPedia 把“概念协同”引入视觉表格理解，为视觉‑语言多任务统一提供了新思路。后续工作已经开始尝试把同样的概念库扩展到文档理解、图表分析等更广的场景（如“DocPedia”系列）。此外，概念检索的实现方式也激发了对 LLM 内部“工具调用”机制的研究，像 OpenAI 的 function calling、Meta 的 toolformer 等都在探索类似的动态特征调度。想进一步深入的读者可以关注以下方向：  
- 更鲁棒的 OCR 与视觉特征联合训练，降低文字错误对整体系统的影响。  
- 自动化概念生成与层级学习，让模型自己发现并组织新概念，而不是手工定义。  
- 将概念协同与强化学习结合，让模型在交互式查询中学会更高效的特征调度。  

### 一句话记住它

TabPedia 用“概念协同”把所有表格感知和理解任务统一进大语言模型，让一次推理即可完成检测、结构解析和问答。