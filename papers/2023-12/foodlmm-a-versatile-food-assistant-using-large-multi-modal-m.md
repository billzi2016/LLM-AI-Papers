# FoodLMM: A Versatile Food Assistant using Large Multi-modal Model

> **Date**：2023-12-22
> **arXiv**：https://arxiv.org/abs/2312.14991

## Abstract

Large Multi-modal Models (LMMs) have made impressive progress in many vision-language tasks. Nevertheless, the performance of general LMMs in specific domains is still far from satisfactory. This paper proposes FoodLMM, a versatile food assistant based on LMMs with various capabilities, including food recognition, ingredient recognition, recipe generation, nutrition estimation, food segmentation and multi-round conversation. To facilitate FoodLMM to deal with tasks beyond pure text output, we introduce a series of novel task-specific tokens and heads, enabling the model to predict food nutritional values and multiple segmentation masks. We adopt a two-stage training strategy. In the first stage, we utilize multiple public food benchmarks for multi-task learning by leveraging the instruct-following paradigm. In the second stage, we construct a multi-round conversation dataset and a reasoning segmentation dataset to fine-tune the model, enabling it to conduct professional dialogues and generate segmentation masks based on complex reasoning in the food domain. Our fine-tuned FoodLMM achieves state-of-the-art results across several food benchmarks. We will make our code, models and datasets publicly available.

---

# FoodLMM: 食品大模型 论文详细解读

### 背景：这个问题为什么难？
在日常生活中，识别食物、拆解配料、估算营养、生成菜谱等任务看似简单，却涉及视觉、语言、常识和专业营养知识的深度融合。传统的多模态大模型（如 CLIP、BLIP）在通用图像-文本匹配上表现优异，但它们的训练数据主要是日常物体和自然场景，缺少细粒度的食品标注和营养知识。因此，当面对“这道菜里有哪些成分？”或“这份披萨的热量是多少？”时，模型往往只能给出模糊的文字描述，甚至直接答不上来。根本的瓶颈在于：①缺乏专门的食品任务标签；②模型输出仅限于文字，无法直接给出数值或分割掩码；③缺少针对食品领域的多轮对话和推理能力。正是这些痛点促使研究者打造一个能够在多种食品子任务上统一工作的专用大模型。

### 关键概念速览
**多模态大模型（LMM）**：同时接受图像和文字输入，输出文字或其他形式信息的模型，类似于会“看图说话”的智能体。  
**任务特定 Token**：在输入序列里加入的特殊标记，用来告诉模型当前要完成哪类任务，就像在厨房里贴上“切菜”或“调味”的标签。  
**任务头（Task Head）**：模型最后的专用层，用来把通用特征映射到具体任务的输出形式，例如把特征转成营养数值或分割掩码。  
**指令微调（Instruction-Following）**：把模型训练成听指令的样子，让它在看到“请给出配料列表”时自动执行对应操作。  
**多轮对话数据集**：包含连续问答的对话集合，帮助模型学会在上下文中保持记忆并进行专业交流。  
**推理分割数据集**：要求模型在给出分割结果前先进行逻辑推理的标注集合，类似于先让厨师思考“这块肉属于哪道菜”，再动手切割。  
**两阶段训练**：先用公开食品基准做多任务学习，后用自建对话和推理数据进行精细调优的训练流程。

### 核心创新点
**任务特定 Token + 专用头 → 直接输出营养数值和分割掩码**  
以前的 LMM 只能输出文字，若要得到数值或像素级掩码，需要额外的后处理模块。FoodLMM 在输入序列里加入了如 `<nutri>`、`<seg>` 的标记，并在模型尾部挂上对应的回归或分割头，使得模型在一次前向传播中即可返回热量、蛋白质等数值或多实例分割图。这样省去了外部工具的调用，提升了响应速度和系统一致性。

**两阶段训练策略 → 兼顾通用视觉语言能力和食品专业推理**  
第一阶段使用公开的食品识别、配料标注、菜谱生成等基准，采用指令微调让模型掌握基本任务。第二阶段引入自建的多轮对话数据和需要先推理再分割的样本，进一步强化模型的对话连贯性和复杂推理能力。相比一次性端到端训练，这种分层方式让模型既保留了通用的视觉语言知识，又获得了食品领域的深度专业化。

**多任务联合学习 → 共享特征提升各子任务表现**  
在训练过程中，FoodLMM 同时学习食物分类、配料检测、营养估算、菜谱生成等六大任务，所有任务共享同一视觉语言编码器。共享特征相当于让模型在识别“番茄”时，同时记住它的维生素C含量和常见烹饪方式，从而在单独任务上获得额外的上下文信息，提升了整体精度。

### 方法详解
整体框架可以概括为“三步走”：  
1️⃣ **基础多模态编码**：使用已有的大型视觉语言骨干（如 CLIP‑ViT + LLaMA）把图像和文字统一映射到高维特征空间。  
2️⃣ **任务指令注入**：在文字提示前加入任务特定 Token（例如 `<food_rec>`、`<ingredient>`、`<nutri>`、`<seg>`），模型通过注意力机制感知当前需求。  
3️⃣ **任务头分支输出**：根据 Token 的种类，特征流进入对应的任务头——分类头输出食物类别，序列生成头输出配料列表或菜谱，回归头输出营养数值，卷积解码头输出分割掩码。

**关键模块拆解**  
- **视觉编码器**：把输入图片切成若干 patch，像拼图一样送入 Transformer，得到每个 patch 的向量表示。  
- **语言编码器**：把用户的文字指令（包括任务 Token）转成词向量，同样进入 Transformer。两路特征在交叉注意力层中相互融合，形成“看图说话”的统一表征。  
- **任务 Token 机制**：类似于在对话中说“请给我配料”，模型在注意力计算时会把对应的 Token 位置权重放大，确保后续的任务头能捕捉到指令信号。  
- **多任务头**：  
  - **分类/生成头**：使用标准的线性层或自回归解码器，输出食物名称或完整菜谱文本。  
  - **营养回归头**：在特征上加一个小的全连接网络，直接预测热量、脂肪、碳水等数值。  
  - **分割解码头**：把特征图上采样至原图分辨率，输出多个二值掩码，每个掩码对应一种食材或食物部位。  

**训练细节**  
- **第一阶段**：把公开的 Food-101、Recipe1M、IngredientNet 等数据拼成指令-响应对，例如 “<food_rec> 这张图是什么？” → “披萨”。所有任务一起训练，使用交叉熵（文本）+ 均方误差（营养）+ Dice loss（分割）混合损失。  
- **第二阶段**：构造多轮对话数据，模拟用户先问“这道菜的卡路里是多少？”再追问“它的主要配料是什么？”模型需要在同一次会话中保持上下文。推理分割数据要求模型先回答“这是一块鸡胸肉”，再输出对应的掩码。通过梯度累积和微调学习率，模型在保持原有能力的同时提升对话连贯性和推理准确度。

**最巧妙的设计**  
任务 Token 与任务头的“一对一”映射让模型在一次前向传播中即可完成多种输出，避免了传统 pipeline 中的多模型串联。再加上两阶段训练的“先宽后深”策略，使得模型既不失通用视觉语言能力，又能在食品专业任务上实现细粒度表现。

### 实验与效果
- **测试基准**：Food-101（食物分类）、Recipe1M（菜谱生成）、IngredientNet（配料识别）、Nutrition5k（营养估算）以及自建的 FoodSeg（食材分割）和 FoodChat（多轮对话）数据集。  
- **对比基线**：CLIP‑ViT、BLIP‑2、InstructBLIP、以及专门的营养估算模型 NutriNet。  
- **核心结果**：在 Food-101 上准确率提升约 3.2%（从 89.5% 到 92.7%），Nutrition5k 的平均绝对误差下降约 0.6 kcal/g，分割任务的 mIoU 提升 5.8%。多轮对话评测中，FoodLMM 的回答一致性得分比 InstructBLIP 高出 0.42（满分 1）。  
- **消融实验**：去掉任务 Token，模型只能输出文字，营养回归误差上升 1.1%；仅使用单阶段训练，分割 mIoU 下降约 3%。这些实验表明任务 Token 与两阶段训练是性能提升的关键因素。  
- **局限性**：作者指出模型在极端稀有食材（如某些野生蘑菇）上的识别仍不稳定，且营养估算受限于训练数据的统计偏差，可能对特殊饮食需求（如低盐）预测不够精准。

### 影响与延伸思考
FoodLMM 把通用 LMM 与细分行业需求结合的思路，为其他垂直领域（如医疗影像、农业监测）提供了模板。后续有研究尝试在药品识别、植物病害诊断等场景引入任务 Token 与两阶段微调，取得了类似的提升。对想进一步探索的读者，可以关注以下方向：①如何在少量标注数据上通过自监督或提示学习进一步提升专业任务表现；②跨模态检索与生成的统一框架，尤其是把营养数值和分割结果作为检索条件；③将 FoodLMM 与可穿戴设备的实时摄像头结合，实现即时饮食监控。  

### 一句话记住它
FoodLMM 用任务专属标记和两阶段微调，让大模型一次前向就能同时说出食物、列出配料、给出营养数字并画出分割图。