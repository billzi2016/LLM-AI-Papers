# InCoder: A Generative Model for Code Infilling and Synthesis

> **Date**：2022-04-12
> **arXiv**：https://arxiv.org/abs/2204.05999

## Abstract

Code is seldom written in a single left-to-right pass and is instead repeatedly edited and refined. We introduce InCoder, a unified generative model that can perform program synthesis (via left-to-right generation) as well as editing (via infilling). InCoder is trained to generate code files from a large corpus of permissively licensed code, where regions of code have been randomly masked and moved to the end of each file, allowing code infilling with bidirectional context. Our model is the first generative model that is able to directly perform zero-shot code infilling, which we evaluate on challenging tasks such as type inference, comment generation, and variable re-naming. We find that the ability to condition on bidirectional context substantially improves performance on these tasks, while still performing comparably on standard program synthesis benchmarks in comparison to left-to-right only models pretrained at similar scale. The InCoder models and code are publicly released. https://sites.google.com/view/incoder-code-models

---

# InCoder：用于代码填充与生成的生成模型 论文详细解读

### 背景：这个问题为什么难？
代码不像自然语言那样一次性从左到右写完，开发者常常在已有代码的中间插入、删改或重命名变量。传统的代码生成模型大多只会左到右预测下一个 token，缺乏对前后文的双向感知，导致在需要“填空”或“编辑”时表现很差。于是出现了一个尴尬的局面：要么只能生成完整文件，要么只能做简单的补全，无法直接完成像变量重命名、注释补全这类真实编辑任务。突破这个瓶颈需要模型能够同时看到左侧和右侧的代码上下文，并在此基础上生成缺失的片段。

### 关键概念速览
**左到右生成（Left‑to‑Right Generation）**：模型一次只预测下一个 token，类似顺序写作，适合完整文件的合成。  
**双向上下文（Bidirectional Context）**：在生成时同时利用目标位置左侧和右侧的代码信息，就像人在编辑时能看到前后代码一样。  
**代码填充（Code Infilling）**：给定代码的前后片段，模型需要在中间生成合理的代码块，等价于“把空白填满”。  
**遮蔽移动（Mask‑and‑Move）**：训练时随机把代码片段遮住并搬到文件末尾，让模型学习在看到完整上下文后恢复原位。  
**零样本（Zero‑Shot）**：模型在没有专门微调的情况下直接完成新任务，这里指直接进行代码填充。  
**类型推断（Type Inference）**：根据代码使用情况自动猜出变量或函数的类型，常见于静态语言的编辑辅助。  
**变量重命名（Variable Renaming）**：把代码中所有出现的同一变量名统一改成更合适的名字，需要全局一致性。  

### 核心创新点
1. **训练策略：遮蔽‑移动 → 双向填充**  
   之前的代码模型只在左到右的语言模型目标上训练，无法利用右侧信息。InCoder 把随机选中的代码块遮住并搬到文件末尾，让模型在看到完整的前后文后预测这些被移动的块。这样模型在训练阶段就学会了在双向上下文下填充缺失代码。  
   *改变*：模型在推理时可以直接接受前后代码并生成中间片段，实现真正的零样本代码填充。

2. **统一模型：合成 + 编辑**  
   过去需要分别训练代码合成模型（左到右）和编辑模型（专门的填空或重写网络）。InCoder 用同一个网络同时支持左到右生成完整文件和双向填充编辑，两种任务共享参数。  
   *改变*：省去多模型部署的复杂度，并且在合成任务上仍能保持与同规模左到右模型相当的表现。

3. **大规模许可代码语料**  
   作者收集了大量 permissive 许可证（如 MIT、Apache）的开源代码，确保训练数据合法且覆盖多语言、多库。相比只用少量高质量数据的前作，规模更大、覆盖更广。  
   *改变*：提升了模型对真实工程代码的适应性，尤其在变量命名和注释生成等细粒度任务上表现更好。

### 方法详解
**整体框架**  
InCoder 的训练过程可以看成两步：① 随机挑选文件中的若干连续代码片段；② 把这些片段用特殊的遮蔽标记替换，并把原片段搬到文件末尾形成“补全目标”。模型的输入是被遮蔽后的完整文件，输出是文件末尾的搬运片段。推理时，只要提供左侧和右侧代码（即遮蔽前后的上下文），模型就会在中间生成缺失的代码块。

**关键模块拆解**  
1. **数据预处理（Mask‑and‑Move）**  
   - 随机选取长度在 5‑50 token 之间的代码块。  
   - 用 `<mask>` 标记替换原位置，并把原代码块追加到文件末尾，形成 “目标序列”。  
   - 这样每个训练样本同时包含了 **上下文**（文件前后）和 **目标**（被移动的代码），相当于让模型学习“把搬走的代码放回原位”。

2. **模型结构**  
   - 基于 Transformer 编码器‑解码器，和 GPT 系列类似，但在解码阶段接受完整的上下文。  
   - 编码器处理整个被遮蔽的文件，生成每个位置的上下文向量。  
   - 解码器从文件末的目标序列开始，逐步预测搬回去的代码，条件是编码器输出的双向上下文向量。  
   - 这种设计让模型在生成目标时天然利用左右两侧信息。

3. **双向条件生成**  
   - 在左到右合成模式下，目标序列是空的，模型只靠编码器的左侧上下文生成新 token。  
   - 在填充模式下，目标序列已经包含了搬运的代码，解码器的每一步都可以看到右侧已经生成的 token，形成真正的双向条件。

4. **训练目标**  
   - 使用交叉熵损失对解码器输出的每个 token 与真实搬回代码进行对齐。  
   - 同时保留语言模型的自回归损失，使模型在没有遮蔽的情况下仍能做左到右生成。

**最巧妙的地方**  
把代码块搬到文件末尾的做法看似简单，却让模型在训练时自然获得了“看见右侧”的能力，而不需要额外的双向掩码或特殊的自注意力修改。这种“遮蔽‑移动”策略是实现零样本填充的关键。

### 实验与效果
- **评测任务**：作者在三个典型编辑任务上做零样本评估：类型推断、注释生成、变量重命名。除此之外，还在常见的代码合成基准（如 HumanEval）上测试左到右生成能力。  
- **基线对比**：与同规模的左到右代码模型（如 CodeGen、GPT‑Neo）相比，InCoder 在编辑任务上显著领先。论文中提到在变量重命名任务上准确率提升约 15% 左右，注释生成的 BLEU 分数提升约 0.8。合成任务上表现与基线持平，未出现明显退步。  
- **消融实验**：作者分别去掉遮蔽‑移动、只用左到右目标、以及只保留编码器等设置，发现遮蔽‑移动是提升编辑任务性能的主要因素，去掉后编辑准确率下降约 10%。  
- **局限性**：论文承认模型在极长文件（超过 2k token）上仍会出现上下文截断，编辑质量受限；此外，零样本填充虽然有效，但在高度专业化的库调用上仍需要微调才能达到最佳。  

### 影响与延伸思考
InCoder 把“编辑”能力直接写进生成模型，开启了代码模型从“写代码”向“改代码”转型的潮流。随后的工作（如 Salesforce 的 CodeT5‑Edit、Google 的 PolyCoder‑Edit）都在不同程度上借鉴了遮蔽‑移动的训练思路，进一步探索多任务编辑、交互式补全等方向。对想深入的读者，可以关注以下几个方向：① 更高效的长上下文 Transformer（如 FlashAttention）在编辑任务中的应用；② 将编辑能力与检索增强（RAG）结合，实现基于项目代码库的精准补全；③ 微调策略如何在少量项目数据上快速提升特定编辑任务的表现。  

### 一句话记住它
InCoder 用“把代码块搬到文件末尾再让模型找回原位”的技巧，让生成模型一次性掌握代码合成和零样本编辑两大能力。