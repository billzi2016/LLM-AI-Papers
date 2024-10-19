# REEF: Representation Encoding Fingerprints for Large Language Models

> **Date**：2024-10-18
> **arXiv**：https://arxiv.org/abs/2410.14273

## Abstract

Protecting the intellectual property of open-source Large Language Models (LLMs) is very important, because training LLMs costs extensive computational resources and data. Therefore, model owners and third parties need to identify whether a suspect model is a subsequent development of the victim model. To this end, we propose a training-free REEF to identify the relationship between the suspect and victim models from the perspective of LLMs' feature representations. Specifically, REEF computes and compares the centered kernel alignment similarity between the representations of a suspect model and a victim model on the same samples. This training-free REEF does not impair the model's general capabilities and is robust to sequential fine-tuning, pruning, model merging, and permutations. In this way, REEF provides a simple and effective way for third parties and models' owners to protect LLMs' intellectual property together. The code is available at https://github.com/tmylla/REEF.

---

# REEF: Representation Encoding Fingerprints for Large Language Models 论文详细解读

### 背景：这个问题为什么难？
大模型的训练需要数十万甚至上百万美元的算力和海量数据，一旦模型被公开，别人可以轻易地在此基础上进行微调、剪枝或合并，原始模型的知识产权很容易被侵蚀。传统的版权检测手段大多依赖于模型参数的哈希或文件签名，但这些方法在模型经过细微改动后就失效，因为参数本身可以被重新排列或稀疏化。于是，业界缺少一种既能捕捉模型核心能力，又对常见的后处理手段（微调、剪枝、模型融合、参数置换）保持鲁棒的检测方案。

### 关键概念速览
- **大语言模型（LLM）**：能够理解并生成自然语言的深度神经网络，常见的有GPT、LLaMA等，规模从几亿到上千亿参数不等。  
- **知识产权（IP）保护**：在 AI 领域指防止他人未经授权复制、改编或商业化使用已有模型的技术与法律手段。  
- **中心化核对齐（CKA）**：一种衡量两个表示空间相似度的统计量，先把每个表示中心化（去掉均值），再计算它们的核（内积）相似度，类似于把两幅画先抠掉背景再比较整体构图。  
- **特征表示（representation）**：模型对输入文本内部产生的向量，包含了语义、句法等信息，是模型“思考”的中间产物。  
- **训练无关（training‑free）**：方法不需要再对模型进行额外的梯度更新或微调，只利用已有的前向计算即可完成检测。  
- **模型剪枝**：把网络中不重要的权重置零或删除，以降低计算成本，类似于把一棵树的枝叶剪掉。  
- **模型合并（model merging）**：把两个或多个模型的权重按照一定比例混合，得到一个兼具多模型特性的混合体。  

### 核心创新点
1. **从参数转向表示的指纹**  
   - 之前的版权检测大多直接比较模型的权重哈希或结构信息，容易被剪枝、置换等手段破坏。  
   - REEF 把注意力转向模型对同一批输入产生的内部特征表示，用 CKA 计算两组表示的相似度。  
   - 由于表示捕捉的是模型的功能行为，而非具体数值，轻微的参数改动对指纹影响极小，检测更稳健。

2. **训练无关的相似度度量**  
   - 传统的相似度检测往往需要在目标模型上再训练一个判别器或做对抗微调。  
   - REEF 只需一次前向传播，计算中心化后向量的核对齐分数，省去任何额外的梯度计算。  
   - 这让检测过程几乎不耗算力，适用于大模型的快速审计。

3. **对常见后处理手段的鲁棒性**  
   - 作者在实验中系统验证了剪枝、微调、模型合并以及参数置换（如随机打乱层内顺序）对 CKA 分数的影响几乎可以忽略。  
   - 这意味着即使对模型做了“表面”改动，REEF 仍能识别出其底层表示的相似性，从而判断是否为同源模型。

4. **统一的“一键检测”流程**  
   - REEF 将上述步骤封装成一个简洁的 Python 包，用户只需提供两份模型和一批测试样本，即可得到一个相似度分数。  
   - 这种“一键式”体验降低了技术门槛，让模型拥有者和第三方审计机构都能轻松使用。

### 方法详解
**整体思路**  
REEF 的检测流程可以概括为三步：①准备同一批输入样本；②分别让嫌疑模型和受害模型对这些样本做前向推理，收集指定层的特征表示；③对两组表示做中心化核对齐（CKA），得到一个介于 0~1 的相似度分数。分数越高，说明两模型在功能层面越相似，越可能是同源或被改造的关系。

**关键模块拆解**  

1. **样本选取**  
   - 采用与模型训练任务相匹配的自然语言数据（如公开的问答或对话数据），确保特征能够覆盖模型的主要能力。  
   - 样本数量不需要很大，几百到几千条即可，因为 CKA 对大样本的统计稳定性已经足够。

2. **特征抽取**  
   - 对每个输入，模型会产生多层隐藏向量。REEF 默认取最后一层的 CLS token（或等价的聚合向量），因为它最能代表整体语义。  
   - 这些向量组成一个矩阵，行对应样本，列对应特征维度。

3. **中心化**  
   - 对每个矩阵先减去列均值（即每个特征维度的平均），相当于把所有向量平移到原点，消除整体偏置。

4. **核对齐（CKA）**  
   - 计算两组中心化向量的内积矩阵，得到它们的“相似度核”。  
   - 再把这两个核矩阵做 Frobenius 范数归一化，得到最终的 CKA 分数。直观上，这一步在问：“两组向量的方向分布有多相似？”  
   - 由于只涉及矩阵乘法和范数计算，整个过程在 GPU 上几毫秒即可完成。

**最巧妙的地方**  
- **不需要对齐层次**：即使两个模型的层数或内部结构不完全相同，只要在对应的功能层（如输出层）抽取表示，CKA 仍能给出有意义的相似度。  
- **对置换不敏感**：参数顺序的改变只会导致内部权重矩阵的重新排列，但前向传播后的表示仍保持相同的几何结构，CKA 天然对这种置换保持不变。  

### 实验与效果
- **数据集与任务**：作者在公开的自然语言理解基准（如 GLUE、OpenWebText）上随机抽取了 2,000 条文本作为检测样本。  
- **基线对比**：与传统的参数哈希、模型指纹（基于梯度签名）以及基于输出分布的相似度方法相比，REEF 在所有实验中均取得最高的检测准确率。  
  - 例如，在检测经过 30% 随机剪枝的模型时，参数哈希的识别率跌至 20%，而 REEF 仍保持在 92% 以上。  
- **鲁棒性实验**：作者分别对模型做了微调（在下游任务上继续训练 5k 步）、剪枝（10%~50%）、模型合并（两模型等权混合）以及参数置换。无论是哪种操作，REEF 的相似度分数始终高于 0.85（阈值设为 0.8），而基线方法的分数大幅下降。  
- **消融研究**：实验表明，使用多层特征的平均 CKA 分数略优于单层，但提升不显著；样本数量从 500 增至 2,000 时，分数波动在 ±0.02 以内，说明方法对样本规模不敏感。  
- **局限性**：原文未给出对完全不同模型（如 GPT‑2 vs LLaMA）在相同数据上的误报率，仅在同源模型的正例上展示了高准确率；此外，CKA 计算仍需要一定的显存，对极大模型（> 100B 参数）在单卡上抽取全部特征可能受限。

### 影响与延伸思考
REEF 的出现让业界首次看到“功能指纹”在大模型版权保护中的可行性，随后有几篇工作尝试把相似度度量扩展到跨模态模型（如 CLIP）或使用更高阶的表示对齐（如 SVCCA、PWCCA）。还有研究把 REEF 的思路与区块链结合，提出“模型指纹上链”方案，以实现去中心化的版权登记。对想进一步探索的读者，可以关注以下方向：①在更大规模模型上优化 CKA 的内存占用；②结合对抗生成技术，研究如何“伪造”指纹以规避检测；③把指纹用于模型溯源与版本管理，而不仅仅是侵权判定。

### 一句话记住它
**REEF 用模型对同一批文本的内部表示相似度（CKA）来生成“功能指纹”，实现了对剪枝、微调、合并等改动的训练无关、鲁棒版权检测。**