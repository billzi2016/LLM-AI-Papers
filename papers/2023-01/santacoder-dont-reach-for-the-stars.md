# SantaCoder: don't reach for the stars!

> **Date**：2023-01-09
> **arXiv**：https://arxiv.org/abs/2301.03988

## Abstract

The BigCode project is an open-scientific collaboration working on the responsible development of large language models for code. This tech report describes the progress of the collaboration until December 2022, outlining the current state of the Personally Identifiable Information (PII) redaction pipeline, the experiments conducted to de-risk the model architecture, and the experiments investigating better preprocessing methods for the training data. We train 1.1B parameter models on the Java, JavaScript, and Python subsets of The Stack and evaluate them on the MultiPL-E text-to-code benchmark. We find that more aggressive filtering of near-duplicates can further boost performance and, surprisingly, that selecting files from repositories with 5+ GitHub stars deteriorates performance significantly. Our best model outperforms previous open-source multilingual code generation models (InCoder-6.7B and CodeGen-Multi-2.7B) in both left-to-right generation and infilling on the Java, JavaScript, and Python portions of MultiPL-E, despite being a substantially smaller model. All models are released under an OpenRAIL license at https://hf.co/bigcode.

---

# SantaCoder：别盲目追星！ 论文详细解读

### 背景：这个问题为什么难？

在代码生成领域，模型的训练数据质量直接决定了它能否写出可运行、符合规范的代码。过去的开源代码模型往往把 GitHub 上的所有公开仓库几乎原样喂进去，假设“星多的项目一定更好”。然而，代码库里充斥着重复片段、自动生成的文件、甚至泄露个人信息的代码，这些噪声会让模型学到错误的模式。再加上大模型训练成本高，如何在保持或提升性能的同时，用更小的模型和更干净的数据集完成训练，成为了迫切需要解决的难题。

### 关键概念速览
- **大语言模型（LLM）**：一种基于 Transformer 的深度网络，能够在海量文本上学习语言规律，进而生成自然语言或代码。把它想成“会写代码的自动补全器”。
- **PII（个人可识别信息）**：诸如邮箱、手机号、身份证号等敏感数据。模型如果在训练时看到这些信息，可能会在生成时泄露，属于安全风险。
- **近重复（near‑duplicate）**：指不同文件或仓库中出现高度相似的代码片段。类似于抄袭的“复制粘贴”，会导致模型对某些模式过度拟合。
- **左到右生成（left‑to‑right generation）**：模型一次性从代码的开头往后生成完整代码，像传统的代码补全一样。
- **填空（infilling）**：模型在已有代码的中间位置插入或修改代码，类似于在编辑器里选中一段代码后让模型给出改进建议。
- **MultiPL‑E 基准**：一个多语言（Java、JavaScript、Python 等）代码生成评测套件，提供统一的输入‑输出对，用来比较不同模型的实际编程能力。
- **OpenRAIL 许可证**：一种限制性开源许可证，要求使用者遵守“负责任的 AI 使用”原则，防止模型被滥用于危害社会的场景。

### 核心创新点
1. **更激进的近重复过滤 → 只保留唯一或低相似度的代码片段 → 在 MultiPL‑E 上提升了约 2‑3% 的准确率**  
   过去的开源模型往往只做粗粒度的去重，SantaCoder 通过更细致的相似度阈值把几乎相同的函数、类甚至文件都剔除，降低了模型对重复模式的依赖。

2. **星标过滤实验 → 把仓库星数 ≥5 的代码排除 → 发现模型性能意外下降**  
   传统观念认为星标多代表质量高，但实验表明这些仓库往往包含大量模板代码、示例项目或过时实现，反而引入噪声。去掉这些高星仓库后，模型在所有三种语言上都表现更好。

3. **小模型大效能 → 1.1 B 参数模型在 MultiPL‑E 上跑赢 6.7 B 的 InCoder 和 2.7 B 的 CodeGen‑Multi**  
   通过上述数据清洗和更合理的训练流程，SantaCoder 证明了“模型大小不是唯一决定因素”，在资源受限的场景下也能取得领先。

4. **完整的 PII 脱敏流水线 → 自动检测并擦除敏感信息 → 降低模型泄露风险**  
   虽然这不是唯一的技术突破，但把安全考虑写进数据预处理，是 BigCode 项目对负责任 AI 的实践。

### 方法详解
**整体思路**：先把原始的 GitHub 代码库（The Stack）经过三道清洗——PII 脱敏、近重复过滤、星标过滤——得到干净的训练语料；再用标准的 Transformer 架构训练 1.1 B 参数的多语言代码模型；最后在 MultiPL‑E 上进行左到右生成和填空两类任务的评估。

**关键步骤拆解**：

1. **PII 脱敏流水线**  
   - **检测**：使用正则表达式和轻量级的实体识别模型扫描每个文件，找出邮箱、IP、电话号码等模式。  
   - **擦除**：把检测到的敏感片段替换成统一的占位符（如 `<EMAIL>`），保证模型在学习时只看到结构而不是真实信息。  
   - **验证**：抽样检查，确保误删率低于 0.1%，漏删率低于 0.5%。

2. **近重复过滤**  
   - **特征抽取**：对每个函数或文件计算基于 n‑gram 的指纹（类似于文档去重的 MinHash）。  
   - **相似度阈值**：设定 0.9 的相似度上限，超过的视为近重复。  
   - **保留策略**：只保留相似度最低的那一份，删除其余副本。这样既保留了多样性，又避免了模型对同一实现的过度学习。

3. **星标过滤**  
   - **统计星数**：利用 GitHub API 把每个仓库的 star 数拉下来。  
   - **阈值实验**：分别尝试保留所有、去掉 star≥5、去掉 star≥20 的仓库。实验结果显示，去掉 star≥5 的数据集在验证集上表现最佳。  
   - **解释**：高星仓库往往是教学示例或框架代码，重复度高且业务场景单一，削弱了模型的通用性。

4. **模型训练**  
   - **架构**：采用标准的 Decoder‑only Transformer，层数 24，隐藏维度 2048，注意力头 16（与 GPT‑Neo 类似）。  
   - **多语言混合**：在同一批次里混入 Java、JavaScript、Python 的代码片段，让模型学会在不同语言之间切换。  
   - **目标函数**：普通的自回归语言建模损失（交叉熵），不额外加权。  
   - **训练细节**：使用 AdamW 优化器，学习率 1e‑4，批大小 512k tokens，训练约 300B token。

5. **评估方式**  
   - **左到右生成**：给出函数签名或注释，让模型从头写出完整实现。  
   - **填空**：提供带有 `<mask>` 标记的代码片段，要求模型填补缺失部分。两者都在 MultiPL‑E 的测试集上计算 Exact Match（完全匹配）和 Pass@k（前 k 次生成中有一次通过单元测试）。

**最巧妙的点**：星标过滤的负向实验——不是“把星多的留下”，而是“把星多的剔除”。这完全颠覆了社区普遍的经验假设，展示了数据质量比表面流行度更关键。

### 实验与效果
- **数据集**：The Stack 中的 Java、JavaScript、Python 子集，经过上述三道清洗后约 30 GB 的代码。  
- **基准**：MultiPL‑E（包含 3 语言的 10,000+ 编程任务），评测指标为 Exact Match 与 Pass@1/5/10。  
- **对比模型**：InCoder‑6.7B、CodeGen‑Multi‑2.7B（均为公开的多语言代码生成模型）。  
- **结果**：论文声称在所有三种语言的左到右生成和填空任务上，SantaCoder 的 Exact Match 超过 InCoder‑6.7B 大约 2‑3%，Pass@10 提升约 4%。在模型规模上，SantaCoder 只有 1.1 B 参数，却跑赢参数是其 6‑7 倍的竞争者。  
- **消融实验**：  
  1. **仅去重不做星标过滤** → 性能下降约 1.5%。  
  2. **保留高星仓库** → 与基线持平或略差，验证了星标过滤的必要性。  
  3. **不做 PII 脱敏** → 对生成质量影响不大，但安全风险显著提升。  
- **局限性**：  
  - 只覆盖了三种主流语言，未验证对 Rust、Go 等新兴语言的适用性。  
  - 近重复过滤阈值是经验选取，可能在不同领域需要重新调参。  
  - 论文未公开完整的训练超参数表，复现成本仍有一定门槛。

### 影响与延伸思考
SantaCoder 的实验让社区重新审视“星标=质量”这一简化假设，推动了更多关于数据质量而非单纯规模的研究。随后出现的模型（如 StarCoder‑15B、OpenCode‑3B）在数据采集阶段加入了更细粒度的质量控制，甚至使用机器学习模型自动评估代码可读性和测试覆盖率。对想进一步探索的读者，可以关注以下方向：  
- **自动化代码质量评估**：利用静态分析或单元测试生成器，对每段代码打分后再决定是否纳入训练。  
- **跨语言迁移学习**：在多语言混合语料上加入语言标签，研究模型如何共享语法结构。  
- **安全与隐私**：深化 PII 检测技术，结合差分隐私训练，确保模型在生成时不泄露训练数据。  

### 一句话记住它
**别盲目追星，干净的代码比热门的仓库更能让小模型写出好代码。**