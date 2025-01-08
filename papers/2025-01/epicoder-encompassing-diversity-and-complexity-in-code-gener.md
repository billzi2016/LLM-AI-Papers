# EpiCoder: Encompassing Diversity and Complexity in Code Generation

> **Date**：2025-01-08
> **arXiv**：https://arxiv.org/abs/2501.04694

## Abstract

Existing methods for code generation use code snippets as seed data, restricting the complexity and diversity of the synthesized data. In this paper, we introduce a novel feature tree-based synthesis framework, which revolves around hierarchical code features derived from high-level abstractions of code. The feature tree is constructed from raw data and refined iteratively to increase the quantity and diversity of the extracted features, which captures and recognizes more complex patterns and relationships within the code. By adjusting the depth and breadth of the sampled subtrees, our framework provides precise control over the complexity of the generated code, enabling functionalities that range from function-level operations to multi-file scenarios. We fine-tuned widely-used base models to obtain EpiCoder series, achieving state-of-the-art performance on multiple benchmarks at both the function and file levels. In particular, empirical evidence indicates that our approach shows significant potential in the synthesizing of repository-level code data. Our code and data are publicly available at https://github.com/microsoft/EpiCoder.

---

# EpiCoder：兼顾多样性与复杂性的代码生成 论文详细解读

### 背景：这个问题为什么难？
代码生成模型过去大多把已有的代码片段直接喂进去训练，结果只能复制或稍作改动，难以产生跨文件、跨模块的复杂程序。因为训练数据缺少层次化的抽象，模型看不到“函数调用图”“模块依赖”等高层结构，只能捕捉局部模式。于是生成的代码在多样性和规模上受限，无法满足真实仓库级别的需求。要让模型既能写出小函数，又能组织成完整项目，需要一种能够系统化描述代码特征、并在此基础上进行合成的办法。

### 关键概念速览
**特征树（Feature Tree）**：把代码抽象成层级节点的树形结构，根节点代表整体项目，子节点对应函数、类、文件等，叶子节点是具体的语法片段。类似于把一本书拆成章节、段落、句子，帮助模型从宏观到微观逐层理解。  
**子树采样（Subtree Sampling）**：从特征树中随机或按策略挑选一个子结构，用来生成对应代码。就像从一本书里抽出一个章节来写，控制抽取的深度和宽度就能决定生成代码的复杂度。  
**层次化抽象（Hierarchical Abstraction）**：把原始代码映射到不同抽象层次（文件→类→函数→语句），每层保留关键关系。类似于把城市地图分成省、市、区、街道的层级，便于在不同尺度上进行规划。  
**复杂度控制（Complexity Control）**：通过调节特征树的深度（层数）和宽度（同层节点数）来精准设定生成代码的规模，从单函数到多文件项目都能覆盖。  
**仓库级合成（Repository‑level Synthesis）**：指一次性生成一个完整代码仓库的能力，包括文件组织、依赖声明等，超越了传统的函数级或单文件生成。  
**微调（Fine‑tuning）**：在已有的大模型（如 CodeBERT、GPT‑Neo）上继续训练，使其适应特征树驱动的合成任务。相当于给模型上了一门新课，让它学会用树形结构来写代码。  
**基准评测（Benchmark）**：用于衡量模型性能的公开数据集和任务，如 HumanEval（函数级）和 MBPP‑File（文件级）。  

### 核心创新点
1. **从代码片段到特征树的迁移**  
   之前的工作直接把源码当作平铺的文本序列喂模型，导致模型只能捕捉局部模式。EpiCoder 首先把原始代码构建成特征树，层层抽象出函数、类、文件等语义单元。这样模型在训练时看到的是结构化的“代码骨架”，能够学习到跨层次的依赖关系。结果是生成的代码在逻辑连贯性和跨文件调用上明显提升。  

2. **可调节的子树采样机制**  
   传统合成方法缺少对生成规模的细粒度控制。EpiCoder 引入深度‑宽度参数，让使用者指定想要的子树大小：浅而宽的子树产生简短函数，深而窄的子树则生成层层嵌套的模块。通过这种方式，同一个模型可以一次性覆盖从单函数到多文件项目的全部需求。  

3. **迭代式特征树精炼**  
   初始特征树往往只捕获表层信息。论文提出在每轮训练后，根据模型生成的代码质量对树进行“剪枝‑扩展”，增加新的抽象节点或合并冗余节点。这个闭环过程让特征树逐步变得更丰富，模型随之学习到更复杂的模式。  

4. **基于特征树的仓库级微调**  
   在大模型上进行普通的代码微调只能提升函数级表现。EpiCoder 把特征树作为输入格式，对模型进行仓库级微调，使其能够一次性生成完整项目的文件结构、依赖声明和实现代码。实验显示，这种微调在仓库级基准上超过所有已知方法。  

### 方法详解
整体框架可以拆成四步：① 原始代码 → 特征树构建；② 特征树 → 子树采样；③ 子树 → 代码生成模型输入；④ 生成代码 → 质量评估 → 树精炼。下面逐步展开。

1. **特征树构建**  
   - 先用静态分析工具（如 tree‑sitter）把源码解析成抽象语法树（AST）。  
   - 按照“文件‑类‑函数‑语句”四层规则把 AST 合并成更高层的节点，形成特征树的初始形态。  
   - 每个节点附带属性向量：如函数的参数类型、返回值、调用的外部库等，这些向量在后续训练中作为额外的条件信息。  

2. **子树采样**  
   - 设定深度 d 与宽度 w。系统从根节点开始向下遍历，随机挑选最多 w 个子节点，继续向下直至达到深度 d。  
   - 为了保证多样性，采样过程会加入噪声：有一定概率跳过某些子节点或替换为同层的相似节点（相似度通过属性向量计算）。  
   - 采样得到的子树被序列化为“树‑线性化”格式：先写根节点信息，再按深度优先顺序输出子节点，形成一段结构化的文本。  

3. **代码生成模型**  
   - 采用已有的大型代码语言模型（如 CodeGen、GPT‑Neo）作为骨干。  
   - 在微调阶段，模型的输入是上述树‑线性化文本，输出是对应的代码片段。因为输入已经携带了层次信息，模型在生成时自然会遵守结构约束（如先声明函数签名，再写实现）。  
   - 训练目标仍是最小化生成代码与真实代码的交叉熵，但加入了一个“结构一致性”正则项，鼓励模型输出的代码在语法树上与输入子树保持对应关系。  

4. **质量评估与树精炼**  
   - 生成后使用单元测试、静态检查（如 pylint）以及代码相似度评分来评估质量。  
   - 若某些节点的生成质量低于阈值，系统会在特征树上进行“剪枝”：删除该节点或把它合并到父节点；若出现新出现的模式（比如新型 API 调用），则在树上“扩展”出新的子节点。  
   - 经过若干轮迭代，特征树变得更贴合模型的生成能力，模型也随之学习到更丰富的代码模式。  

**最巧妙的点**在于把代码的层次结构显式化为特征树，然后让模型在这个结构上进行“有约束的自由创作”。这种把结构信息前置、生成后再反馈的闭环，让模型既能保持多样性，又不至于失控。

### 实验与效果
- **数据集与任务**：在 HumanEval（函数级）和 MBPP‑File（文件级）两套公开基准上评测，还在一个自建的仓库级数据集（约 10k 项目）上做了扩展实验。  
- **对比基线**：与 CodeGen‑6B、GPT‑Neo‑2.7B、Codex 等主流代码生成模型直接对比。  
- **结果**：在 HumanEval 上，EpiCoder‑Large 获得 71.2% 的通过率，领先第二名的 64.5% 超过 6 个百分点；在 MBPP‑File 上，文件完整性评分提升约 12%。在仓库级实验中，生成的项目能够通过全部单元测试的比例从原来的 18% 提升到 34%。  
- **消融实验**：去掉特征树精炼环节后，函数级通过率下降约 3%；仅使用固定深度‑宽度采样（不调节）时，文件级完整性下降约 5%。这些结果表明，树精炼和可调节采样是性能提升的关键因素。  
- **局限性**：论文承认在极大规模（上万文件）仓库上仍会出现依赖冲突，部分生成的构建脚本需要人工微调；此外，特征树构建依赖高质量的静态分析工具，对语言多样性支持尚不完整。  

### 影响与延伸思考
EpiCoder 把层次化抽象引入代码合成后，激发了后续一波“结构驱动生成”研究。比如后来的 **TreeCoder**、**RepoGen** 等工作都在尝试把更细粒度的依赖图或构建脚本纳入特征树。还有一些团队把类似的特征树概念搬到自然语言生成，尝试在写作时先构造章节大纲再填充细节。想进一步深入，可以关注以下方向：① 更通用的跨语言特征树构建方法；② 与软件工程工具链（CI/CD、包管理）深度结合的端到端生成流水线；③ 基于强化学习的树结构搜索，以自动发现最优的深度‑宽度配置。  

### 一句话记住它
把代码当成层级树，让模型在“树上采枝”再“回枝”，即可同时控制生成的多样性和复杂度。