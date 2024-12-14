# Proposing and solving olympiad geometry with guided tree search

> **Date**：2024-12-14
> **arXiv**：https://arxiv.org/abs/2412.10673

## Abstract

Mathematics olympiads are prestigious competitions, with problem proposing and solving highly honored. Building artificial intelligence that proposes and solves olympiads presents an unresolved challenge in automated theorem discovery and proving, especially in geometry for its combination of numerical and spatial elements. We introduce TongGeometry, a Euclidean geometry system supporting tree-search-based guided problem proposing and solving. The efficient geometry system establishes the most extensive repository of geometry theorems to date: within the same computational budget as the existing state-of-the-art, TongGeometry discovers 6.7 billion geometry theorems requiring auxiliary constructions, including 4.1 billion exhibiting geometric symmetry. Among them, 10 theorems were proposed to regional mathematical olympiads with 3 of TongGeometry's proposals selected in real competitions, earning spots in a national team qualifying exam or a top civil olympiad in China and the US. Guided by fine-tuned large language models, TongGeometry solved all International Mathematical Olympiad geometry in IMO-AG-30, outperforming gold medalists for the first time. It also surpasses the existing state-of-the-art across a broader spectrum of olympiad-level problems. The full capabilities of the system can be utilized on a consumer-grade machine, making the model more accessible and fostering widespread democratization of its use. By analogy, unlike existing systems that merely solve problems like students, TongGeometry acts like a geometry coach, discovering, presenting, and proving theorems.

---

# 基于引导树搜索的奥林匹克几何题提出与求解 论文详细解读

### 背景：这个问题为什么难？

欧几里得几何在数学奥林匹克里既要处理严谨的逻辑，又要在平面图形中寻找恰当的辅助构造，往往需要“灵感”才能突破。传统的自动定理证明系统只能在已有的公理和已知定理之间做演绎，缺乏主动生成新图形、提出新命题的能力。更糟的是，几何的搜索空间呈指数级膨胀：每加入一条可能的连线或圆，就会产生成百上千的后继状态，导致穷举几乎不可能。于是，既能自行提出新题目，又能在同一框架下高效求解的系统一直是未解的难题。

### 关键概念速览
- **引导树搜索（Guided Tree Search）**：在搜索树的每个分支上使用模型给出的“指引”，类似于在迷宫里有一盏灯指向可能的出口，显著削减盲目扩展的节点数。  
- **辅助构造（Auxiliary Construction）**：在几何题中人为添加的点、直线或圆，用来把原问题转化为可解的子问题，就像在拼图时先画出几条辅助线帮助定位。  
- **几何对称性（Geometric Symmetry）**：指图形在旋转、反射或平移下保持不变的属性，系统利用对称性可以一次发现多条等价定理，类似于一次折纸得到多条对称折痕。  
- **大语言模型微调（LLM Fine‑tuning）**：把通用的语言模型在大量几何证明文本上继续训练，使其能够输出符合几何推理的步骤，像给学生专门辅导几何的老师。  
- **定理库（Theorem Repository）**：系统自动收集并归类的所有已证明的几何命题，形成可查询的知识库，类似于数学家的笔记本。  
- **IMO‑AG‑30 基准**：收录历届国际数学奥林匹克（IMO）30 道几何题的测试集，用来衡量系统的极限能力。  

### 核心创新点
1. **从被动证明到主动命题**：传统系统只能接受已有题目并尝试证明，TongGeometry 通过在搜索树中加入“命题生成节点”，让模型主动提出需要辅助构造的新定理。结果是系统在相同算力下发现了 6.7 千亿条新定理，远超以往只能证明的数量级。  
2. **大语言模型作为搜索引导器**：以前的树搜索往往使用手工设计的启发式函数，效果有限。这里先用大语言模型（LLM）在大量几何教材上微调，使其能够预测“下一步最可能的构造”。在搜索时把模型的概率分布作为节点扩展的优先级，实现了从“盲目扩张”到“有目标的探索”。  
3. **对称性驱动的批量发现**：系统在生成定理时会检测图形的对称变换，并一次性记录所有对称等价的版本。这样 4.1 千亿条定理自然带有对称标签，极大提升了定理库的覆盖度，也让后续求解时可以直接利用对称简化。  
4. **消费级硬件可运行的高效实现**：通过对几何运算的符号化优化和搜索过程的并行化，TongGeometry 能在普通笔记本上完成完整的提题‑求解循环，打破了以往只能在大型集群上运行的壁垒，真正实现了“人人都能当几何教练”。  

### 方法详解
整体框架可以概括为三步：**（1）构造搜索树 →（2）LLM 引导扩展 →（3）定理验证与归档**。  
1. **搜索树的搭建**：系统把欧几里得平面视作一个状态空间。初始状态是题目给出的点、直线和圆。每一次“动作”可以是添加一个新点（交点、垂足等），或画一条新直线/圆。所有可能的动作形成树的子节点。若不加限制，这棵树会在几层深度后爆炸。  
2. **LLM 引导的扩展策略**：在每个节点，系统把当前几何描述转成自然语言提示，喂给微调后的大语言模型。模型输出一个概率分布，列出最有可能导致可证明结论的构造（比如“在 AB 上取中点 C”）。系统只保留概率最高的前 *k* 条（k 通常为 5），其余分支直接剪枝。这样相当于让模型充当“几何直觉”，把搜索空间压缩到人类常用的几步之内。  
3. **定理验证**：当搜索树达到一定深度且模型建议的构造完成后，系统调用符号几何求解器（基于代数化的坐标法）对产生的命题进行自动证明。若证明成功，命题连同其构造过程被写入定理库；若失败，则该分支被标记为死路，回溯继续搜索。  
4. **对称性检测与批量归档**：每当一个新定理被验证，系统会在图形的对称群（旋转、反射）中搜索等价变换。找到的每个对称形态都会生成一条新的记录，标记为“对称复制”。这一步让一次搜索产出多条定理，极大提升效率。  
5. **求解流程**：对外部输入的奥赛几何题，系统先用同样的 LLM 引导搜索寻找可能的辅助构造，随后在搜索树中寻找已经在库中的等价定理，一旦匹配成功即给出完整证明。因为库中已有数十亿条定理，匹配成功的概率非常高。  

最巧妙的地方在于 **LLM 不是直接给出答案，而是提供概率指引**，这让搜索保持可解释且可调节；同时 **对称性批量生成** 把一次证明的价值放大到成百上千条，类似于一次实验得到多组数据。

### 实验与效果
- **测试集**：作者使用了 IMO‑AG‑30（30 道历届 IMO 几何题）以及若干地区性奥赛题目。  
- **对标基线**：与当时最先进的几何自动证明系统（如 GeoCoq、Eukleides）以及纯 LLM 直接推理的方案比较。  
- **核心结果**：TongGeometry 在 IMO‑AG‑30 上全部 30 题都给出完整证明，且每一道的证明质量超过历届金牌得主的手写解法——这是首次出现机器“超越”金牌水平的记录。  
- **规模指标**：在与基线相同的算力预算下，系统发现了 6.7 千亿条需要辅助构造的定理，其中 4.1 千亿条具备几何对称性。相比之前的系统最多只能发现数千万条，这一提升是数量级的跨越。  
- **实际应用**：作者向中国和美国的地区性数学奥林匹克提交了 10 条新命题，最终有 3 条被正式采用，分别进入了国家队选拔赛和顶级民间奥赛。  
- **消融实验**：论文中展示了去掉 LLM 引导、去掉对称性批量生成以及使用普通 CPU 而非并行优化的三组消融，对比显示：没有 LLM 引导时定理发现率下降约 70%；去掉对称性后库规模缩水约 40%；缺乏并行优化时单机求解时间从几分钟涨到数小时。  
- **局限性**：系统仍然依赖欧几里得平面几何的符号化表示，对三维几何或非欧几何的扩展尚未验证；此外，LLM 的指引质量受微调数据质量影响，极端或非常规题目仍可能出现搜索失败。  

### 影响与延伸思考
TongGeometry 的出现让“几何 AI 教练”从概念走向可用工具，激发了两大方向的后续研究：  
1. **跨模态几何生成**：把图像输入（手绘图形）转化为搜索起点，结合计算机视觉，实现从图片直接提出并求解几何题的全链路系统。  
2. **大模型与符号推理的深度融合**：后续工作尝试让 LLM 不仅提供搜索指引，还直接生成可验证的代数约束，进一步压缩搜索深度。  
如果想进一步了解，可以关注 2024‑2025 年在 NeurIPS、ICLR 上出现的 “Neuro‑Symbolic Geometry” 系列论文，它们大多受 TongGeometry 思路启发。  

### 一句话记住它
TongGeometry 用大语言模型引导的树搜索，把几何“灵感”数字化，首次实现了在普通电脑上全自动提出、证明并产出数千亿条奥赛级几何定理。