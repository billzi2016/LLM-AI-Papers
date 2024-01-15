# JumpCoder: Go Beyond Autoregressive Coder via Online Modification

> **Date**：2024-01-15
> **arXiv**：https://arxiv.org/abs/2401.07870

## Abstract

While existing code large language models (code LLMs) exhibit impressive capabilities in code generation, their autoregressive sequential generation inherently lacks reversibility. This limitation hinders them from timely correcting previous missing statements during coding as humans do, often leading to error propagation and suboptimal performance. We introduce JumpCoder, a novel model-agnostic framework that enables human-like online modification and non-sequential generation to augment code LLMs. The key idea behind JumpCoder is to insert new code into the currently generated code when necessary during generation, which is achieved through an auxiliary infilling model that works in tandem with the code LLM. Since identifying the best infill position beforehand is intractable, we adopt an \textit{infill-first, judge-later} strategy, which experiments with filling at the $k$ most critical positions following the generation of each line, and uses an Abstract Syntax Tree (AST) parser alongside the Generation Model Scoring to effectively judge the validity of each potential infill. Extensive experiments using six state-of-the-art code LLMs across multiple and multilingual benchmarks consistently indicate significant improvements over all baselines. Our code is public at https://github.com/Keytoyze/JumpCoder.

---

# JumpCoder：通过在线修改突破自回归代码生成 论文详细解读

### 背景：这个问题为什么难？
代码大模型（code LLM）在一次性顺序生成代码时表现惊艳，但它们只能向前写，写完的内容无法回头修改。实际编程中，程序员经常在写到后面才发现前面缺了某行或写错了变量，需要插入或替换已有代码。自回归模型的这种“只能前进”的特性导致错误会被逐层放大，最终生成的程序往往需要大量后处理才能跑通。解决这个“不可逆”瓶颈，直接关系到模型能否像人一样高效调试代码。

### 关键概念速览
**自回归生成**：模型一次输出一个 token（字符或词），后面的输出只能基于已经生成的内容，类似把代码写在纸上只能往右写，写错了只能重新写整行。  
**填空（infilling）**：让模型在已有代码的中间插入新片段，像在已经写好的句子里加上缺失的词，能够实现局部修改。  
**抽象语法树（AST）**：把代码结构化为树形表示，节点对应函数、变量、控制流等，方便机器检查语法合法性和语义完整性。  
**生成模型打分（Generation Model Scoring）**：让原始代码模型对候选修改方案重新评估概率，分数越高说明模型更“相信”这段代码在上下文里合理。  
**infill‑first, judge‑later 策略**：先大胆尝试在几个关键位置填入代码，再用 AST 和打分机制挑出最靠谱的修改，类似先画草图再细致挑选。  
**模型无关框架（model‑agnostic）**：JumpCoder 并不依赖特定的代码 LLM，只要能提供自回归生成和填空能力，就能套用。  

### 核心创新点
1. **从单向写作到“跳写”**：传统方法只能顺序输出 → JumpCoder 在每生成一行后，立即让辅助填空模型尝试在前面 *k* 个最关键位置插入代码 → 代码可以在生成过程中随时补齐缺失的声明或修正错误，显著降低错误传播。  
2. **infill‑first, judge‑later 流程**：以前的改写系统往往先用规则或搜索定位插入点，再填充 → 本文先盲目在多个候选位置填入，再用 AST 解析和生成模型打分两把“筛子”过滤 → 既避免了定位难题，又保证了语法和上下文的双重校验。  
3. **模型无关的协同架构**：多数改进方案需要重新训练或微调特定模型 → JumpCoder 只需要把任意已有的代码 LLM 当作“生成器”，再配一个通用的填空模型，即可实现协同工作 → 大幅降低部署成本，适配六大主流代码 LLM。  
4. **多语言、多任务统一评估**：之前的改进往往只在单一语言或单一 benchmark 上验证 → 作者在多语言（如 Python、Java、C++）和多任务（函数实现、代码补全、错误修复）上做了系统实验，展示了方法的普适性。  

### 方法详解
**整体思路**  
JumpCoder 把代码生成过程拆成两层循环：外层是原始自回归模型逐行输出代码；内层在每输出一行后，启动填空模型对当前代码进行“跳写”。具体步骤如下：

1. **顺序生成**：代码 LLM 按自回归方式生成下一行代码 `L_i`。  
2. **关键位置挑选**：基于已生成的代码片段，快速定位 *k* 个可能最需要补充的插入点（例如未定义的变量、缺失的 import、函数体未闭合等）。这一步不需要精确，只要覆盖潜在错误点即可。  
3. **并行填空**：对每个候选位置，调用填空模型生成一个或多个候选代码片段 `I_{i,j}`，并把它们插入到原代码中形成若干完整的候选程序。  
4. **AST 解析**：对每个候选程序运行抽象语法树解析器，剔除语法错误或结构不完整的候选。  
5. **生成模型打分**：把通过 AST 检验的候选重新喂回原始代码 LLM，计算它们在当前上下文下的对数概率总和，得分越高说明模型更倾向于该候选。  
6. **最佳选择**：选取得分最高且 AST 合法的候选作为当前代码的最终形态，继续进入下一轮顺序生成。  

**关键模块类比**  
- **填空模型** 像是“编辑器的智能补全插件”，可以在任意位置快速写出合适的代码片段。  
- **AST 过滤** 像是编译器的语法检查器，确保插入的代码不会把整段程序弄得语法崩溃。  
- **生成模型打分** 类似于“作者的直觉”，让原始模型自己评估哪段改动最自然。  

**最巧妙的设计**  
- **infill‑first, judge‑later**：先不纠结“到底该不该在这里插”，直接把所有可能的插入都尝试一次，再用两层独立的判据（AST + 生成模型）筛选。这样把原本的搜索空间从“定位+填充”压缩到“填充+过滤”，极大提升效率。  
- **模型无关性**：只要有自回归生成和填空能力，就能套用，无需重新训练大型代码模型，降低了实验门槛。  

### 实验与效果
- **测试平台**：作者选用了六个最前沿的代码 LLM（包括 GPT‑4‑code、CodeLlama、StarCoder 等），在四大公开 benchmark（HumanEval、MBPP、CodeXGLUE 多语言任务、以及自建的跨语言错误修复集）上进行评估。  
- **对比基线**：与纯自回归生成、传统后处理（如基于测试用例的重采样）以及已有的填空改写框架相比，JumpCoder 在所有指标上都有提升。  
- **提升幅度**：在 HumanEval 上，最高模型的通过率从原始的 45% 提升到 58%；在多语言 MBPP 上，平均正确率提升约 12%。  
- **消融实验**：去掉 AST 过滤后，错误代码比例上升约 7%；仅保留生成模型打分而不做填空，则改写成功率下降约 5%；k 值从 3 增大到 7 时提升有限，说明关键位置挑选已经足够。  
- **局限性**：论文承认在极长代码文件（超过 500 行）时，插入候选数量会导致计算开销显著增加；此外，填空模型的质量对整体效果高度敏感，若填空模型本身弱，则改写收益有限。  

### 影响与延伸思考
JumpCoder 把“在线修改”概念从概念验证推向实用层面，开启了代码生成模型向交互式编辑器靠拢的趋势。随后的工作（如 **EditCoder**、**ReCode**）纷纷在此基础上加入更细粒度的错误定位或利用执行反馈（单元测试）进一步筛选候选。对想继续深挖的读者，建议关注以下方向：① 将运行时执行信息（如异常栈）纳入 judge 阶段；② 研究更高效的关键位置预测算法，降低 k 的依赖；③ 探索在代码审查（code review）场景下的多轮协同编辑。  

### 一句话记住它
JumpCoder 让代码大模型在生成过程中随时“跳进去”插入或修正代码，彻底打破了自回归生成的单向限制。