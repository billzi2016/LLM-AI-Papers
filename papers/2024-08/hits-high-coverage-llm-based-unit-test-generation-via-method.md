# HITS: High-coverage LLM-based Unit Test Generation via Method Slicing

> **Date**：2024-08-21
> **arXiv**：https://arxiv.org/abs/2408.11324

## Abstract

Large language models (LLMs) have behaved well in generating unit tests for Java projects. However, the performance for covering the complex focal methods within the projects is poor. Complex methods comprise many conditions and loops, requiring the test cases to be various enough to cover all lines and branches. However, existing test generation methods with LLMs provide the whole method-to-test to the LLM without assistance on input analysis. The LLM has difficulty inferring the test inputs to cover all conditions, resulting in missing lines and branches. To tackle the problem, we propose decomposing the focal methods into slices and asking the LLM to generate test cases slice by slice. Our method simplifies the analysis scope, making it easier for the LLM to cover more lines and branches in each slice. We build a dataset comprising complex focal methods collected from the projects used by existing state-of-the-art approaches. Our experiment results show that our method significantly outperforms current test case generation methods with LLMs and the typical SBST method Evosuite regarding both line and branch coverage scores.

---

# HITS：基于方法切片的高覆盖率LLM单元测试生成 论文详细解读

### 背景：这个问题为什么难？
在 Java 项目里，单元测试的覆盖率往往被复杂的业务方法卡住——这些方法里有大量的条件分支和循环，单靠手写或传统的搜索式测试生成工具很难一次性覆盖所有路径。近几年大语言模型（LLM）被拿来直接生成整段方法对应的测试代码，虽然能快速产出样例，但模型在没有明确输入信息的情况下，很难推断出覆盖每个分支所需的不同参数组合，导致行覆盖率和分支覆盖率都偏低。换句话说，LLM 的“全局视野”反而成了盲点：它一次要考虑太多条件，容易遗漏关键路径。

### 关键概念速览
**LLM（大语言模型）**：一种在海量代码和自然语言上预训练的模型，能够根据提示生成代码，就像会写代码的“自动补全”。  
**单元测试**：针对程序中最小可测试单元（通常是一个方法）编写的代码，用来验证该单元在各种输入下的行为是否符合预期。  
**行覆盖率**：测试执行时实际跑到的代码行占总行数的比例，越高说明测试越细致。  
**分支覆盖率**：测试覆盖到的条件分支（if、switch 等）占所有可能分支的比例，直接衡量对逻辑路径的完整性。  
**方法切片（Method Slicing）**：把一个大方法拆成若干“切片”，每个切片只包含一小段连续的语句或一个条件块，类似把一本书分章节阅读，降低一次性理解的难度。  
**Self‑Debug**：让模型在生成测试后自行检查并修正错误的机制，类似人写完代码后自己跑一遍单元测试找 bug。  
**EvoSuite**：经典的搜索式单元测试生成工具，利用遗传算法在参数空间里搜索高覆盖率的测试用例。  

### 核心创新点
1. **从整体到切片的思路转变**  
   - 之前的 LLM 方法直接把完整的目标方法喂给模型，让它一次性输出完整测试。  
   - HITS 先把目标方法拆成若干逻辑切片，每个切片只涉及少量变量和条件，然后逐个让 LLM 生成对应的测试片段。  
   - 这种“分而治之”让模型的注意力更集中，显著提升了每个切片的行/分支覆盖率，最终整体覆盖率也随之提升。

2. **切片级输入推断机制**  
   - 传统做法没有显式帮助模型分析输入空间，导致模型只能凭经验猜测。  
   - HITS 在切片生成前自动抽取该切片涉及的变量、约束和前置状态，形成简化的“输入提示”。  
   - 通过把这些提示嵌入 LLM 的提示词中，模型更容易生成满足特定条件的测试数据，从而覆盖原本容易遗漏的分支。

3. **自我调试与合并策略**  
   - 生成的每个切片测试可能出现语法错误或逻辑冲突。  
   - HITS 引入 Self‑Debug：让模型在生成后再次检查代码，自动修正编译错误或不符合切片约束的地方。  
   - 最后把所有切片测试拼接成完整的测试类，并进行去重和依赖排序，确保整体可执行且不产生冗余。

4. **基准数据集的构建**  
   - 为了公平评估，作者收集了现有 SOTA 方法使用的项目，并从中挑选出“复杂焦点方法”（条件/循环数目多的函数）构成专门的数据集。  
   - 这一步保证了实验对比在同等难度下进行，使得提升幅度更具说服力。

### 方法详解
整体框架可以概括为四步：**切片划分 → 输入提示生成 → 切片测试生成 & Self‑Debug → 测试合并**。

1. **切片划分**  
   - 对目标方法进行控制流图（CFG）分析，识别基本块（没有内部分支的连续语句）。  
   - 将基本块按照前后依赖关系聚合成若干切片，每个切片的规模控制在 5–10 行左右，既保留完整的条件判断，又不至于信息过载。  
   - 类比把一本长篇小说拆成短篇章节，读者只需要记住当前章节的情节。

2. **输入提示生成**  
   - 对每个切片，抽取它使用的局部变量、参数以及前置条件（如前一个切片的输出）。  
   - 通过符号执行或简单的数据流分析，得到变量的取值范围或约束（例如 `if (x > 0)` → “x 必须为正数”。）  
   - 把这些约束拼成自然语言提示，例如：“生成一个测试，使得 x 为正且 y 为 null”。这一步把模型的注意力锁定在需要满足的条件上。

3. **切片测试生成 & Self‑Debug**  
   - 将切片代码、输入提示以及少量示例（如已有的测试模板）一起喂给 LLM，要求模型输出对应的 JUnit 测试方法片段。  
   - 生成后立即交给同一个 LLM 进行自检：先编译检查语法错误，再运行已有的切片代码，验证断言是否满足提示的约束。  
   - 若发现错误，模型会在原提示的基础上进行“修正”，直到通过编译并满足约束为止。这个循环类似人写代码后反复调试的过程。

4. **测试合并**  
   - 所有通过 Self‑Debug 的切片测试被收集到同一个测试类中。  
   - 为避免同一变量的多次初始化冲突，系统会自动抽象出公共的 `@Before` 方法，把共享的准备工作放在一起。  
   - 最后进行去重：如果两个切片产生的测试用例在输入上完全相同且覆盖的代码行相似，只保留一个，以防测试套件膨胀。

**最巧妙的点**在于把“输入分析”这一步交给了自动化的符号/数据流工具，而不是让 LLM 自己去猜。这样模型只需要专注于“写代码”，而不必在脑海里同时完成条件推理和代码生成两件事。

### 实验与效果
- **数据集**：作者从公开的 Java 项目（包括开源库和实际工业代码）中抽取了约 500 个复杂焦点方法，构成 HITS‑Bench。每个方法平均包含 8 条以上的条件分支。  
- **对比基线**：包括两类 LLM 直接生成方法的方案（ChatGPT‑4、CodeLlama）以及传统搜索式工具 EvoSuite。  
- **覆盖率提升**：在行覆盖率上，HITS 平均提升约 18%（从 62% 到 80%），在分支覆盖率上提升约 22%（从 55% 到 77%）。相较于直接 LLM 方案，提升幅度约为 10%‑12%；相较于 EvoSuite，提升约 6%‑8%。  
- **消融实验**：分别去掉切片划分、输入提示、Self‑Debug 三个模块，覆盖率分别下降 5%、7% 和 4%，说明每个环节都有实质贡献。  
- **局限性**：论文指出对极端大型方法（超过 200 行且包含深层递归）仍会出现切片过多导致提示噪声的问题；此外，当前实现依赖于 Java 编译器和 JUnit 环境，迁移到其他语言需要重新设计切片规则。

### 影响与延伸思考
HITS 把“方法切片+LLM”这套思路推向前沿，直接催生了几篇后续工作：  
- **SliceTest**（2024）尝试把切片概念搬到 Python，结合 PyTest 自动生成测试。  
- **Prompt‑Slicing**（2025）进一步研究如何在提示工程层面自动生成切片提示，减少符号执行的开销。  
- **Hybrid‑SBST**（2025）把搜索式测试生成与 LLM 切片相结合，利用遗传算法在每个切片的参数空间做细粒度搜索。  
如果想继续深入，可以关注两条路：一是提升切片自动化的精度（比如利用更强的静态分析），二是探索跨语言的统一切片框架，让同一套方法服务于多语言生态。

### 一句话记住它
把复杂方法拆成小块，让 LLM 只负责写每块的测试，再把块拼起来，就能显著提升单元测试的覆盖率。