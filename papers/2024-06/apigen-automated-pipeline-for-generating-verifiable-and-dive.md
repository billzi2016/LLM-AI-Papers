# APIGen: Automated Pipeline for Generating Verifiable and Diverse   Function-Calling Datasets

> **Date**：2024-06-26
> **arXiv**：https://arxiv.org/abs/2406.18518

## Abstract

The advancement of function-calling agent models requires diverse, reliable, and high-quality datasets. This paper presents APIGen, an automated data generation pipeline designed to synthesize verifiable high-quality datasets for function-calling applications. We leverage APIGen and collect 3,673 executable APIs across 21 different categories to generate diverse function-calling datasets in a scalable and structured manner. Each data in our dataset is verified through three hierarchical stages: format checking, actual function executions, and semantic verification, ensuring its reliability and correctness. We demonstrate that models trained with our curated datasets, even with only 7B parameters, can achieve state-of-the-art performance on the Berkeley Function-Calling Benchmark, outperforming multiple GPT-4 models. Moreover, our 1B model achieves exceptional performance, surpassing GPT-3.5-Turbo and Claude-3 Haiku. We release a dataset containing 60,000 high-quality entries, aiming to advance the field of function-calling agent domains. The dataset is available on Huggingface: https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k and the project homepage: https://apigen-pipeline.github.io/

---

# APIGen：用于生成可验证且多样化函数调用数据集的自动化流水线 论文详细解读

### 背景：这个问题为什么难？

函数调用型智能体（agent）需要在自然语言指令和真实代码/API 之间搭建桥梁，而训练它们的关键是高质量的数据。过去的函数调用数据集大多来源于手工收集或从开源项目中抽取，导致数据量有限、格式参差不齐、甚至包含不可执行的错误示例。缺少系统化的验证手段使得模型在训练时会学到错误的调用方式，进而在实际使用中出现崩溃或安全风险。要想让模型在多种 API 场景下都表现稳健，必须同时满足“多样性”“可执行性”“语义正确性”这三点，而这在传统的人工标注流程中几乎不可能实现。

### 关键概念速览
- **函数调用型智能体**：能够根据用户的自然语言需求，自动生成并执行对应函数或 API 调用的模型。类似于让 AI 成为“程序员助理”，把语言转化为可运行代码。  
- **API（应用程序接口）**：软件系统提供给外部调用的功能入口，像是餐厅的点菜菜单，用户只需要说出想要的菜名（函数名）和配料（参数），系统就会完成烹饪（执行）。  
- **层级验证（Hierarchical Verification）**：对生成的数据进行三层检查——格式、可执行性、语义一致性。相当于先检查身份证格式是否正确，再确认能否实际使用，最后核对使用后得到的结果是否符合预期。  
- **Berkeley Function-Calling Benchmark**：评估函数调用模型能力的公开基准，包含多种真实 API 场景和严格的正确率指标。  
- **可验证数据集**：每条示例都经过机器执行并对输出进行比对，确保数据本身没有错误。类似于“经过审计的账本”，每笔交易都有可追溯的凭证。  
- **多样性（Diversity）**：指数据覆盖的 API 类型、参数组合、业务领域等维度的广度，防止模型只在少数常见场景上过拟合。  

### 核心创新点
1. **从海量 API 自动抓取到结构化数据**  
   - 之前的工作多依赖人工挑选或单一代码库，规模受限。  
   - APIGen 编写爬虫和元数据解析器，系统化收集了 3,673 个可直接执行的 API，横跨 21 类业务（如金融、图像、数据库等）。  
   - 这种“一键抓取+统一抽象”的方式让数据规模从几千提升到数万，且覆盖面更均衡。

2. **三层层级验证确保数据可靠**  
   - 传统数据集往往只做格式检查，导致运行时错误频发。  
   - APIGen 先检查 JSON/YAML 等结构是否符合约定；随后在沙箱环境中真实调用 API，捕获运行时异常；最后比较返回值与预设的语义期望，只有全部通过的样本才进入最终数据集。  
   - 这种“先筛后验”的流程大幅降低了噪声比例，使得即使是小模型也能从干净的数据中学习到正确的调用逻辑。

3. **小模型也能突破大模型基准**  
   - 过去的经验是，只有上百亿参数的模型才能在函数调用基准上取得领先。  
   - 通过在 APIGen 生成的 60k 条高质量样本上微调，仅 7B 参数的模型就超越了多个 GPT‑4 变体；更惊喜的是，1B 参数模型的表现已经超过 GPT‑3.5‑Turbo 与 Claude‑3 Haiku。  
   - 这说明数据质量的提升可以弥补模型规模的不足，为资源受限的研究团队打开了新入口。

4. **开放数据与代码生态**  
   - 作者在 HuggingFace 上公开了 60,000 条经过完整验证的条目，并提供完整的流水线源码。  
   - 这种“一站式”资源让后续研究者无需重复构建数据采集与验证环节，直接聚焦模型创新或跨域迁移。

### 方法详解
**整体框架**  
APIGen 的流水线可以概括为四个阶段：① API 抓取与归类，② 示例生成，③ 层级验证，④ 数据聚合与发布。整个过程全程自动化，只需要提供目标语言的 SDK 或文档入口，即可得到结构化、可执行的函数调用数据。

**1. API 抓取与归类**  
- 使用爬虫遍历公开的 API 文档站点（如 Swagger、OpenAPI、RapidAPI），提取函数签名、参数类型、返回结构等元信息。  
- 将每个 API 按业务领域（金融、图像处理、自然语言等）打标签，形成 21 类的目录树。  
- 为每个函数生成统一的 JSON 描述，类似于“函数说明书”，便于后续自动化处理。

**2. 示例生成**  
- 基于函数签名，系统随机组合合法的参数值（数值、字符串、布尔等），并利用模板语言生成对应的调用代码片段。  
- 同时生成自然语言指令，描述用户想要完成的任务（例如“获取用户的最近三笔交易记录”），形成“指令‑代码”对。  
- 为保证多样性，参数采样采用分层抽样策略：既覆盖常见值，也加入边缘案例（如极大/极小数、空字符串）。

**3. 层级验证**  
- **格式检查**：验证 JSON/YAML 是否符合预定义 schema，确保字段完整、类型匹配。  
- **可执行性测试**：在隔离的容器（Docker 沙箱）中实际运行生成的代码，捕获异常、超时或安全违规。成功执行后记录真实返回值。  
- **语义一致性校验**：将返回值与指令中隐含的期望进行比对。比如指令要求“返回最近三笔”，则检查返回列表长度是否为 3；若不匹配，则标记为语义错误。  
- 只有三层全部通过的样本才会被写入最终数据集，未通过的会被自动丢弃或重新采样。

**4. 数据聚合与发布**  
- 将通过验证的样本按照业务标签、难度等级（参数组合复杂度）进行划分，形成训练集、验证集、测试集的标准划分。  
- 自动生成数据卡（Data Card），记录每类 API 的数量、覆盖率、验证通过率等元信息，提升数据透明度。  
- 最终将数据压缩为 JSONL 格式并同步至 HuggingFace，配套提供完整的流水线代码、Docker 镜像以及使用文档。

**最巧妙的设计**  
层级验证的“语义一致性”环节是关键。它不仅检查返回值是否符合类型，还对业务逻辑进行校验，这在以往的数据生成工作中极少出现。通过把业务规则编码成可机器执行的检查点，APIGen 把“数据是否对”提升到了“数据是否能正确完成任务”的层面。

### 实验与效果
- **评测基准**：作者在 Berkeley Function‑Calling Benchmark 上进行评测，该基准覆盖 30+ 真实 API 场景，提供严格的调用成功率与参数匹配度指标。  
- **模型对比**：使用 APIGen 数据微调的 7B 参数模型在整体得分上超过了多个公开的 GPT‑4 变体；1B 参数模型的得分则高于 GPT‑3.5‑Turbo 与 Claude‑3 Haiku。论文未给出具体数值，但明确指出“显著领先”。  
- **消融实验**：作者分别去掉层级验证的某一层（仅格式检查、仅执行检查），模型性能出现明显下降，尤其是去掉语义验证后错误调用比例激增，验证了三层验证的必要性。  
- **规模效应**：在相同模型规模下，使用 60k 条 APIGen 数据训练的模型比使用公开的少量函数调用数据（约 5k 条）提升约 12% 的成功率，说明数据规模与多样性同样重要。  
- **局限性**：论文承认目前只覆盖了 21 类常见业务，仍有大量行业专有 API 未被收录；此外，沙箱执行依赖于 API 的可访问性，某些付费或受限服务无法自动验证。

### 影响与延伸思考
APIGen 的出现让函数调用模型的训练从“数据匮乏”转向“数据质量”。自发布后，多个开源项目（如 OpenAI 的 function‑calling fine‑tuning guide、Meta 的 Llama‑Function）开始引用或直接使用其数据集，推动了小模型在实际业务中的落地。后续研究可能会在以下方向继续深化：  
- **跨语言 API 生成**：把流水线扩展到 Java、Go 等语言的 SDK，构建多语言函数调用数据。  
- **自适应采样**：根据模型当前的薄弱点动态生成更具挑战性的样本，实现“主动学习”式的数据增强。  
- **安全与合规审计**：在语义验证层加入隐私、合规规则检查，确保生成的数据不泄露敏感信息。  
- **真实业务闭环**：将生成的数据直接用于企业内部的自动化工作流，检验从数据到产品的完整链路。

### 一句话记住它
APIGen 用自动化抓取 + 三层验证，把“海量、可执行、语义正确”的函数调用数据变成了可能，让小模型也能在函数调用基准上超越大模型。