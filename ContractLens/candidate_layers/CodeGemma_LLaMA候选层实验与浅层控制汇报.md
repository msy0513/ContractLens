# CodeGemma 与 LLaMA：SmartIntent 候选层定位及浅层控制实验汇报

## 1. 汇报范围

本汇报总结已经完成、且采用同一 Stage 4 实验协议的两组实验：

- CodeGemma-7B 的 SmartIntent Solidity 源码候选层定位；
- LLaMA-3.1-8B 的 SmartIntent Solidity 源码候选层定位；
- CodeGemma-7B 的四类浅层线索控制实验。

当前正在运行的 CodeGemma `EOS-query + overlap-corrected` 新版实验尚未完成，因此不混入本汇报的正式模型比较。

## 2. 实验目的

候选层实验试图回答：冻结代码模型参数后，哪些层的内部表示包含可由轻量线性探针读取的 unsafe-intent 相关信号？这些信号是否同时表现为：

1. unsafe-intent 与无 unsafe-intent 合约在表示空间中发生类别中心分离；
2. 模型处理两类合约时出现不同的注意力分布；
3. 该层表示能够被 Logistic Regression 线性解码。

浅层控制实验进一步回答：上述分类能力主要来自 unsafe-intent 语义，还是来自注释、SPDX、pragma、标识符、代码长度等表面线索？

## 3. 数据集

### 3.1 来源

数据来自 Hugging Face 数据集 `web3se/smart-contract-intent-vul-dataset` 中的 `web3_all.sql.gz`。解析后得到 32,141 份 BSC Solidity 合约源码。

数据包含十类 annotated unsafe intent：

- Blacklist
- Mint
- Honeypot
- Fee
- DisableTrading
- MaxSell
- MaxTX
- Reflection
- Reward
- Rebase

原始解析数据中，21,166 份合约至少带有一个上述标签，10,975 份合约没有上述标签。

### 3.2 Binary 候选层任务

从解析数据构建2,000份唯一合约的平衡二分类任务：

| 标签 | 定义 | 数量 |
|---|---|---:|
| 1 | 至少有一个 annotated unsafe-intent 标签 | 1,000 |
| 0 | 没有任何上述 annotated unsafe-intent 标签 | 1,000 |
| 合计 | 唯一 Solidity 合约 | 2,000 |

这里的标签不能直接写成“经过安全审计的 malicious/benign”。更严谨的任务名称是 `unsafe-intent present / no annotated unsafe intent`。

### 3.3 三个固定 trial

使用 seed 0、1、2 建立三个独立的 creator-group-aware trial。每个 trial 都使用全部2,000份合约，但划分不同；同一 creator group 不跨训练、验证和测试集。

| Split | 总数 | Label 0 | Label 1 |
|---|---:|---:|---:|
| Train | 1,280 | 640 | 640 |
| Validation | 319 | 160 | 159 |
| Test | 401 | 200 | 201 |

不同 seed 的训练 creator group 数分别为1,225、1,227、1,227。测试标签只用于最终评估，不参与层选择或候选集合构造。

## 4. 模型和冻结设置

| 项目 | CodeGemma-7B | LLaMA-3.1-8B |
|---|---:|---:|
| 参数总数 | 8,537,680,896 | 7,504,924,672 |
| 可训练参数 | 0 | 0 |
| 模式 | `eval()` | `eval()` |
| 推理精度 | BF16 | BF16 |
| 表示层 | 0–28，共29层 | 0–32，共33层 |
| Hidden size | 3,072 | 4,096 |

模型通过以下方式冻结：

```python
model = AutoModel.from_pretrained(model_path)
model.requires_grad_(False)
model.eval()

with torch.inference_mode():
    outputs = model(...)
```

因此 backbone 不更新，只有后续 Logistic Regression probe 被训练。

## 5. 候选层定位流程

### 5.1 逐层提取代码表示

完整源码采用2,048-token滑窗，窗口重叠512 tokens。当前 Stage 4 实现先在每个窗口内对 non-padding tokens 取均值，再对该合约的所有窗口等权平均。

窗口表示为：

\[
h_{l,w}(x)=\frac{\sum_t m_{w,t}H_{l,w,t}(x)}{\sum_t m_{w,t}}
\]

合约表示为：

\[
h_l(x)=\frac{1}{W_x}\sum_{w=1}^{W_x}h_{l,w}(x)
\]

作用：将每层不同长度的 token hidden states 压缩为一个固定维度向量，用于类别中心计算和逐层 probe。

技术说明：该 Stage 4 实现是“窗口内 mean，再对窗口 mean”，重叠 token 可能被重复计权。正在运行的 corrected 实验会改为每个全局代码 token 只统计一次，但 corrected 结果不属于本报告。

### 5.2 类别中心余弦距离

对每个 seed，只使用该 seed 的1,280份训练合约。第 \(l\) 层两类中心为：

\[
\mu_{0,l}=\frac{1}{|X_0|}\sum_{x\in X_0}h_l(x),\qquad
\mu_{1,l}=\frac{1}{|X_1|}\sum_{x\in X_1}h_l(x)
\]

余弦距离为：

\[
D_{\cos}(l)=1-\frac{\mu_{0,l}^{\top}\mu_{1,l}}
{\|\mu_{0,l}\|\|\mu_{1,l}\|}
\]

按照 DNA 的 \(\alpha=1.0\) 规则：

\[
L_{\mathrm{sep}}=\{l\mid D_{\cos}(l)>\mu_D+\sigma_D\}
\]

作用：寻找两类合约平均表示方向显著分离的层。它提供表示空间的几何证据，但没有考虑类内方差，因此不能单独等同于分类性能。

### 5.3 注意力分布差异

三个 seed 的训练集去重并集包含1,876份合约，attention只抽取一次；计算某个 seed 时，只索引该 seed自己的1,280份训练合约。

当前 Stage 4 使用每个窗口的最后有效 token作为 query，对所有attention heads取均值：

\[
A_l(x,t)=\frac{1}{H}\sum_{h=1}^{H}A_{l,h}[q_x,t]
\]

不同长度源码的位置映射到32个全局位置 bins。窗口attention在全局源码位置上累加，然后逐层做L1归一化。分别计算两类训练样本的平均attention：

\[
\bar A_{0,l}=\frac{1}{|X_0|}\sum_{x\in X_0}\tilde A_l(x),\qquad
\bar A_{1,l}=\frac{1}{|X_1|}\sum_{x\in X_1}\tilde A_l(x)
\]

注意力分布距离为：

\[
D_{\mathrm{attn}}(l)=\|\bar A_{0,l}-\bar A_{1,l}\|_2
\]

严格内部局部峰值构成：

\[
L_{\mathrm{attn}}=\{l\mid D_{\mathrm{attn}}(l)>D_{\mathrm{attn}}(l-1)
\land D_{\mathrm{attn}}(l)>D_{\mathrm{attn}}(l+1)\}
\]

Transformer attention block \(b\) 对齐 representation layer \(b+1\)。

作用：寻找模型处理两类源码时信息路由差异相对突出的层。Attention差异不是因果证明，也可能受到代码位置和长度影响。

### 5.4 逐层 Logistic Regression

每一层独立训练一个轻量线性probe：

\[
z_l=w_l^{\top}h_l+b_l
\]

具体配置：

- `StandardScaler`标准化；
- `LogisticRegression(max_iter=3000, class_weight="balanced")`；
- 在1,280份训练合约上拟合；
- 在319份验证合约上选择层；
- 在401份测试合约上做最终评估；
- 报告Accuracy、AUROC、AUPRC和F1。

DNA Accuracy版本定义为：

\[
L_{\mathrm{prob}}^{\mathrm{Acc}}=
\{l\mid Acc_{val}(l)\geq0.95\max_j Acc_{val}(j)\}
\]

由于任务主要关注排序式分类能力，还补充了AUROC版本：

\[
L_{\mathrm{prob}}^{\mathrm{AUROC}}=
\{l\mid AUROC_{val}(l)\geq\max_j AUROC_{val}(j)-0.01\}
\]

作用：验证某层表示是否包含可由简单线性决策边界读取的类别信号。验证集负责选层，测试集不参与候选集合构造。

### 5.5 三集合交集

最终候选关键层为：

\[
L_{\mathrm{critical}}=
L_{\mathrm{sep}}\cap L_{\mathrm{attn}}\cap L_{\mathrm{prob}}
\]

交集要求某层同时具有：

1. 类别中心几何分离；
2. 类别相关注意力变化；
3. 线性可解码能力。

作用：减少依赖单一指标选择层的偶然性。交集只能定位候选层，仍不能自动排除注释、名称、长度等共同混杂因素。

## 6. CodeGemma 候选层结果

### 6.1 三个 seed

| Seed | 最佳Val AUROC层 | Val AUROC | Test AUROC | Test AUPRC | Test F1 | Test Accuracy | Accuracy交集 | AUROC交集 |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0 | 12 | 0.9526 | 0.9583 | 0.9390 | 0.9015 | 0.9002 | `{24}` | 空集 |
| 1 | 24 | 0.9517 | 0.9636 | 0.9581 | 0.9270 | 0.9277 | `{24}` | `{24}` |
| 2 | 12 | 0.9575 | 0.9378 | 0.8900 | 0.9146 | 0.9152 | `{24}` | 空集 |

测试指标均来自“由该seed验证集AUROC选出的层”，不是测试曲线事后峰值层。

跨seed均值与总体标准差：

| 指标 | CodeGemma |
|---|---:|
| Validation AUROC | 0.9539 +/- 0.0026 |
| Test AUROC | 0.9533 +/- 0.0111 |
| Test AUPRC | 0.9290 +/- 0.0287 |
| Test F1 | 0.9143 +/- 0.0104 |
| Test Accuracy | 0.9144 +/- 0.0112 |

### 6.2 固定 Seed 1 的集合

\[
L_{\mathrm{sep}}=\{24,25,26,28\}
\]

\[
L_{\mathrm{attn}}=\{4,11,14,18,20,24\}
\]

\[
L_{\mathrm{prob}}^{\mathrm{AUROC}}=
\{12,14,15,16,17,19,20,21,22,23,24,25,26,28\}
\]

因此：

\[
L_{\mathrm{critical}}^{\mathrm{AUROC}}=\{24\}
\]

Accuracy版本三个seed均得到Layer 24。由于Accuracy阈值较宽，几乎所有层都进入 \(L_{\mathrm{prob}}^{\mathrm{Acc}}\)，Layer 24主要由更严格的 \(L_{\mathrm{sep}}\) 与 \(L_{\mathrm{attn}}\) 重合确定。AUROC版本只有Seed 1保留Layer 24，说明严格AUROC交集的跨seed稳定性有限。

## 7. LLaMA 候选层结果

### 7.1 三个 seed

| Seed | 最佳Val AUROC层 | Val AUROC | Test AUROC | Test AUPRC | Test F1 | Test Accuracy | Accuracy交集 | AUROC交集 |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0 | 13 | 0.9423 | 0.9454 | 0.9300 | 0.8933 | 0.8928 | 空集 | 空集 |
| 1 | 30 | 0.9401 | 0.9521 | 0.9543 | 0.8732 | 0.8653 | 空集 | 空集 |
| 2 | 0 | 0.9383 | 0.9406 | 0.9357 | 0.8943 | 0.8928 | 空集 | 空集 |

跨seed均值与总体标准差：

| 指标 | LLaMA |
|---|---:|
| Validation AUROC | 0.9402 +/- 0.0016 |
| Test AUROC | 0.9460 +/- 0.0047 |
| Test AUPRC | 0.9400 +/- 0.0104 |
| Test F1 | 0.8870 +/- 0.0097 |
| Test Accuracy | 0.8836 +/- 0.0129 |

### 7.2 固定 Seed 1 的集合

\[
L_{\mathrm{sep}}=\{2,23,24,32\}
\]

\[
L_{\mathrm{attn}}=\{5,7,9,13,16,19,22,26,29\}
\]

\[
L_{\mathrm{prob}}^{\mathrm{AUROC}}=\{11,16,17,18,26,30,31,32\}
\]

三个集合没有共同元素：

\[
L_{\mathrm{critical}}^{\mathrm{AUROC}}=\varnothing
\]

LLaMA并不是没有分类能力。三个seed的Test AUROC均约0.94以上，说明hidden states中存在可线性读取的任务信号；但几何分离、attention峰值和probe高分分布在不同层，没有形成三指标共同支持的单一关键层。

## 8. 两个模型的结果比较

| 指标 | CodeGemma | LLaMA |
|---|---:|---:|
| 平均Test AUROC | 0.9533 | 0.9460 |
| 平均Test AUPRC | 0.9290 | 0.9400 |
| 平均Test F1 | 0.9143 | 0.8870 |
| 平均Test Accuracy | 0.9144 | 0.8836 |
| Accuracy三集合交集 | 三个seed均为Layer 24 | 三个seed均为空 |
| AUROC三集合交集 | Seed 1为Layer 24 | 三个seed均为空 |

结果表明：

1. 两个冻结模型都包含可由线性probe读取的unsafe-intent相关任务信号。
2. CodeGemma的平均AUROC、F1和Accuracy略高，且三类信号在Layer 24发生汇聚。
3. LLaMA的AUPRC略高，但几何、attention和probe信号分布更分散，没有严格三集合交集。
4. 最佳层编号不是性能高低。CodeGemma的Layer 12/24和LLaMA的Layer 0/13/30表示不同seed选择出的信号位置，不表示层号越小越好。
5. CodeGemma有29个表示层，LLaMA有33个表示层。跨模型解释位置时应使用相对深度，而不能只比较绝对层号。

该结果支持“存在可读的任务相关信号”，但不能单凭三集合交集宣称已经定位纯粹的latent malicious-contract semantic knowledge。

## 9. CodeGemma 浅层控制实验

### 9.1 为什么需要控制实验

类别中心距离、attention差异和linear probe可能同时利用相同的数据偏差。例如，如果unsafe-intent样本更常包含特定注释或函数名，那么三个指标都可能很高，交集仍然无法区分词面捷径与安全语义。

控制实验用于回答：原始高分是否主要由注释、头部信息、标识符、长度或标签泄漏解释？

所有控制实验沿用同一2,000份合约、同一creator-group-aware trial、冻结CodeGemma和验证集选层规则。变换后的源码按地址与正式任务对齐，测试标签不参与选层。

### 9.2 去注释、SPDX和pragma

操作：删除Solidity源码中的注释、SPDX license identifier和pragma版本声明，然后重新抽取29层冻结表示，并在所有层重新训练probe。

目的：检查模型是否主要依赖源码头部、自然语言注释、许可证和编译器版本信息。

### 9.3 标识符匿名化

操作：将合约名、函数名、变量名和参数名替换为匿名占位符，再重新抽取表示和逐层训练probe。

目的：检查模型是否通过诸如`blacklist`、`mint`、`tax`、`owner`等命名模式完成分类，而不是理解代码行为。

### 9.4 长度匹配

操作：在每个固定split内部匹配label 0和label 1的token长度分布，并搜索全部层。匹配后每个seed的测试样本数分别为184、158和178。

目的：排除“某类合约通常更长或更短”这一简单分类捷径。

### 9.5 随机标签

操作：保持Layer 24 frozen features不变，只打乱训练标签，运行60次probe。

目的：检查probe流程是否存在标签泄漏或即使随机标签也能获得异常高分的问题。预期AUROC应接近0.5，AUPRC接近正类比例。

### 9.6 控制实验结果

| 实验 | 三个seed选择层 | Test AUROC | Test AUPRC | Test F1 | Test Accuracy |
|---|---|---:|---:|---:|---:|
| 原始源码 | 12 / 24 / 12 | 0.9533 +/- 0.0111 | 0.9290 +/- 0.0287 | 0.9143 +/- 0.0104 | 0.9144 +/- 0.0112 |
| 去注释、SPDX、pragma | 5 / 6 / 16 | 0.5054 +/- 0.0125 | 0.5216 +/- 0.0185 | 0.4995 +/- 0.0227 | 0.5071 +/- 0.0150 |
| 标识符匿名化 | 5 / 5 / 5 | 0.5028 +/- 0.0263 | 0.5057 +/- 0.0230 | 0.5004 +/- 0.0173 | 0.5012 +/- 0.0041 |
| 长度匹配、搜索全部层 | 12 / 23 / 19 | 0.8915 +/- 0.0244 | 0.8652 +/- 0.0384 | 0.8456 +/- 0.0193 | 0.8455 +/- 0.0254 |
| Layer 24随机训练标签，60次 | 固定Layer 24 | 0.5235 +/- 0.0780 | 0.4926 +/- 0.0484 | 0.5095 +/- 0.0511 | 0.5134 +/- 0.0549 |

### 9.7 控制结果解析

1. 长度匹配后Test AUROC仍为0.8915，说明合约长度不是原始0.9533 AUROC的唯一解释，但长度确实贡献了一部分信号。
2. 随机标签结果回到约0.5，说明probe训练与测试流程没有明显的简单标签泄漏。
3. 删除注释/头部后AUROC降到0.5054，标识符匿名化后降到0.5028，均接近随机水平。
4. 因此，当前原始源码任务的强分类能力高度依赖注释、头部信息和/或标识符词面线索。
5. Layer 24仍是原始数据分布下稳定的三信号候选层，但控制实验不支持将其直接解释为独立于表面线索的纯unsafe-intent语义层。

去注释和匿名化控制较为激进，也会改变tokenization和源码长度。下一步应将注释、SPDX和pragma分别移除，并采用AST/语法感知的标识符重命名，再进行更细致的因果判断。

## 10. 当前可以支持的结论

建议在汇报中使用以下表述：

> 在SmartIntent Solidity源码二分类任务上，冻结的CodeGemma-7B和LLaMA-3.1-8B hidden states均包含可由Logistic Regression读取的unsafe-intent相关任务信号。CodeGemma在原始源码分布下表现出更集中的三指标汇聚，Layer 24是其稳定候选层；LLaMA的几何分离、注意力变化和线性可解码信号分布在不同层，严格三集合交集为空。然而，CodeGemma浅层控制实验显示原始高分强烈依赖注释、头部和标识符信息，因此目前结果证明的是原始数据分布下的可分信号，而不是已经排除表面捷径的纯latent malicious-contract semantic knowledge。

## 11. 结果文件

- CodeGemma正式候选层：`runs/stage4_smartintent_full_trials/`
- LLaMA正式候选层：`runs/stage4_smartintent_llama_full_trials/`
- CodeGemma控制实验：`runs/stage4_smartintent_controls/`
- 数据统计：`runs/stage3_smartintent/dataset_summary.json`
- 固定trial manifests：`runs/stage4_smartintent_full_trials/manifests/`

