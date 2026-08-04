# UA-A-LMC-VAE：当前实现的 Loss 解析

> 本节以当前仓库代码为准，重点解释默认 `model=covae` 时实际参与反向传播的 loss，并逐项核对《UA-A-LMC-VAE 改进方案简要报告.pdf》。文末保留了原 README 全文，供进一步查阅。

## 1. 一句话总览

当前 `CoVAE` 的生成器/自编码器目标由五部分组成：相邻噪声层级之间的输出一致性（重建）loss、VAE 的 KL loss、可选的直接去噪 loss、可选的 GAN 生成器 loss，以及本项目新增的 **不确定性感知、非对称潜均值一致性 loss（UA-A-LMC）**：

$$
\mathcal L_{\mathrm{total}}
=\frac{1}{B}\sum_{i=1}^{B}
\left(\mathcal L_{\mathrm{rec},i}
+\mathcal L_{\mathrm{den},i}
+\mathcal L_{\mathrm{KL},i}\right)
+\mathcal L_{\mathrm{GAN}}
+\lambda_{\mathrm{UA}}(s)\mathcal L_{\mathrm{UA-A-LMC}}.
$$

默认配置中 `use_gan: False`，所以 $\mathcal L_{\mathrm{GAN}}=0$；`denoiser_loss_mode: l2`、`lambda_latent_consistency: 0.1`，所以直接去噪项和 UA-A-LMC 项均启用。总目标的实际组装位置是 `models/covae.py:272-273`。

## 2. 训练样本、时间层级与两条分支

训练步记为 $s$，batch size 记为 $B$。代码先通过 `step_schedule` 得到当前时间网格 $\{\tau_j\}$，然后对每个样本随机选择一对相邻层级：

$$
j_i\sim\operatorname{Uniform}\{0,\ldots,|\tau|-2\},\qquad
t_i=\tau_{j_i+1},\qquad r_i=\tau_{j_i},\qquad r_i<t_i.
$$

- **高噪声/学生分支**：`precond(x, t, noise, labels)` 输出 $(x_{t,i},\mu_{h,i},\sigma_{h,i},d_i)$，代码变量分别是 `x_t`、`mu`、`std`、`denoiser_x`。该分支正常保留梯度。
- **低噪声/教师分支**：`precond(x, r, noise, labels)` 输出 $(x_{r,i},\mu_{l,i},\sigma_{l,i})$，代码变量分别是 `x_r`、`mu_r`、`std_r`。整个调用位于 `torch.no_grad()` 中，因此它是停止梯度的教师目标。
- 两条分支复用同一个 `noise`。当 $j_i=0$ 时，边界目标直接使用原图 $x_i$；UA-A-LMC 通过 `latent_mask = (idxs > 0)` 排除这些没有有效低噪声后验的样本。

这里的“高/低噪声”在当前 CoVAE 中具体表现为**相邻时间条件 $t_i>r_i$**。编码器接收原始 `x` 和由时间生成的 embedding，采样噪声用于潜变量重参数化；代码并没有字面构造 PDF 中的 $x_l=x+s_l\epsilon_l$、$x_h=x+s_h\epsilon_h$ 两张加性噪声图像。

## 3. 基础逐元素误差

`loss_mode` 和 `denoiser_loss_mode` 最终调用 `models/covae_base.py:_loss_fn`。对预测 $a$、目标 $b$：

$$
\ell(a,b)=
\begin{cases}
(a-b)^2, & \texttt{l2},\\
\sqrt{(a-b)^2+c^2}-c, & \texttt{huber},\\
\operatorname{BCEWithLogits}(a,b), & \texttt{bce},
\end{cases}
$$

其中 Huber/Charbonnier 形式的 $c=0.00054\sqrt{\operatorname{numel}(a_0)}$。各分量先逐元素计算，再对非 batch 维求和。

## 4. 原有 CoVAE Loss 各部分

### 4.1 输出空间一致性 / 重建项

目标 $y_i$ 为：

$$
y_i=
\begin{cases}
x_i, & j_i=0,\\
x_{r,i}, & j_i>0\ \text{且不是 BCE},\\
\operatorname{sigmoid}(x_{r,i}), & j_i>0\ \text{且使用 BCE}.
\end{cases}
$$

实际 loss 为：

$$
\mathcal L_{\mathrm{rec},i}
=w_{\mathrm{rec}}(t_i)\sum_p\ell(x_{t,i,p},\operatorname{sg}(y_{i,p})).
$$

$p$ 遍历图像/特征的所有非 batch 元素，`sg` 表示停止梯度；代码通过 `x_r.detach()` 实现。`rec_weight_mode` 决定：

$$
w_{\mathrm{rec}}(t)=
\begin{cases}
1/t,&\texttt{linear},\\
1/t^2,&\texttt{square},\\
1,&\texttt{ones}.
\end{cases}
$$

默认 `rec_weight_mode: linear`。该项对应 PDF 所称的原有“输出空间/像素一致性”项，而不是单独配置了一个 `lambda_pixel`。

### 4.2 KL 项

高噪声分支编码器给出：

$$
q_\phi(z\mid x,t_i)=\mathcal N(\mu_{h,i},\operatorname{diag}(\sigma_{h,i}^2)),
\qquad p(z)=\mathcal N(0,I).
$$

于是：

$$
\mathcal L_{\mathrm{KL},i}
=w_{\mathrm{KL}}(t_i)
\sum_d D_{\mathrm{KL}}\!\left(q_\phi(z_d\mid x,t_i)\,\|\,\mathcal N(0,1)\right).
$$

代码令 `w_KL = 1 / _get_loss_weights(t, kl_weight_mode)`，因此：

$$
w_{\mathrm{KL}}(t)=
\begin{cases}
t,&\texttt{linear},\\
t^2,&\texttt{square},\\
1,&\texttt{ones}.
\end{cases}
$$

默认 `kl_weight_mode: square`，即 $w_{\mathrm{KL}}(t)=t^2$。当前实现没有独立的 `lambda_KL`；时间权重就是该项的显式缩放。

### 4.3 直接去噪项

若 `denoiser_loss_mode` 非空：

$$
\mathcal L_{\mathrm{den},i}
=w_{\mathrm{rec}}(t_i)c_{\mathrm{den}}(t_i)
\sum_p\ell_{\mathrm{den}}(d_{i,p},x_{i,p}),
$$

$$
c_{\mathrm{den}}(t)
=\lambda_{\mathrm{den}}
+(1-\lambda_{\mathrm{den}})
\left(1-\frac{t-\sigma_{\min}}{\sigma_{\max}-\sigma_{\min}}\right).
$$

符号对应：`d_i = denoiser_x`，$\lambda_{\mathrm{den}}=$ `lambda_denoiser`，$\sigma_{\min},\sigma_{\max}=$ `sigma_min`, `sigma_max`。默认 `lambda_denoiser: 1.0`，因此 $c_{\mathrm{den}}(t)=1$。

### 4.4 可选 GAN 项

仅当 `use_gan=True` 且 `step >= gan_warmup_steps` 时启用。生成器基本目标为 $-D(\hat x)$，随后乘时间 mask、$w_{\mathrm{rec}}(t)$、训练进度和 `gan_lambda`；判别器使用 hinge loss：

$$
\mathcal L_D=\frac12\left[
\mathbb E\max(0,1-D(x))+
\mathbb E\max(0,1+D(\operatorname{sg}(\hat x)))
\right].
$$

默认配置关闭 GAN，因此它不参与当前默认总 loss。

## 5. UA-A-LMC 核心 Loss

### 5.1 非对称潜均值误差

对有效样本 $i$ 和潜维度/位置 $d$：

$$
e_{i,d}=\left(\mu_{h,i,d}-\operatorname{sg}(\mu_{l,i,d})\right)^2.
$$

代码传入 `_get_ua_lmc_loss(mu_h=mu, mu_l=mu_r, ...)`。虽然函数内部写作 `(mu_h - mu_l).square()`，但 `mu_r` 来自外层 `torch.no_grad()`，所以梯度只更新高噪声学生分支，不会让低噪声教师分支被高噪声表示反向拉动。这对应 PDF 的“非对称教师机制”。

### 5.2 样本级噪声可靠性权重

当前加性/VE 时间尺度使用：

$$
\operatorname{SNR}_{h,i}=\frac{1}{t_i^2+\varepsilon},
\qquad
\operatorname{SNR}_{l,i}=\frac{1}{r_i^2+\varepsilon}.
$$

高噪声置信度和噪声间隔权重为：

$$
w_{\mathrm{conf},i}
=\left(\frac{\operatorname{SNR}_{h,i}}
{\operatorname{SNR}_{h,i}+\kappa}\right)^\gamma,
$$

$$
w_{\mathrm{gap},i}
=\exp\left[-\eta\left|
\log\operatorname{SNR}_{l,i}-\log\operatorname{SNR}_{h,i}
\right|\right],
\qquad
w_{\mathrm{noise},i}=w_{\mathrm{conf},i}w_{\mathrm{gap},i}.
$$

- $\kappa=$ `ua_lmc_kappa`：SNR 置信度曲线的转折尺度；越大，整体权重越保守。
- $\gamma=$ `ua_lmc_gamma`：置信度曲线的锐度；越大，低 SNR 样本衰减越强。
- $\eta=$ `ua_lmc_eta`：相邻噪声层级距离的衰减强度；设为 0 即令 $w_{\mathrm{gap}}=1$。
- $\varepsilon=$ `ua_lmc_eps`：防止除零和 `log(0)`。

### 5.3 潜维度级不确定性权重

编码器返回的是标准差 `std`，所以代码先平方得到后验方差 $v_h=\sigma_h^2$、$v_l=\sigma_l^2$：

$$
u_{i,d}=\frac{1}{v_{l,i,d}+v_{h,i,d}+\varepsilon},
$$

$$
w_{\mathrm{unc},i,d}
=\operatorname{sg}\!\left[
\operatorname{clip}(u_{i,d},u_{\min},u_{\max})
\right],
$$

$$
\widetilde w_{\mathrm{unc},i,d}
=\frac{w_{\mathrm{unc},i,d}}
{\operatorname{mean}_{j,k}(w_{\mathrm{unc},j,k})+\varepsilon}.
$$

- $u_{\min}=$ `ua_lmc_umin`、$u_{\max}=$ `ua_lmc_umax`：维度置信度的截断上下界。
- `.detach()` 阻止模型通过主动增大后验方差来降低一致性惩罚。
- 均值归一化使该权重在不同 batch/训练阶段的平均尺度约为 1。

### 5.4 完整 UA-A-LMC 与预热

设有效样本集合 $\mathcal I=\{i\mid j_i>0\}$，其大小为 $B'$，每个样本共有 $D$ 个潜元素（包含通道和空间位置），代码的 `.mean()` 等价于：

$$
\boxed{
\mathcal L_{\mathrm{UA-A-LMC}}
=\frac{1}{B'D}\sum_{i\in\mathcal I}\sum_{d=1}^{D}
w_{\mathrm{noise},i}
\widetilde w_{\mathrm{unc},i,d}
\left(\mu_{h,i,d}-\operatorname{sg}(\mu_{l,i,d})\right)^2
}.
$$

它可直观理解为：

$$
\text{潜均值误差}\times\text{噪声对可靠性}\times\text{潜维度可靠性}.
$$

该项的总系数采用线性预热：

$$
R=\max\!\left(\left\lfloor
S_{\mathrm{total}}f_{\mathrm{warmup}}\right\rfloor,1\right),
\qquad
\lambda_{\mathrm{UA}}(s)
=\lambda_{\mathrm{latent}}
\min\left(\frac{s+1}{R},1\right).
$$

- $S_{\mathrm{total}}=$ `total_training_steps`。
- $f_{\mathrm{warmup}}=$ `ua_lmc_warmup_frac`，默认 0.1，即前 10% 训练步线性预热。
- $\lambda_{\mathrm{latent}}=$ `lambda_latent_consistency`，就是 PDF 中的 $\lambda_{UA}$，默认 0.1。
- 若 `ua_lmc_warmup_frac <= 0`，从第一步起直接使用完整 `lambda_latent_consistency`。
- 若 `lambda_latent_consistency <= 0`，UA-A-LMC 完全关闭。

## 6. 重点符号、代码变量与配置速查

| 数学符号 | 代码变量/配置 | 含义 | 默认值 |
| --- | --- | --- | --- |
| $s$ | `step` | 当前模型训练步 | 运行时变量 |
| $t_i$ | `t` | 高噪声/学生时间尺度 | 随机采样 |
| $r_i$ | `r` | 相邻的低噪声/教师时间尺度 | 随机采样，$r_i<t_i$ |
| $\mu_h,\sigma_h$ | `mu`, `std` | 学生后验均值与标准差 | 编码器输出 |
| $\mu_l,\sigma_l$ | `mu_r`, `std_r` | 无梯度教师后验均值与标准差 | 编码器输出 |
| $v_h,v_l$ | `std.square()`, `std_r.square()` | 两分支后验方差 | 运行时变量 |
| $\lambda_{UA}$ | `lambda_latent_consistency` | UA-A-LMC 最大权重 | `0.1` |
| $\kappa$ | `ua_lmc_kappa` | SNR 权重转折参数 | `1.0` |
| $\gamma$ | `ua_lmc_gamma` | SNR 权重锐度 | `2.0` |
| $\eta$ | `ua_lmc_eta` | 噪声距离衰减系数 | `0.5` |
| $u_{\min},u_{\max}$ | `ua_lmc_umin`, `ua_lmc_umax` | 不确定性权重截断范围 | `0.5`, `2.0` |
| $\varepsilon$ | `ua_lmc_eps` | 数值稳定常数 | `1e-6` |
| $f_{\mathrm{warmup}}$ | `ua_lmc_warmup_frac` | UA-A-LMC 预热占总步数比例 | `0.1` |
| $w_{\mathrm{rec}}$ | `rec_weight_mode` | 输出一致性时间权重模式 | `linear` |
| $w_{\mathrm{KL}}$ | `kl_weight_mode` | KL 时间权重模式 | `square` |
| $\lambda_{\mathrm{den}}$ | `lambda_denoiser` | 直接去噪项的时间插值下限 | `1.0` |

这些配置定义在 `conf/model/covae.yaml`，经 `utils/model_utils.py` 传入 `CoVAE`。UA-A-LMC 只支持 `latent_type: gaussian`；当其权重大于 0 且使用 categorical posterior 时，构造函数会直接抛出 `NotImplementedError`。

## 7. 是否已实现 PDF 中的改进？

**结论：核心算法已经实现，而且默认 `model=covae` 配置会启用它；但不能把 PDF 中所有文字都理解为已逐字实现，也不能说报告要求的实验已经完成。**

| PDF 改进/要求 | 当前状态 | 代码依据与说明 |
| --- | --- | --- |
| 非对称教师：低噪声均值停止梯度 | 已实现 | `models/covae.py:212-224`：低噪声分支整体位于 `torch.no_grad()`，UA loss 只向高噪声分支传播 |
| 高噪声 SNR 置信度权重 | 已实现 | `models/covae.py:131-135`：`w_conf` 使用 $\mathrm{SNR}_h/(\mathrm{SNR}_h+\kappa)$ 的 $\gamma$ 次方 |
| 噪声距离权重 | 已实现 | 同上：`w_gap = exp(-eta * abs(log(snr_l)-log(snr_h)))` |
| 后验方差不确定性权重 | 已实现 | `models/covae.py:137-141`：两分支方差之和的倒数、截断、停止梯度、均值归一化 |
| UA-A-LMC 加入总 loss | 已实现 | `models/covae.py:215-227,272-273` |
| $\lambda_{UA}$ 线性预热 | 已实现 | `models/covae.py:122-128`，以总训练步比例配置，而不是直接配置绝对步数 $R$ |
| 不增加网络或额外编码次数 | 基本符合 | 没有新增网络和可训练参数；复用原 CoVAE 已有的高/低时间分支编码结果 |
| 显式构造 $x_l=x+s_l\epsilon_l$、$x_h=x+s_h\epsilon_h$ | 未按字面实现 | 当前编码器输入仍是 `x`，高低噪声语义由时间条件 `t/r` 和潜变量采样表达；这是适配现有 CoVAE 的实现，而非两张显式加噪图 |
| 报告中的对比、消融、超参数、鲁棒性实验 | 无法由 loss 代码认定为已完成 | PDF 第 6 节是实验计划；仓库虽有 FID/重建回调和已有运行结果，但没有完整、结构化地覆盖报告列出的 A-E 消融及全部指标 |

另有两个应当知道的实现差异：

1. PDF 把总目标抽象写成 $\mathcal L_{recon}+\lambda_{KL}\mathcal L_{KL}+\lambda_{pixel}\mathcal L_{pixel}+\lambda_{UA}\mathcal L_{UA}$；当前 CoVAE 没有独立的 `lambda_KL` 和 `lambda_pixel`，而是分别使用 `kl_weight_mode` 与 `rec_weight_mode` 生成随时间变化的权重，并额外包含直接去噪项及可选 GAN 项。
2. PDF 写“对 $\mu_l$ 和权重 detach”；当前代码对低噪声整条前向使用 `torch.no_grad()`，对不确定性权重显式 `.detach()`。两者在 UA-A-LMC 的梯度方向上满足报告目的，并且前者更节省教师分支的 autograd 内存。

## 8. 三个动态权重的范围与消融方法（实验重点）

除外层随训练步预热的 `lambda_latent_consistency` 外，新增 loss 内部实际还有三个随样本、噪声层级或后验输出动态变化的乘法权重：`w_conf`、`w_gap`、`w_unc`。因此，一个潜元素最终承受的动态权重是

$$
w_{i,d}=w_{\mathrm{conf},i}\,w_{\mathrm{gap},i}\,
\widetilde w_{\mathrm{unc},i,d}.
$$

下面的范围假设当前合理配置 $\kappa,\gamma,\eta\ge 0$、$0<u_{\min}\le u_{\max}$，并采用当前默认值 `sigma_min=0.05`、`sigma_max=3`、`kappa=1`、`gamma=2`、`eta=0.5`、`umin=0.5`、`umax=2.0`、`eps=1e-6`。

| 动态权重 | 理论/配置范围 | 当前默认配置下的范围 | 说明 |
| --- | --- | --- | --- |
| $w_{\mathrm{conf}}=[1+\kappa(t_h^2+\varepsilon)]^{-\gamma}$ | $(0,1]$；在 $t_h\in[\sigma_{\min},\sigma_{\max}]$ 上有明确闭区间 | 约 $[0.0100,\ 0.9950]$ | $t_h$ 越大（噪声越强），权重越小。两个端点分别由 `sigma_max` 和 `sigma_min` 给出。 |
| $w_{\mathrm{gap}}=\exp[-\eta|\log\mathrm{SNR}_l-\log\mathrm{SNR}_h|]$ | $(0,1]$；在配置的噪声区间内，下界不小于 $\exp[-\eta\log\frac{\sigma_{\max}^2+\varepsilon}{\sigma_{\min}^2+\varepsilon}]$ | 全区间保守包络约 $[0.01667,\ 1]$ | 代码只抽取**相邻**且 $t_l>0$ 的层级，所以一次实际运行的最小值通常明显高于该保守下界；因相邻层级不同，实际值通常也严格小于 1，但层级变密时可趋近 1。 |
| 截断后的 $w_{\mathrm{unc}}$（归一化前） | $[u_{\min},u_{\max}]$ | $[0.5,2.0]$ | 这是 `clamp` 直接保证的范围，但还不是进入 loss 的最终权重。 |
| $\widetilde w_{\mathrm{unc}}=w_{\mathrm{unc}}/(\operatorname{mean}(w_{\mathrm{unc}})+\varepsilon)$（归一化后） | 单个元素的保守范围为 $[u_{\min}/(u_{\max}+\varepsilon),\ u_{\max}/(u_{\min}+\varepsilon)]$，batch 全体均值约为 1 | 约 $(0.25,4.0)$ 的保守包络，实际均值约为 1 | 归一化会改变逐元素上下界，所以不能把最终 `w_unc` 误写成 `[0.5,2.0]`。由于分母是同一批权重的均值，有限 batch 中通常达不到两个保守端点。 |

特别注意：三个权重的乘积没有一个与上述单项范围同样紧的固定范围。按默认参数把三个保守包络直接相乘只能得到非常宽松的约 $(4.17\times10^{-5},3.98)$；实际值受“相邻时间层级”和 `w_unc` 的 batch 均值归一化共同约束，不能据此判断典型权重大小，实验中应分别记录三个权重的均值、最小值和最大值。

### 8.1 推荐消融实验表

这些参数已存在于 `conf/model/covae.yaml`，可直接在 `run.slurm` 的 `python main.py ...` 命令末尾追加 Hydra override。每组实验应只改变表中所列项，其余配置、随机种子、训练步数和评估流程保持一致。

| 实验 | 保留的动态机制 | 在 `run.slurm` 中追加/修改 | 是否需要改 `models/covae.py` |
| --- | --- | --- | --- |
| A：完整 UA-A-LMC（基线） | `w_conf + w_gap + w_unc` | `model.ua_lmc_kappa=1.0 model.ua_lmc_gamma=2.0 model.ua_lmc_eta=0.5 model.ua_lmc_umin=0.5 model.ua_lmc_umax=2.0` | 否 |
| B：去掉 SNR 置信度权重 | `w_gap + w_unc` | `model.ua_lmc_kappa=0`，此时 `w_conf=1` | 否 |
| C：去掉噪声间隔权重 | `w_conf + w_unc` | `model.ua_lmc_eta=0`，此时 `w_gap=1` | 否 |
| D：去掉后验不确定性权重 | `w_conf + w_gap` | 近似方案：`model.ua_lmc_umin=1 model.ua_lmc_umax=1` | 严格置 1 时需要，见下文 |
| E：三个动态权重全部去掉（普通非对称 LMC） | 均不保留 | `model.ua_lmc_kappa=0 model.ua_lmc_eta=0 model.ua_lmc_umin=1 model.ua_lmc_umax=1` | 严格置 1 时需要，见下文 |
| F：关闭整个新增 loss（CoVAE 对照） | UA-A-LMC 不参与总 loss | `model.lambda_latent_consistency=0` | 否 |

`umin=umax=1` 后，当前第 141 行仍会计算 $1/(1+\varepsilon)$，数值约为 `0.999999`，对实验而言可视为 1。若要求数学上严格的消融，应在 D/E 实验分支中将 `models/covae.py` 的第 137--141 行替换为：

```python
w_unc = torch.ones_like(mu_h)
```

不要通过把 `ua_lmc_eps` 改为 0 来追求严格等于 1，因为该参数还用于 SNR 除法和对数的数值稳定。也不建议用极端的 `umin/umax` 间接消融。最稳妥的实验顺序是先跑 A、B、C、D 四组单因素实验，再用 E 判断三个动态权重整体的贡献，最后用 F 区分“普通潜一致性本身”与完整 UA-A-LMC 的收益。

### 8.2 权重尺度分析

| 训练步 | `w_noise` 平均值 | 中位数 | 最大值 |
| ---: | ---: | ---: | ---: |
| 0 | 0.031 | 0.060 | 0.060 |
| 100k | 0.274 | 0.370 | 0.513 |
| 200k | 0.463 | 0.546 | 0.838 |
| 300k | 0.526 | 0.600 | 0.952 |
| 400k | 0.537 | 0.610 | 0.974 |

---

# 以下为原 README 详细内容（完整保留）

# Current Training Loss Implementation

This README is derived from the current implementation in this repository. It documents the code as implemented, not a paper or external design document.

Primary inspected files:

- `models/covae.py`
- `models/covae_base.py`
- `utils/model_utils.py`
- `lightning_modules/lightning_cm.py`
- `conf/model/covae.yaml`
- `conf/config.yaml`, `conf/network/autoencoder.yaml`, and dataset configs under `conf/dataset/`

The default Hydra config selects `model: covae`, so the main loss below is the `CoVAE.loss` implementation selected by `utils/model_utils.py:get_model` when `cfg.model.name == 'covae'`.

## 1. Overall Loss

### Default `CoVAE` Generator/Autoencoder Loss

For a batch `x = {x_i}_{i=1}^B` and training step `s`, `models/covae.py:CoVAE.loss` samples one adjacent time interval per batch item.

Let:

- `N_s = self._step_schedule(s)` from `models/covae_base.py:CoVAEBase._step_schedule`.
- `tau = self._get_time_steps(N_s)` from `models/covae.py:CoVAE._get_time_steps`, with `tau_0 = 0` prepended to the configured time grid.
- `j_i ~ Uniform({0, ..., len(tau)-2})`.
- `t_i = tau_{j_i+1}` and `r_i = tau_{j_i}`.
- `epsilon_i = self.sample_noise(...)`.
- `(x_t_i, mu_t_i, std_t_i, den_i) = precond(x_i, t_i, epsilon_i)`.
- `(x_r_i, mu_r_i, _, _) = precond(x_i, r_i, epsilon_i)` under `torch.no_grad()`, unless the whole batch has `j_i == 0`, in which case `x_r = x` and `mu_r = None`.

The total returned loss is:

```text
L_total(s) = (1 / B) * sum_i [
    L_rec_i
  + L_den_i
  + L_kl_i
] + L_gan(s) + lambda_latent_consistency * L_latent
```

where disabled terms are exactly zero in the code.

#### Base Elementwise Loss

Implemented in `models/covae_base.py:CoVAEBase._loss_fn`.

For prediction `a`, target `b`, and mode `m`:

```text
ell_m(a, b) =
  (a - b)^2                                      if m == 'l2'
  sqrt((a - b)^2 + c^2) - c                     if m == 'huber'
  BCEWithLogits(a, b) with reduction='none'     if m == 'bce'
```

For Huber mode:

```text
c = 0.00054 * sqrt(numel(a[0]))
```

The code computes this elementwise, then sums over non-batch dimensions where each loss component needs a per-example scalar.

#### Reconstruction Term

Implemented in `models/covae.py:CoVAE.loss` and weighted by `models/covae.py:CoVAE._get_rec_loss_weights`.

The target is:

```text
y_i = x_i                         if j_i == 0
y_i = sigmoid(x_r_i)              if j_i > 0 and loss_mode == 'bce'
y_i = x_r_i                       if j_i > 0 and loss_mode != 'bce'
```

Then:

```text
L_rec_i = w_rec(t_i) * sum_d ell_loss_mode(x_t_i[d], stopgrad(y_i[d]))
```

with:

```text
w_rec(t) =
  1 / t       if rec_weight_mode == 'linear'
  1 / t^2     if rec_weight_mode == 'square'
  1           if rec_weight_mode == 'ones'
```

Purpose: train the model output at the higher/noisier time `t_i` to match the lower-time target `r_i`, with a boundary target of the original input when `r_i = 0`.

#### KL Term

Implemented in `models/covae.py:CoVAE.loss`, with distributions built by `models/covae.py:CoVAE._get_distribution`.

The posterior and prior are:

```text
q_i = distribution(mu_t_i, std_t_i + 1e-8)
p_i = distribution(0, 1)
```

For `latent_type == 'gaussian'`, this is `Normal(mu, std)`. For `latent_type == 'categorical'`, `mu` is reshaped and used as categorical logits; the prior uses zero logits.

The contribution is:

```text
L_kl_i = w_kl(t_i) * sum_k KL(q_i,k || p_i,k)
```

where:

```text
w_kl(t) = 1 / get_loss_weights(t, kl_weight_mode)
```

Equivalently:

```text
w_kl(t) =
  t       if kl_weight_mode == 'linear'
  t^2     if kl_weight_mode == 'square'
  1       if kl_weight_mode == 'ones'
```

Purpose: regularize the encoded latent distribution toward the configured prior.

#### Denoiser Term

Implemented in `models/covae.py:CoVAE.loss`; the denoiser output is produced in `models/covae.py:CoVAE._decode_fn`.

This term is enabled when `self.denoiser_loss_mode` is truthy. In that case the decoder output is split along the channel dimension:

```text
raw_delta_i, den_i = chunk(decoder(z_i, emb_i), 2, dim=channel)
x_t_i = stopgrad(den_i) + alpha(t_i) * raw_delta_i
alpha(t) = (t - sigma_min) / (sigma_max - sigma_min)
```

The denoiser loss contribution is:

```text
beta(t) = lambda_denoiser + (1 - lambda_denoiser) * (1 - (t - sigma_min) / (sigma_max - sigma_min))

L_den_i = w_rec(t_i) * sum_d beta(t_i) * ell_denoiser_loss_mode(den_i[d], x_i[d])
```

If `denoiser_loss_mode` is false, `L_den_i = 0`.

Purpose: train the auxiliary denoiser output `den_i` directly toward the clean input while the main output uses a detached denoiser skip connection plus a scaled residual.

#### Latent Consistency Term

Implemented in `models/covae.py:CoVAE.loss`.

This term is enabled only when:

```text
lambda_latent_consistency > 0 and mu_r is not None
```

The implementation masks out interval samples with `j_i == 0`:

```text
L_latent = mean_{i: j_i > 0} sum_k (mu_t_i[k] - stopgrad(mu_r_i[k]))^2
```

If disabled, or if the whole batch has `j_i == 0`, the code sets this term to a scalar zero tensor.

Purpose: encourage latent means from adjacent time levels to stay close. The current `LMC-VAE` implementation uses plain squared mean distance; it does not use uncertainty weighting in `models/covae.py`.

#### GAN Generator Term

Implemented in `models/covae.py:CoVAE.loss`.

This term is enabled only when:

```text
use_gan == True and step >= gan_warmup_steps
```

If denoiser mode is enabled, the generator input to the discriminator is:

```text
gan_input_i = x_t_i      if j_i > 1
gan_input_i = den_i      otherwise
```

If denoiser mode is disabled:

```text
gan_input_i = x_t_i
```

The per-example generator score is:

```text
g_i = -mean(discriminator(clamp(gan_input_i, -1, 1)))
```

The time mask is:

```text
mask_idx = int(end_scales / ((total_training_steps - gan_warmup_steps) / (step + 1 - gan_warmup_steps)))
mask_i = 1 if t_i <= full_tau[mask_idx] else 0
```

where `full_tau = self._get_time_steps(end_scales + 1)`.

The final generator adversarial contribution is:

```text
progress(s) = (step + 1 - gan_warmup_steps) / (total_training_steps - gan_warmup_steps)

L_gan(s) = mean_i [
    g_i * mask_i * w_rec(t_i) * progress(s) * gan_lambda
]
```

If disabled, `L_gan(s) = 0`.

Purpose: adversarially train generated/reconstructed outputs after the warmup period.

### Discriminator Loss

The discriminator has its own optimizer step in `lightning_modules/lightning_cm.py:LightningConsistencyModel.training_step`. It is not included in `L_total`; it is optimized separately after the generator/autoencoder update when:

```text
use_gan == True and step >= model.gan_warmup_steps
```

Implemented in `models/covae_base.py:CoVAEBase.discriminator_loss` and `models/covae_base.py:CoVAEBase.hinge_d_loss`:

```text
logits_real = discriminator(real)
logits_fake = discriminator(clamp(fake, -1, 1))

L_D = 0.5 * [
    mean(ReLU(1 - logits_real))
  + mean(ReLU(1 + logits_fake))
]
```

The fake tensor passed to this discriminator loss is `x_generated`, the `x_t` returned by `CoVAE.loss`.

### Alternate `CoVAESimple` Loss Path

The repository also contains `models/covae_simple.py`, selected only when `cfg.model.name == 'covae_simple'` in `utils/model_utils.py:get_model`. Its loss is different and does not include the KL or latent consistency terms from `CoVAE.loss`.

For `CoVAESimple.loss`:

```text
L_simple = mean_i [
    w_i * sum_d ell_loss_mode(x_t_i[d], stopgrad(y_i[d]))
  + I_norm * norm_strength * w_norm_i * sum_k z0_i[k]^2
  + I_den * c_skip_i * w_i * sum_d ell_denoiser_loss_mode(den_i[d], x_i[d])
] + L_gan_simple
```

where:

```text
w_i = 1 / (sigma_{j_i+1} - sigma_{j_i})
```

`I_norm` is enabled when `norm_strength > 0`; `w_norm_i` is either the fixed first-interval weight or the adaptive current interval weight depending on `norm_weight`. `I_den` is enabled when `denoiser_loss_mode` is truthy. `L_gan_simple` follows the same adversarial structure as `CoVAE`, but uses `w_i` from `CoVAESimple._get_loss_weights`.

## 2. Loss Components

### Reconstruction Loss

- Input variables: `x_t`, `x_r`, `x`, `idxs`, `t`, `loss_mode`, `rec_weight_mode`.
- Output: per-example weighted reconstruction scalar `L_rec_i` and logged unweighted `rec_loss`.
- Weighting coefficient: `w_rec(t)` from `rec_weight_mode`.
- Enabled: always in `CoVAE.loss`.
- Disabled: never disabled by a config flag.
- Final contribution: included inside `(rec_loss + denoiser_loss + kl_loss).mean()`.

### KL Loss

- Input variables: `mu`, `std`, `latent_type`, `t`, `kl_weight_mode`.
- Output: per-example weighted KL scalar `L_kl_i` and logged unweighted `kl_loss`.
- Weighting coefficient: `w_kl(t) = 1 / get_loss_weights(t, kl_weight_mode)`.
- Enabled: always in `CoVAE.loss`.
- Disabled: not disabled by a config flag in `CoVAE.loss`.
- Final contribution: included inside `(rec_loss + denoiser_loss + kl_loss).mean()`.

### Denoiser Loss

- Input variables: `denoiser_x`, `x`, `t`, `denoiser_loss_mode`, `lambda_denoiser`, `sigma_min`, `sigma_max`, `rec_weight_mode`.
- Output: per-example weighted denoiser scalar `L_den_i` and logged unweighted `denoiser_loss`.
- Weighting coefficient: `w_rec(t) * beta(t)`, where `beta(t)` is defined in the formula above.
- Enabled: when `denoiser_loss_mode` is truthy.
- Disabled: when `denoiser_loss_mode` is false, in which case the code sets `denoiser_loss = 0.`.
- Final contribution: included inside `(rec_loss + denoiser_loss + kl_loss).mean()`.

### Latent Consistency Loss

- Input variables: `mu`, `mu_r`, `idxs`, `lambda_latent_consistency`.
- Output: one scalar `L_latent` and logged `latent_consistency_loss` when enabled.
- Weighting coefficient: `lambda_latent_consistency`.
- Enabled: when `lambda_latent_consistency > 0` and `mu_r is not None`.
- Disabled: when `lambda_latent_consistency <= 0`, or when the whole batch has `idxs == 0`.
- Final contribution: added after the main mean as `lambda_latent_consistency * latent_consistency_loss`.

### GAN Generator Loss

- Input variables: `x_t`, `denoiser_x`, `idxs`, `t`, discriminator output, `gan_lambda`, `gan_warmup_steps`, `total_training_steps`, `end_scales`, `rec_weight_mode`.
- Output: scalar `L_gan` and logged `generator_loss` before weighting by `w_rec`, progress, and `gan_lambda`.
- Weighting coefficient: `w_rec(t) * progress(step) * gan_lambda * mask`.
- Enabled: when `use_gan` is true and `step >= gan_warmup_steps`.
- Disabled: when `use_gan` is false or the warmup has not completed.
- Final contribution: added to the generator/autoencoder loss as `+ gan_loss`.

### Discriminator Hinge Loss

- Input variables: real batch `inputs`, fake `x_generated`, discriminator logits.
- Output: scalar discriminator loss.
- Weighting coefficient: none beyond the hardcoded `0.5` in hinge loss.
- Enabled: when `use_gan` is true and `step >= model.gan_warmup_steps`.
- Disabled: when `use_gan` is false or the warmup has not completed.
- Final contribution: optimized in a separate manual optimizer step; not included in `L_total` returned by `CoVAE.loss`.

### `CoVAESimple`-Only Norm Loss

- Input variables: encoded latent `z_0`, `norm_strength`, `norm_weight`, interval loss weights.
- Output: weighted latent norm penalty.
- Weighting coefficient: `norm_strength * w_norm`.
- Enabled: only in `CoVAESimple.loss` when `norm_strength > 0`.
- Disabled: when `norm_strength <= 0`, and not present in `CoVAE.loss`.
- Final contribution: added to `total_loss` before the final batch mean in `CoVAESimple.loss`.

## 3. Hyperparameters

The table below lists training-related configuration keys found in the inspected config files and code paths. Defaults are taken from the current YAML files. Dataset values list all dataset config defaults when they differ by dataset.

| Hyperparameter | Default Value | Description |
| --- | --- | --- |
| `defaults.dataset` | `mnist` | Default Hydra dataset config group selected by `conf/config.yaml`. |
| `defaults.model` | `covae` | Default Hydra model config group selected by `conf/config.yaml`. |
| `defaults.network` | `autoencoder` | Default Hydra network config group selected by `conf/config.yaml`. |
| `use_logger` | `True` | Enables W&B logger creation and config logging. |
| `project` | `covae` | W&B project name. |
| `fast_dev_run` | `False` | Passed to `lightning.Trainer`. |
| `enable_progress_bar` | `False` | Passed to `lightning.Trainer`. |
| `root_dir` | `.` | W&B save directory and Lightning default root directory. |
| `requeue` | `True` | Present in config; not read by the inspected training path. |
| `log_samples` | `True` | Enables sample generation callbacks. |
| `compute_fid` | `True` | Enables FID callbacks and model checkpoint callback. |
| `log_rec` | `False` | Enables diagnostic reconstruction callback. |
| `compute_rec_fid` | `False` | Enables reconstruction FID behavior in callbacks. |
| `seed` | `42` | Global seed and rank-adjusted per-process seed. |
| `extra_name` | `''` | Present in config; not read by the inspected training path. |
| `reload` | `False` | Enables W&B artifact checkpoint reload path. |
| `run_path` | `''` | W&B run path/id used during reload/resume. |
| `ckpt_path` | `''` | Local checkpoint path used for checkpoint resume. |
| `log_model` | `all` | Passed to `WandbLogger(log_model=...)`. |
| `deterministic` | `False` | Passed to `lightning.Trainer`. |
| `sync_batchnorm` | `False` | Passed to `lightning.Trainer`. |
| `log_frequency` | `10000` | Trainer logging interval; also callback/checkpoint interval. |
| `precision` | `bf16-mixed` | Passed to `lightning.Trainer`. |
| `accumulate_grad_batches` | `1` | Passed to `lightning.Trainer`. |
| `accelerator` | `auto` | Passed to `lightning.Trainer`. |
| `strategy` | `auto` | Passed to `lightning.Trainer`. |
| `devices` | `auto` | Passed to `lightning.Trainer`; also used to compute `batch_multiplier`. |
| `gradient_clip_val` | `0` | If greater than zero, clips generator/autoencoder gradients by norm. |
| `batch_multiplier` | `1` | Used in run naming; overwritten with GPU count when multiple devices are detected. |
| `model.name` | `covae` | Selects `CoVAE` when equal to `covae`; can select `CoVAESimple` when set to `covae_simple`. |
| `model.ema_rate` | `0.9999` | EMA interpolation rate used by `LightningConsistencyModel.ema_update`. |
| `model.learning_rate` | `1e-4` | Learning rate for RAdam optimizers. |
| `model.weight_decay` | `0` | Weight decay for RAdam optimizers. |
| `model.step_schedule` | `exp` | Schedule for number of time steps: `exp` or `none`. |
| `model.sigma_min` | `0.05` | Minimum sigma/time value. |
| `model.sigma_max` | `3` | Maximum sigma/time value. |
| `model.rho` | `7` | Karras sigma schedule exponent. |
| `model.start_scales` | `2` | Initial scale count for exponential step schedule. |
| `model.end_scales` | `256` | Maximum scale count for schedule and GAN masking. |
| `model.total_training_steps` | `400000` | Main configured training length and schedule horizon. |
| `model.loss_mode` | `huber` | Elementwise loss mode for reconstruction. |
| `model.denoiser_loss_mode` | `l2` | Elementwise loss mode for denoiser term; falsey disables denoiser branch. |
| `model.use_gan` | `False` | Enables generator adversarial loss and discriminator optimizer step. |
| `model.gan_warmup_steps` | `0` | Step threshold before GAN terms start. |
| `model.gan_lambda` | `1` | Scalar multiplier for generator adversarial loss. |
| `model.time_scale` | `karras` | Time grid mode for `CoVAE`: `linear`, `log`, or `karras`. |
| `model.rec_weight_mode` | `linear` | Reconstruction and denoiser time weighting mode. |
| `model.kl_weight_mode` | `square` | KL time weighting mode. |
| `model.lambda_denoiser` | `1.` | Controls `beta(t)` in the `CoVAE` denoiser loss. |
| `model.lambda_latent_consistency` | `0.` | Scalar multiplier for plain latent mean consistency in `CoVAE.loss`. |
| `model.kernel` | `ve` | `CoVAESimple` kernel selector: variance exploding or linear interpolant. |
| `model.p_mean` | `-1.1` | `CoVAESimple` lognormal timestep sampling mean. |
| `model.p_std` | `2.0` | `CoVAESimple` lognormal timestep sampling standard deviation. |
| `model.sigma_data` | `0.5` | Kernel scaling parameter used by `CoVAESimple` kernels. |
| `model.norm_strength` | `0.001` | `CoVAESimple` latent norm penalty strength. |
| `model.mid_t` | `0.821` | Present in `CoVAESimple` constructor and config; stored as `self.mid_t`. |
| `model.norm_weight` | `fixed` | `CoVAESimple` norm penalty weighting mode: `fixed` or `adaptive`. |
| `model.noise_schedule` | `lognormal` | `CoVAESimple` interval sampling mode: `uniform` or `lognormal`. |
| `model.latent_type` | `gaussian` | Latent distribution type for `CoVAE`: `gaussian` or `categorical`. |
| `model.latent_shape` | `[]` | Shape used for categorical latent reshaping. |
| `network.name` | `autoencoder` | Selects `AutoEncoder` in `utils/model_utils.py:get_neural_net`. |
| `network.model_channels` | `64` | Base channel count for encoder/decoder. |
| `network.channel_mult_enc` | `[2, 2, 2]` | Encoder channel multipliers and latent spatial downsampling depth. |
| `network.channel_mult_dec` | `[2, 2, 2]` | Decoder channel multipliers. |
| `network.num_blocks_enc` | `2` | Number of encoder blocks per resolution. |
| `network.num_blocks_dec` | `2` | Number of decoder blocks per resolution. |
| `network.attn_resolutions` | `[16]` | Resolutions where attention blocks are enabled. |
| `network.dropout` | `0.2` | Dropout probability in UNet blocks. |
| `network.z_channels` | `4` | Latent channel count when `network.final_dim` is not an integer. |
| `network.final_dim` | `None` | If integer, switches latent shape to `[final_dim]`; otherwise uses spatial latent shape. |
| `dataset.name` | `mnist`; variants: `cifar10`, `celeba64` | Dataset selector used by `utils/datamodule_utils.py:get_datamodule`. |
| `dataset.size` | `celeba64: 64` | CelebA resize/size parameter; only present in `conf/dataset/celeba64.yaml`. |
| `dataset.batch_size` | `mnist: 128`; `cifar10: 128`; `celeba64: 128` | Batch size per device passed to datamodules. |
| `dataset.num_workers` | `mnist: 8`; `cifar10: 8`; `celeba64: 8` | DataLoader worker count passed to datamodules. |
| `dataset.data_dir` | `.` for all dataset configs | Dataset storage path. |
| `dataset.sample_shape` | `mnist: [256, 1, 28, 28]`; `cifar10: [256, 3, 32, 32]`; `celeba64: [256, 3, 64, 64]` | Sample tensor shape used by generation callbacks. |
| `dataset.fid_sample_shape` | `mnist: [500, 1, 28, 28]`; `cifar10: [500, 3, 32, 32]`; `celeba64: [500, 3, 64, 64]` | Sample tensor shape used by FID callbacks. |
| `dataset.n_dataset_samples` | `50000` for all dataset configs | Reference sample count passed to FID callbacks. |
| `dataset.input_shape` | `mnist: [1, 28, 28]`; `cifar10: [3, 32, 32]`; `celeba64: [3, 64, 64]` | Configured input shape; not directly read by inspected training code. |
| `dataset.plot_type` | `grid` for all dataset configs | Plot mode passed to generation callbacks. |
| `dataset.img_resolution` | `mnist: 28`; `cifar10: 32`; `celeba64: 64` | Image resolution used to construct the autoencoder and latent shape. |
| `dataset.in_channels` | `mnist: 1`; `cifar10: 3`; `celeba64: 3` | Input channel count for autoencoder and discriminator. |
| `dataset.out_channels` | `mnist: 1`; `cifar10: 3`; `celeba64: 3` | Decoder output channel count. Denoiser mode chunks decoder output into two channel groups. |

## 4. Source Code Reference

| Component | File | Function |
| --- | --- | --- |
| Model selection and config-to-constructor mapping | `utils/model_utils.py:48-100` | `get_model` |
| Autoencoder construction | `utils/model_utils.py:26-46` | `get_neural_net` |
| Discriminator construction | `utils/model_utils.py:19-24` | `get_discriminator` |
| Lightning training step and optimizer flow | `lightning_modules/lightning_cm.py:23-74` | `configure_optimizers`, `training_step` |
| Total `CoVAE` generator/autoencoder loss | `models/covae.py:151-227` | `CoVAE.loss` |
| Time grid for `CoVAE` | `models/covae.py:73-89` | `CoVAE._get_time_steps` |
| Step schedule | `models/covae_base.py:78-94` | `CoVAEBase._step_schedule` |
| Karras sigma schedule | `models/covae_base.py:63-76` | `CoVAEBase._get_sigmas_karras` |
| Base elementwise loss modes | `models/covae_base.py:52-61` | `CoVAEBase._loss_fn` |
| Reconstruction loss | `models/covae.py:182-190`, `models/covae.py:218` | `CoVAE.loss` |
| Reconstruction/denoiser time weights | `models/covae.py:91-104` | `CoVAE._get_rec_loss_weights`, `CoVAE._get_loss_weights` |
| KL distribution construction | `models/covae.py:36-41` | `CoVAE._get_distribution` |
| KL loss | `models/covae.py:219-223` | `CoVAE.loss` |
| KL time weights | `models/covae.py:94-104` | `CoVAE._get_kl_loss_weights`, `CoVAE._get_loss_weights` |
| Denoiser output split and skip output | `models/covae.py:106-113` | `CoVAE._decode_fn` |
| Denoiser loss | `models/covae.py:193-200` | `CoVAE.loss` |
| Latent consistency loss | `models/covae.py:175-180`, `models/covae.py:226` | `CoVAE.loss` |
| GAN generator loss | `models/covae.py:202-216` | `CoVAE.loss` |
| Discriminator hinge loss | `models/covae_base.py:96-106` | `hinge_d_loss`, `discriminator_loss` |
| `CoVAESimple` total loss | `models/covae_simple.py:100-174` | `CoVAESimple.loss` |
| `CoVAESimple` interval weights | `models/covae_simple.py:56-57` | `CoVAESimple._get_loss_weights` |
| `CoVAESimple` timestep sampling | `models/covae_simple.py:36-54` | `_lognormal_timestep_distribution`, `_get_indices` |
| Callback construction for sampling, FID, diagnostics, checkpoints | `utils/callback_utils.py:10-50` | `get_callbacks`, `get_delete_checkpoints_callback` |
| Datamodule selection | `utils/datamodule_utils.py:7-21` | `get_datamodule` |

## 5. 在非 Slurm 节点上通过 SSH 启动 CIFAR-10 四卡训练

本节适用于 `gpu2`、`gpu3`、`gpu4` 这类已经移出 Slurm 管理、但单个节点内有 4 张 GPU 的机器。以下命令只使用一台机器内的 4 张卡，不是跨三台机器训练。

### 5.1 登录节点并创建持久终端

以 `gpu2` 为例：

```bash
ssh gpu2
tmux new -s ua-lmc-cifar
```

激活环境并检查 4 张卡是否可用：

```bash
source /home/lyy/software/miniconda3/etc/profile.d/conda.sh
conda activate covae
cd /home/lyy/VAE/UA-A-LMC-VAE

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
nvidia-smi
```

如果需要退出 SSH 但保留训练，按 `Ctrl-b`，再按 `d`，使 tmux 会话在后台继续运行。重新连接后使用：

```bash
tmux attach -t ua-lmc-cifar
```

### 5.2 启动完整 UA-A-LMC-VAE 四卡实验

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python main.py \
  --config-path conf \
  --config-name config \
  dataset=cifar10 \
  'devices=[0,1,2,3]' \
  strategy=ddp \
  model=covae \
  log_frequency=10000 \
  log_model=all \
  project=ua-a-lmc-cifar \
  dataset.num_workers=16 \
  dataset.batch_size=256 \
  model.total_training_steps=400000 \
  compute_rec_fid=True \
  compute_fid=False \
  model.step_schedule=exp \
  model.start_scales=2 \
  model.end_scales=256 \
  model.sigma_min=0.05 \
  model.sigma_max=3 \
  model.time_scale=karras \
  model.rho=7 \
  network=autoencoder \
  gradient_clip_val=200 \
  model.rec_weight_mode=linear \
  model.kl_weight_mode=square \
  log_rec=True \
  'network.attn_resolutions=[16,8]' \
  deterministic=True \
  network.z_channels=16 \
  network.model_channels=128 \
  'network.channel_mult_enc=[2,2,4]' \
  'network.channel_mult_dec=[2,2,4]' \
  network.num_blocks_enc=4 \
  network.num_blocks_dec=4 \
  model.denoiser_loss_mode=l2 \
  dataset.out_channels=6 \
  model.loss_mode=huber \
  model.lambda_denoiser=0.1 \
  model.use_gan=False \
  model.lambda_latent_consistency=0.01 \
  model.ua_lmc_gamma=1 \
  model.ua_lmc_disable_dynamic_weight=false \
  2>&1 | tee "logs/direct-4gpu-$(date +%Y%m%d-%H%M%S).log"
```

只执行一次 `python main.py`。Lightning 会从这个主进程派生 4 个 DDP rank；不要手工启动四次 Python，也不要在非 Slurm 节点上添加 `srun` 或 `sbatch`。这里必须使用带引号的 `'devices=[0,1,2,3]'`，不要改成 `devices=4`。每张卡的 batch size 是 256，全局 batch size 是 1024，因此 CIFAR-10 每个 rank 每 epoch 约有 `50000 / 1024 = 48.83`，即 49 个 batch。

### 5.3 对照实验开关

关闭动态不确定性权重、保留普通 latent mean consistency（LMC）时，把完整命令最后的参数改为：

```bash
model.ua_lmc_disable_dynamic_weight=true
```

运行不含 latent consistency 的原始 CoVAE 对照时，设置：

```bash
model.lambda_latent_consistency=0
```

三组对比实验应保持 GPU 数、每卡 batch size、网络结构、训练步数、随机种子和其他 loss 参数一致。

### 5.4 检查是否确实为四卡训练

启动日志必须同时出现以下信息：

```text
Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/4
Initializing distributed: GLOBAL_RANK: 1, MEMBER: 2/4
Initializing distributed: GLOBAL_RANK: 2, MEMBER: 3/4
Initializing distributed: GLOBAL_RANK: 3, MEMBER: 4/4
distributed_backend=nccl
All distributed processes registered. Starting with 4 processes
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0,1,2,3]
LOCAL_RANK: 1 - CUDA_VISIBLE_DEVICES: [0,1,2,3]
LOCAL_RANK: 2 - CUDA_VISIBLE_DEVICES: [0,1,2,3]
LOCAL_RANK: 3 - CUDA_VISIBLE_DEVICES: [0,1,2,3]
```

另开一个 SSH 终端检查显卡和进程：

```bash
ssh gpu2
nvidia-smi
ps -ef | rg 'python .*main.py'
```

4 张卡都应有显存占用和持续的计算利用率。日志中的 `49` training batches 是正常值。Lightning 关于 `srun` 未使用、DDP grad stride、epoch metric 未设置 `sync_dist=True` 的提示不是单卡退化或训练退出；真正需要立即处理的是 `CUDA out of memory`、`NCCL error`、`Traceback`、进程被 `Killed`，或者日志只出现一个 rank。

当前训练默认 `compute_fid=False`，不会在训练中每 10000 步生成 50000 张图片计算完整 FID；`compute_rec_fid=True` 在这个前提下只触发较轻量的重建样本记录。建议保留每 10000 步的 checkpoint，训练后再用 `evaluate_fid.py` 统一评估各 checkpoint，避免训练期间的多卡重复 FID 计算显著拖慢训练。
