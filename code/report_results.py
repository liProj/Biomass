# executed inside make_report.py (namespace: doc, H, Pp, table, fig, R, ex, pd, np, json)
S = f"{R}/summary"
H("4. 新方法：PPK-FIP")
Pp("PPK-FIP（pseudo-processive kinetics with a fingerprint-informed prior，伪持续性动力学 + 指纹信息先验）保留原文的 Ping Pong bi bi 机理核心，做了四点改进：")
for t in ["(1) 机理扩展：加入伪持续性（pM、pD：新生成的单酯/二酯在离开活性位点之前被再次酰化的概率，对应传质限制）和批内一阶酶失活（kd）。pM=pD=0、kd=0 时严格退化为原文模型。",
          "(2) 精确时间重参数化：所有反应共享同一个 Ping Pong 分母，所以组成只取决于反应度 c，时间由 5 个路径积分的线性组合给出：kE·τ(c) = J0 + FA·J1 + FAZ·JZ + FAM·JM + FAD·JD，失活通过 τ=(1−e^(−kd·t))/kd 处理。路径用 numba 编译的 RK4 只积分一次，改变 kE、F 和 kd 时每次求值不到 1 ms，误差 < 6e-5 摩尔百分比。正是这一点让全局多起点优化、145 折交叉验证和蒙特卡洛模拟变得可行。",
          "(3) 指纹信息先验：原文两步法相当于给 SM、SD 一个无限紧的先验（固定），联合估计相当于平坦先验。PPK-FIP 使用中间的、校准过的先验：ln SX ~ N(ln 指纹估计, (λ·相对置信半宽/1.96)²)；pM、pD 用半正态收缩先验（尺度 s_p），kd 用弱先验。用 MAP 估计，求解时把先验残差附加到最小二乘残差向量后面。",
          "(4) 模型平均与元验证：简化 F 集和全 F 集两个版本按 BIC 权重做平均；超参数 (λ, s_p) ∈ {1,3}×{0.05,0.2} 由“留一案例”元验证选择，即目标案例只使用其余 5 个案例的交叉验证结果来选超参数，完全不使用目标案例的数据。最终 6 个案例都选中了 λ=3。"]:
    Pp(t)
fig("fig01_workflow", "图 4-1　原文两步法与 PPK-FIP 的流程对比")

H("5. 实验结果")
H("5.1 全数据拟合（原文协议）", 2)
if ex(f"{R}/final/insample.csv"):
    I = pd.read_csv(f"{R}/final/insample.csv")
    table(I[I.method.isin(["ORIG", "ORIG-full", "PPK-FIP"])][["case", "method", "Fobj", "nT", "k", "BIC_total"]], "表 5-1　全数据拟合：Fobj 与 BIC_tot（计入所有估计参数）", fmt=2)
    Pp("原文印出的最优 Fobj：Åkerman 70 °C 59.0、100 °C 117.9、Bornadel 329.9、Tao 0.25 121.2、Tao 0.35/0.45 325.8、Mao 1475.9。PPK-FIP 在 6 个拟合中 Fobj 全部更低（降低 3.7%–73%）；BIC_tot 有 5/6 优于原方法的重拟合结果。"
       "唯一例外是 Åkerman 70 °C：其持续性和失活参数虽然被收缩到接近 0，但参数个数照样计入 BIC，因此 BIC 略差。")
for n, c in (("fig06_fits_time", "图 5-1　全数据拟合曲线（虚线：原方法重拟合；实线：PPK-FIP）"), ("fig07_paths_c", "图 5-2　c 域组成路径与估计的持续性参数"),
             ("fig08_insample", "图 5-3　样本内 Fobj 与 ΔBIC_tot"), ("fig09_residuals", "图 5-4　残差诊断（Bornadel 早期 M、D 的系统性残差被消除）")):
    fig(n, c)
if ex(f"{JD}/paper/tables/params.tex"):
    Pp("PPK-FIP 参数估计（全数据）见论文表 params。Bornadel 数据强烈支持伪持续性（pM≈0.18）；Åkerman 100 °C 支持批内失活（kd≈0.018 h⁻¹）。")

H("5.2 留一时间点交叉验证（主要的样本外评价）", 2)
if ex(f"{S}/loto_tests.csv"):
    Pp("每一折删掉一个采样时间点上测到的全部物种，所有方法（包括原文两步法的两个步骤）都在剩余数据上从头重新估计，再预测被删掉的值。共 145 折、533 个观测。")
    table(pd.read_csv(f"{S}/loto_rmse.csv"), "表 5-2　LOTO RMSE（摩尔百分比），各方法 × 各案例")
    table(pd.read_csv(f"{S}/loto_tests.csv")[["ref", "case", "rmse_new", "rmse_ref", "ratio", "obs_win_frac", "median_diff", "p", "p_holm"]],
          "表 5-3　PPK-FIP 对原方法 / 原全模型的配对 Wilcoxon 检验（Holm 校正）", fmt=4)
    T = pd.read_csv(f"{S}/loto_tests.csv")
    a1 = T[(T.ref == "ORIG") & (T.case == "ALL")].iloc[0]; a2 = T[(T.ref == "ORIG-full") & (T.case == "ALL")].iloc[0]
    Pp(f"结论：总体 RMSE 从 {a1.rmse_ref:.3f}（原方法）降到 {a1.rmse_new:.3f}（−{100*(1-a1.ratio):.1f}%，p={a1.p:.1e}）；相对原全模型（{a2.rmse_ref:.3f}）降低 {100*(1-a2.ratio):.1f}%（p={a2.p:.1e}）。"
       "逐案例看：对原方法 4 个显著更好、2 个不显著（两个 Åkerman 数据集），没有显著更差的；对原全模型 3 个显著更好，Åkerman 70 °C 显著更差（这是采样点最少的曲线）。")
    fig("fig10_loto", "图 5-5　LOTO 交叉验证：各案例 RMSE 与比值（星号颜色表示配对中位差的方向）")
H("5.3 向前外推（压力测试）", 2)
if ex(f"{S}/fcst_tests.csv"):
    table(pd.read_csv(f"{S}/fcst_rmse.csv"), "表 5-4　外推 RMSE（用 t ≤ 0.5 t_max 训练，预测剩余部分）")
    Pp("外推对所有方法都很难，而且每个案例只有一个划分，统计功效很低。PPK-FIP 把 Åkerman 100 °C 的外推误差减半，在 Tao 0.25 上更好；但在 Åkerman 70 °C 和 Mao 上比原方法差。总体差异不显著。如实报告，不作为优势的证据。")
    fig("fig11_forecast_bars", "图 5-6　外推 RMSE 与比值"); fig("fig12_forecast", "图 5-7　外推曲线示例（阴影为训练区间）")
H("5.4 跨条件迁移、模拟研究", 2)
fig("fig16_transfer", "图 5-8　跨条件迁移：用同体系其他条件学到的组成路径预测留出条件")
if ex(f"{R}/sim/sim.csv"):
    s = pd.read_csv(f"{R}/sim/sim.csv")
    table(s.groupby(["scenario", "method"])[["curve_rmse", "SM_err", "SD_err"]].mean().reset_index(), "表 5-5　蒙特卡洛模拟（每种情景 60 次）：曲线误差与选择性对数偏差的均值")
    Pp("S1（真值就是原模型）：PPK-FIP 与原方法一样好，不会凭空制造持续性；不加先验的联合估计则会过拟合。S2（伪持续性真值）：原方法的 SM 高估约 76%，曲线误差 1.21；PPK-FIP 几乎无偏，误差 0.26。S3（批内失活）：曲线误差从 1.74（原方法）和 0.94（原全模型）降到 0.72。")
    fig("fig15_simulation", "图 5-9　模拟研究")
H("5.5 与黑箱机器学习模型的对比", 2)
Pp("对比模型：高斯过程（Matérn-5/2 + 白噪声）、MLP 集成（5 个种子）、预训练表格基础模型 TabICL（v2 回归器，上下文学习）、Neural ODE（RK4 求解器训练）。输入为时间和条件描述量（水活度或批次），并给出已知的初始状态。")
if ex(f"{S}/loto_rmse.csv"):
    rm = pd.read_csv(f"{S}/loto_rmse.csv", index_col=0)
    table(rm[[c for c in ["ORIG", "PPK-FIP", "GP", "MLP-ens", "TabICL", "NODE"] if c in rm.columns]].reset_index(), "表 5-6　LOTO：机理模型 vs 黑箱模型")
Pp("诚实结论：在采样密集的曲线上做内插时，Neural ODE 和 GP 可以比机理模型更好（尤其是 Mao，它的数字化数据和完全转化并不一致）；但在稀疏的 Åkerman 曲线上，以及在全部外推任务中，黑箱模型都失败了（外推 RMSE 20–41），因为它们没有编码 TMP 守恒和顺序 Ping Pong 结构。")
fig("fig13_blackbox", "图 5-10　机理模型与黑箱基线的对比（对数坐标）")
H("5.6 消融实验", 2)
if ex(f"{S}/loto_ablation.csv"):
    table(pd.read_csv(f"{S}/loto_ablation.csv"), "表 5-7　消融（LOTO RMSE；p 为与完整 PPK-FIP 的配对 Wilcoxon 检验）")
    Pp("主要结论：(a) 两步法（原文）在 Bornadel、Tao 0.25 上明显更差；(b) 去掉先验（联合估计）会让稀疏的 Åkerman 曲线变差，说明指纹信息应该保留，但要作为先验而不是约束；(c) λ=1 的窄先验（接近两步法）比 λ=3 差，元验证在 6 个案例中都选中了 λ=3；"
       "(d) 只用全 F 集与模型平均的结果相近（2.109 vs 2.121），说明模型平均不是必需的组件，如实报告；(e) 去掉伪持续性后 RMSE 升到 2.289（Bornadel 从 1.05 升到 1.72），说明伪持续性是样本外增益的主要来源；(f) 去掉失活项后 LOTO 为 2.115，并没有变差，外推时甚至更好（主要是 Mao）。所以失活项只改善了 Åkerman 100 °C 的样本内拟合，没有得到样本外证据的支持。这一点如实报告；没有根据测试结果事后删掉它。")
    fig("fig14_ablation", "图 5-11　消融实验")
H("5.7 扩展实验：多任务池化与 GP 偏差项混合模型", 2)
fig("fig17_pooling", "图 5-12　多任务池化（同一体系的不同条件共享路径 / F 常数）")
Pp("池化在 Åkerman 100 °C、Tao 两组上明显降低误差（例如 Åkerman 100 °C：3.51 → 2.20），但在 Åkerman 70 °C 上更差；用留一案例元验证选择池化层级后，总体 RMSE 2.117，与不池化的 2.121 基本相同，没有净收益，因此 PPK-FIP 最终不采用池化，作为扩展实验报告。")
if ex(f"{R}/cv/loto_hybrid.csv.gz"):
    hb = pd.read_csv(f"{R}/cv/loto_hybrid.csv.gz"); hb["se"] = hb.err ** 2
    table(hb.groupby(["case", "procedure"]).se.mean().pow(.5).unstack().reset_index(), "表 5-8　PPK-FIP + GP 偏差项（Kennedy–O'Hagan 型混合）的 LOTO RMSE")
    if ex(f"{R}/cv/fcst_hybrid.csv.gz"):
        hf = pd.read_csv(f"{R}/cv/fcst_hybrid.csv.gz"); hf["se"] = hf.err ** 2
        table(hf.groupby(["case", "procedure"]).se.mean().pow(.5).unstack().reset_index(), "表 5-9　混合模型的外推 RMSE（GP 远离数据时回归到 0，自动退回机理模型）")
    fig("fig18_hybrid", "图 5-13　混合模型：机理均值 + GP 偏差项")
    Pp("结论：PPK-FIP-GP 的 LOTO 总 RMSE 为 1.290，比原方法（2.533）低 49%（p<0.001，145 折中 126 折更好），也低于最强的黑箱模型 Neural ODE（1.791）；外推时与纯机理的 PPK-FIP 一样（11.86），远好于所有黑箱模型（20–41）。"
       "需要如实说明：给原方法加上同样的 GP 偏差项，内插误差也能降到 1.307，与 PPK-FIP-GP 没有显著差异（p=0.89）。也就是说，内插上的大部分增益来自 GP 偏差项这一通用手段；而 PPK-FIP 的机理改进对选择性估计、外推和可解释性才是关键。"
       "建议：参数估计用 PPK-FIP；需要在实验时间范围内做最准确的预测时用 PPK-FIP-GP。")
H("6. 诊断与可复现性")
Pp("求解器基准：精确重参数化（路径缓存）每次求值约 0.8 ms，LSODA 约 10 ms，Radau 约 1.4 s。原文全模型：200 个随机起点只有 46%–94% 能到达全局最优，FA 方向的剖面目标函数单调下降到边界（结构不可辨识）。所有代码在 Biomass/code/，一键重建脚本见 finalize.sh。")
H("7. 结论")
Pp("(1) 原文的数据完整公开，复现核对 36/36 项全部一致；但原文部分全模型拟合没有到达最优、全模型不可辨识、Tao 案例无法复现。"
   "(2) PPK-FIP 在原文自己的指标（Fobj）上 6/6 更好；在严格的样本外 LOTO 交叉验证上，总体误差比原方法低 16.2%、比原全模型低 8.2%（均 p<0.001）。"
   "(3) 外推和 Åkerman 70 °C 是没有超过原文的地方，已如实报告并分析了原因（采样点太少，以及外推只有单次划分）。"
   "(4) 论文意义：对于用生物质来源的脂肪酸和 TMP 生产生物润滑剂的过程设计，在传质受限反应器里得到的“表观选择性”不能不加持续性校正就直接迁移到别的反应器；动力学参数应当配合样本外验证来报告，而不能只看样本内拟合统计量。")
H("8. 输出文件位置")
for t in ["~/Documents/mdpi_papers/Biomass/screening.csv —— 80 篇论文的筛选评估", "~/Documents/mdpi_papers/Biomass/data_check.md —— 数据核对表",
          "~/Documents/mdpi_papers/Biomass/code/ —— 全部代码", "~/Documents/mdpi_papers/Biomass/results/ —— 全部结果文件",
          "~/Documents/mdpi_papers/Biomass/figures/ —— 全部图（PNG + PDF）", "~/Documents/mdpi_papers/Biomass/paper/main.pdf —— 英文 LaTeX 论文", "~/Documents/mdpi_papers/Biomass/paper/main_zh.pdf —— 中文版论文",
          "~/Documents/mdpi_papers/Biomass/report.docx —— 本报告"]:
    Pp("• " + t)
