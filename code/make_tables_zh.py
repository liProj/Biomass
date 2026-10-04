"""Chinese versions of paper/tables/*.tex -> paper/tables_zh/*.tex (captions by label, cell text by dictionary)."""
import os, re, glob
from kinetics import JD
SRC, DST = f"{JD}/paper/tables", f"{JD}/paper/tables_zh"; os.makedirs(DST, exist_ok=True)
CAP = {
  "tab:data": "用于模型分析的反应曲线，数据来自 Nishimura 等 \\cite{nishimura2026} 的补充材料。$n_t$ 为 $t=0$ 之后的采样时刻数。",
  "tab:datacheck": "公开数据与 Nishimura 等 \\cite{nishimura2026} 报告统计量的一致性核对，核对量涉及观测数、残差平方和及案例组成。",
  "tab:reprofp": "初始指纹模型的选择性估计与拟合统计量。各单元格依次列出 Nishimura 等 \\cite{nishimura2026} 的报告值和本文重新估计值。",
  "tab:reprodyn": "初始动态模型的数值核对。报告值来自 Nishimura 等 \\cite{nishimura2026}。“参数代入”表示固定文献动力学参数、优化标量酶载量 $E=k_E/k_A$ 后的目标值；“多起点”表示 30 个起点中找到的最小目标值。“一致”表示参数代入值与报告目标值的相对差异不超过 1.5\\%。",
  "tab:transfer": "组成路径的跨条件迁移误差。使用同一体系的其他曲线估计路径，在反应度域预测留出曲线，误差为 RMSE（摩尔\\%）。",
  "tab:loto": "留一时间点交叉验证的 RMSE（摩尔\\%）。全部估计步骤在每折训练集上重新执行。比值为 PPK-FIP 与基线的 RMSE 之比，O、F 分别表示 ORIG 和 ORIG-full。$p$ 为配对平方误差的 Wilcoxon 检验，经六个案例间 Holm 校正，合并行不作校正。",
  "tab:fcst": "前半段训练的向前外推 RMSE（摩尔\\%）。训练窗口为 $t\\le0.5t_\\mathrm{max}$，参数仅使用训练观测重新估计。比值与 $p$ 的定义同表~\\ref{tab:loto}。",
  "tab:lotobb": "机理模型与数据驱动基线的留一时间点交叉验证 RMSE（摩尔\\%）。",
  "tab:fcstbb": "机理模型与数据驱动基线的向前外推 RMSE（摩尔\\%）。",
  "tab:insample": "全数据拟合的残差平方和与信息准则。报告列来自 Nishimura 等 \\cite{nishimura2026}，依次为 $F_\\mathrm{obj}$ 与不计指纹选择性的 BIC。重新估计列与 PPK-FIP 列依次为 $F_\\mathrm{obj}$ 与计入全部参数的 BIC$_\\mathrm{tot}$；PPK-FIP 使用模型加权平均参数数目 $\\bar k$。",
  "tab:params": "PPK-FIP 在 BIC 权重较大模型形式下的参数估计与线性化 95\\% 置信半宽。$w_\\mathrm{simpl}$ 为简化形式权重。$k_E$ 达到上界 1000 提示当前数据下的参数约束不足，需结合图~\\ref{fig:landscape} 的剖面判断。f、s 分别表示完整与简化集总常数形式。",
  "tab:sim": "蒙特卡洛参数恢复结果，每种情景重复 60 次，以均值 $\\pm$ 标准差表示。曲线 RMSE 相对于稠密时间网格上的无噪声真值计算。",
  "tab:ablation": "PPK-FIP 组件消融的逐案例与合并 LOTO RMSE（摩尔\\%）。$p$ 为变体与完整 PPK-FIP 的配对平方误差 Wilcoxon 检验，最后一列为合并外推 RMSE。每个变体在每折训练集上重新估计。",
  "tab:pooling": "跨条件参数池化的 LOTO RMSE（摩尔\\%）。L1 共享组成路径，L1h 进一步对集总常数施加层次收缩（$\\tau=\\ln3$），L2 完全共享集总常数。留一案例元验证选择池化层级，最后一行比较元选择与独立估计 L0 的合并误差。",
  "tab:hybrid": "高斯过程残差校正前后的机理模型 RMSE（摩尔\\%）。高斯过程逐组分拟合训练残差，零均值偏差项在远离训练数据时衰减。"
}
REP = [  # (pattern, replacement) applied to the table body
 ("All cases (pooled)", "全部案例（合并）"), ("Mao batches 1--3", "Mao 批次 1--3"),
 ("Profile", "曲线"), (" & Table & ", " & 表 & "), ("System &", "体系 &"), ("Species", "组分"), ("Case &", "案例 &"),
 ("oleic", "油酸"), ("caprylic", "辛酸"), ("pelargonic", "壬酸"), ("batch 4 (excluded)", "批次 4（已排除）"), ("batch 5 (excluded)", "批次 5（已排除）"),
 ("batch ", "批次 "), ("M,D,T (+c)", "M,D,T（+c）"),
 ("Check & Paper & Public data & Match", "核对项 & 原文 & 公开数据 & 是否一致"), ("YES", "是"),
 ("printed BIC recomputed with public nT=", "用公开数据的 nT="), (" (Fobj=", " 复算原文 BIC（Fobj="), ("Fingerprinting fit ", "指纹拟合 "),
 (": sum of squares reproduced on public data", "：用公开数据复现残差平方和"), ("Number of literature sources / case studies", "文献来源 / 案例数"),
 ("Akerman datasets (70 and 100 C)", "Åkerman 数据集（70 和 100 °C）"), ("Tao datasets (aw 0.25/0.35/0.45)", "Tao 数据集（aw 0.25/0.35/0.45）"),
 ("Mao batches tabulated / used", "Mao 批次（列出 / 使用）"), (" full:", " 全模型："), (" simplified:", " 简化模型："),
 ("Model & $F_\\mathrm{obj}$ printed & at printed $\\theta$", "模型 & 原文 $F_\\mathrm{obj}$ & 原文参数处"), ("$F_\\mathrm{obj}$ global", "全局 $F_\\mathrm{obj}$"),
 ("BIC printed", "原文 BIC"), ("BIC global", "全局 BIC"), ("consistent", "一致"),
 (" & full & ", " & 全模型 & "), (" & simplified & ", " & 简化模型 & "), (" & yes \\\\", " & 是 \\\\"), (" & no \\\\", " & 否 \\\\"),
 ("Held-out profile & Original path & Processive path & Ratio", "留出曲线 & 原路径 & 持续性路径 & 比值"), ("\nAll &", "\n全部 &"),
 ("ratio$_\\mathrm{O}$", "比值$_\\mathrm{O}$"), ("ratio$_\\mathrm{F}$", "比值$_\\mathrm{F}$"),
 ("MLP ens.", "MLP 集成"), ("Neural ODE", "神经 ODE"),
 ("Paper simpl. $F_\\mathrm{obj}$/BIC", "原文简化 $F_\\mathrm{obj}$/BIC"), ("Paper full", "原文全模型"), ("ORIG refit", "ORIG 重拟合"), ("ORIG-full refit", "ORIG-full 重拟合"),
 ("$F$ set", "$F$ 集"),
 ("Scenario (truth) & Estimator & curve RMSE vs truth & bias", "情景（真值） & 估计方法 & 相对真值的曲线 RMSE & 偏差"), (" & bias ", " & 偏差 "),
 ("S1 plain", "S1 原模型"), ("S2 pseudo-processive", "S2 伪持续性"), ("S3 deactivation", "S3 失活"),
 ("Variant & ", "变体 & "), ("LOTO all", "LOTO 合并"), ("$p$ vs full", "对完整方法的 $p$"), ("Fcst all", "外推合并"),
 ("PPK-FIP (full)", "PPK-FIP（完整）"), ("simplified F only (no averaging)", "仅简化 F 集（无平均）"), ("full F only (no averaging)", "仅完整 F 集（无平均）"),
 ("no fingerprint prior (joint, BMA)", "无指纹先验（联合，BMA）"), ("no prior, BIC selection", "无先验，BIC 选择"),
 ("two-step, simplified (ORIG)", "两步法，简化（ORIG）"), ("two-step, full (ORIG-full)", "两步法，完整（ORIG-full）"),
 ("(no meta-selection)", "（无元选择）"), ("fixed $\\lambda$", "固定 $\\lambda$"), ("no pseudo-processivity", "无伪持续性"),
 ("no deactivation", "无失活"), ("no processivity, no deactivation", "无持续性、无失活"), ("PPK-FIP (re-run, consistency check)", "PPK-FIP（重跑，一致性检查）"),
 ("L1 path", "L1 路径"), ("L1h hier.", "L1h 层级"), ("L2 path+$F$", "L2 路径+$F$"), ("meta-choice", "元选择"),
 ("Protocol & Case", "协议 & 案例"), ("Extrapolation", "外推"),
]
for p in sorted(glob.glob(f"{SRC}/*.tex")):
    s = open(p, encoding="utf-8").read()
    lab = re.search(r"\\label\{([^}]+)\}", s).group(1)
    if lab in CAP:
        i = s.index("\\caption{"); depth = 0; j = i + len("\\caption")
        for k in range(j, len(s)):
            if s[k] == "{": depth += 1
            elif s[k] == "}":
                depth -= 1
                if depth == 0: end = k; break
        s = s[:i] + "\\caption{" + CAP[lab] + "}" + s[end + 1:]
    head, body = s.split("\\begin{tabular}", 1)
    for a, b in REP: body = body.replace(a, b)
    body = re.sub(r"Table (\d)", r"原文表 \1", body); body = re.sub(r", k=(\d+)\)", r"，k=\1）", body); body = body.replace("： ", "：")
    for old, new in [("原文参数处", "参数代入"), ("原文简化", "报告简化"), ("原文全模型", "报告全模型"), ("原文表", "文献表"), ("原文", "文献报告"), ("复现残差平方和", "核对残差平方和"), ("全局", "多起点"), ("原路径", "无伪持续性路径"), ("持续性路径", "伪持续性路径"), ("无伪伪持续性", "无伪持续性"), ("S1 原模型", "S1 初始模型"), ("Protocol", "协议"), ("：", "，")]:
        body = body.replace(old, new)
    open(f"{DST}/{os.path.basename(p)}", "w", encoding="utf-8").write(head + "\\begin{tabular}" + body)
print(len(glob.glob(f"{DST}/*.tex")), "tables translated")
