# Biomass #12 数据核对表（硬性要求）

论文：Nishimura, Voll, Krieger, Mitchell. *A Semi-Mechanistic Approach to Modeling Lipase-Catalyzed Processes with Multiple Competing Reactions*. Biomass 2026, 6, 12. doi:10.3390/biomass6010012

数据来源：MDPI 补充材料 `biomass-4043327-supplementary.pdf`（28 页），表 S1–S6、S17–S21 由 `code/parse_supp.py` 程序化解析为 `data/profiles.csv`（198 行，11 个反应曲线），没有手工录入任何数字。

原文没有直接写出样本量，但每张表都印了 Fobj、BIC 和参数个数 k；由于 BIC = nT·ln(Fobj/nT) + k·ln(nT)，**只有当公开数据的点数 nT 完全正确时，才能复算出原文印出的 BIC**。另外，指纹模型的残差平方和要求每一个数值都和原文用到的一致，才能复现到小数点后一位。

**结论：36/36 项硬性核对全部一致 → 通过。**（样本数、数据集/批次数、唯一时间点、无重复、缺失值模式与原文描述一致）

| 核对项 | 原文 | 公开数据 | 一致 | 说明 |
|---|---|---|---|---|
| Table 1 Akerman70 full: printed BIC recomputed with public nT=36 (Fobj=59.0, k=5) | 35.7 | 35.7 | YES | 9 sampling times x 4 species; BIC is only reproduced if nT is exact |
| Table 1 Akerman100 full: printed BIC recomputed with public nT=28 (Fobj=117.9, k=5) | 56.9 | 56.9 | YES | 7 sampling times x 4 species; BIC is only reproduced if nT is exact |
| Table 2 Akerman70 simplified: printed BIC recomputed with public nT=36 (Fobj=70.9, k=3) | 35.1 | 35.1 | YES | 9 sampling times x 4 species; BIC is only reproduced if nT is exact |
| Table 2 Akerman100 simplified: printed BIC recomputed with public nT=28 (Fobj=118.0, k=3) | 50.3 | 50.3 | YES | 7 sampling times x 4 species; BIC is only reproduced if nT is exact |
| Table 3 Bornadel full: printed BIC recomputed with public nT=120 (Fobj=329.9, k=5) | 145.3 | 145.3 | YES | 30 sampling times x 4 species; BIC is only reproduced if nT is exact |
| Table 3 Bornadel simplified: printed BIC recomputed with public nT=120 (Fobj=329.9, k=4) | 140.5 | 140.5 | YES | 30 sampling times x 4 species; BIC is only reproduced if nT is exact |
| Table 4 Tao025 full: printed BIC recomputed with public nT=64 (Fobj=121.2, k=5) | 61.7 | 61.7 | YES | 16 sampling times x 4 species; BIC is only reproduced if nT is exact |
| Table 4 Tao035+045 full: printed BIC recomputed with public nT=144 (Fobj=325.8, k=5) | 142.4 | 142.4 | YES | 18+18 sampling times x 4 species; BIC is only reproduced if nT is exact |
| Table 5 Tao025 simplified: printed BIC recomputed with public nT=64 (Fobj=123.5, k=3) | 54.5 | 54.5 | YES | 16 sampling times x 4 species; BIC is only reproduced if nT is exact |
| Table 5 Tao035+045 simplified: printed BIC recomputed with public nT=144 (Fobj=326.7, k=3) | 132.9 | 132.9 | YES | 18+18 sampling times x 4 species; BIC is only reproduced if nT is exact |
| Table 6 Mao1-3 full: printed BIC recomputed with public nT=141 (Fobj=1475.9, k=7) | 365.7 | 365.7 | YES | 15+16+16 sampling times x 3 species; BIC is only reproduced if nT is exact |
| Table 6 Mao1-3 simplified: printed BIC recomputed with public nT=141 (Fobj=1771.5, k=5) | 381.6 | 381.6 | YES | 15+16+16 sampling times x 3 species; BIC is only reproduced if nT is exact |
| Number of literature sources / case studies | 4 | 4 | YES | Akerman, Bornadel, Tao, Mao |
| Akerman datasets (70 and 100 C) | 2 | 2 | YES |  |
| Tao datasets (aw 0.25/0.35/0.45) | 3 | 3 | YES |  |
| Mao batches tabulated / used | 5 / 3 | 5 / 3 | YES | batches 4-5 excluded by authors (Fig S4) |
| Akerman70: rows / duplicated time stamps / missing (c,M,D,T) / missing Z | - | 10 / 0 / 0 / 0 | YES |  |
| Akerman70: max |Z+M+D+T-100| and max |M+2D+3T-c| | 0 (definition) | 0.0001 / 0.0001 | YES | mass balance of TMP and definition of c (Eq. 15) |
| Akerman100: rows / duplicated time stamps / missing (c,M,D,T) / missing Z | - | 8 / 0 / 0 / 0 | YES |  |
| Akerman100: max |Z+M+D+T-100| and max |M+2D+3T-c| | 0 (definition) | 0.0001 / 0.0002 | YES | mass balance of TMP and definition of c (Eq. 15) |
| Bornadel: rows / duplicated time stamps / missing (c,M,D,T) / missing Z | - | 31 / 0 / 0 / 0 | YES |  |
| Bornadel: max |Z+M+D+T-100| and max |M+2D+3T-c| | 0 (definition) | 3.0198 / 0.0002 | INFO | mass balance of TMP and definition of c (Eq. 15); Z was digitised independently in the Bornadel source (Mitchell et al. 2022) - these are the exact values the paper fitted (fingerprint SS 840.6 reproduced) |
| Tao025: rows / duplicated time stamps / missing (c,M,D,T) / missing Z | - | 17 / 0 / 0 / 0 | YES |  |
| Tao025: max |Z+M+D+T-100| and max |M+2D+3T-c| | 0 (definition) | 0.0001 / 0.0002 | YES | mass balance of TMP and definition of c (Eq. 15) |
| Tao035: rows / duplicated time stamps / missing (c,M,D,T) / missing Z | - | 19 / 0 / 0 / 0 | YES |  |
| Tao035: max |Z+M+D+T-100| and max |M+2D+3T-c| | 0 (definition) | 0.0001 / 0.0003 | YES | mass balance of TMP and definition of c (Eq. 15) |
| Tao045: rows / duplicated time stamps / missing (c,M,D,T) / missing Z | - | 19 / 0 / 0 / 0 | YES |  |
| Tao045: max |Z+M+D+T-100| and max |M+2D+3T-c| | 0 (definition) | 0.0001 / 0.0003 | YES | mass balance of TMP and definition of c (Eq. 15) |
| Mao1: rows / duplicated time stamps / missing (c,M,D,T) / missing Z | - | 16 / 0 / 0 / 16 | YES | Z not reported by Mao et al. (paper: 'did not allow calculation of Z') |
| Mao1: |M+2D+3T-c| (c from 3x acid conversion, independent measurement) | not required | median 2.7, max 17.0 | INFO | c derived from %conv A, species from mass %; differ because Z unknown and digitising noise |
| Mao2: rows / duplicated time stamps / missing (c,M,D,T) / missing Z | - | 17 / 0 / 0 / 17 | YES | Z not reported by Mao et al. (paper: 'did not allow calculation of Z') |
| Mao2: |M+2D+3T-c| (c from 3x acid conversion, independent measurement) | not required | median 9.7, max 20.8 | INFO | c derived from %conv A, species from mass %; differ because Z unknown and digitising noise |
| Mao3: rows / duplicated time stamps / missing (c,M,D,T) / missing Z | - | 17 / 0 / 0 / 17 | YES | Z not reported by Mao et al. (paper: 'did not allow calculation of Z') |
| Mao3: |M+2D+3T-c| (c from 3x acid conversion, independent measurement) | not required | median 6.0, max 9.8 | INFO | c derived from %conv A, species from mass %; differ because Z unknown and digitising noise |
| Mao4: rows / duplicated time stamps / missing (c,M,D,T) / missing Z | - | 20 / 0 / 0 / 20 | YES | Z not reported by Mao et al. (paper: 'did not allow calculation of Z') |
| Mao4: |M+2D+3T-c| (c from 3x acid conversion, independent measurement) | not required | median 14.5, max 55.5 | INFO | c derived from %conv A, species from mass %; differ because Z unknown and digitising noise |
| Mao5: rows / duplicated time stamps / missing (c,M,D,T) / missing Z | - | 24 / 0 / 0 / 24 | YES | Z not reported by Mao et al. (paper: 'did not allow calculation of Z') |
| Mao5: |M+2D+3T-c| (c from 3x acid conversion, independent measurement) | not required | median 18.2, max 28.8 | INFO | c derived from %conv A, species from mass %; differ because Z unknown and digitising noise |
| Fingerprinting fit Akerman: sum of squares reproduced on public data | 74.2 | 74.2 | YES | nT=64 (requires every row and value to match) |
| Fingerprinting fit Bornadel: sum of squares reproduced on public data | 840.6 | 840.6 | YES | nT=120 (requires every row and value to match) |
| Fingerprinting fit Tao: sum of squares reproduced on public data | 535.3 | 535.3 | YES | nT=208 (requires every row and value to match) |
| Fingerprinting fit Mao: sum of squares reproduced on public data | 1616.0 | 1616.0 | YES | nT=141 (requires every row and value to match) |

说明：
- “类别分布”在这里对应各数据集/批次的组成（4 个文献来源、Åkerman 2 个温度、Tao 3 个水活度、Mao 5 个批次中用 3 个），全部与原文一致。
- 缺失值：Mao 的 Z 全部缺失，原文明确说明 Mao 的数据无法计算 Z，并因此从目标函数中删掉了 Z 项 —— 与原文一致，不算偏差。
- 重复行：所有数据集中都没有重复的时间点。
- INFO 项（不影响通过）：Bornadel 的 Z 是从原始文献独立数字化得到的，Z+M+D+T 偏离 100 最多 3%；Mao 的 c 是由酸转化率×3 得到的，和 M+2D+3T 不完全相等。这两点都是原文所用数据本身的性质（指纹模型 SS 840.6 / 1616.0 被精确复现即可证明），不是公开数据和论文之间的不一致。
- 方法说明：酶浓度 [E]T 在原文里没有给出数值。kA 只以 kA·[E]T 的乘积形式出现，所以本文拟合 kE = kA·[E]T，再用 kE/kA 反推出酶载量：Åkerman 2.0 %w/w，Bornadel 5.0 %w/w，都是整数，说明一致。