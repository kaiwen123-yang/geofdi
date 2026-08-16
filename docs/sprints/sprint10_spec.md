# GeoFDI Sprint 10 — 全稿 v0 + 收尾三实验 + 机载代理基准 — Claude Code 执行指令

你是本项目执行工程师。仓库 `~/research/geofdi`（Sprint 9 完成：real-data-milestone-2、theory-part3-v1、理论零 TODO、65 测试通过）。本冲刺三块：**M0 收尾实验与元数据定案 → W 全稿 v0（主体）→ N 机载代理基准（今日笔记本执行 + 明日 NUC 文档）**。工作名定为 **GeoFDI**（冻结前保留改名窗口）。

## 防丢失机制
本文件存 `docs/sprints/sprint10_spec.md` 并 commit；`docs/sprints/sprint10_progress.md` 门项清单每块更新推送。每块一个 review 包。铁律沿用：确定性 seed、图 ≤1600px、包 <20 MB、预注册先于运行、未兑现如实报告、代码零盘符字面量。

---

## Block M0 — 元数据定案与三个收尾实验（commit 按项分）

1. **场地标签定案**（用户确认）：xb=场地A（室内外衔接、半粗糙/半光滑瓷砖）、nmb=场地B（软劣化、较光滑）、by=场地C（粗糙）。`go2_session_meta_form.md` 的 [inferred] 全部翻 [confirmed by operator 2026-08-16]；R2 表按确认配对重发一版（数值不变则只改标注，变了如实报）。
2. **xb4/nmb3 切分实验**（e20 addendum；预注册先 commit：xb4 的遍历内报警源于场地A的表面切换线——若室内半/室外半各自检验均回带内而跨半段拼接报警，则 P-A 修正为"表面切换是跨段条件变化而非会话内 nuisance"；若单半段内仍报警，则为真遍历内异常，进硬件节残余异常段）：用 RTK 位置把 xb4 切为室内半/室外半，各自跑 H₀′；nmb3 同法按前/后半切。产出：2×2 切分结果表 + 结论一句。
3. **B1 摩擦补行**（e21 addendum）：残差 R⁻（解析动量观测器版）补跑 e21 的摩擦故障格（friction_scale ×{1.5,2,3}，R=20），并入基线总表——闭合"经典法赢摩擦"的解释（e13a 已证摩擦需残差通道）。claims 表数据就绪。
4. **进度文件登记**：M1 网盘跨天数据入库位 `raw/m1/nominal-crossday/` 待用户下载；两封邮件挂起（用户指示）。

**M0 门**：标签定案 commit；切分表 + 判定；摩擦行并表。

---

## Block W — 全稿 v0（tag `paper-draft-v0`）

### W1 论文工程
- `paper/` 目录：IEEEtran 双栏（`tlmgr install ieeetran` 或从 CTAN 取 IEEEtran.cls 入 `paper/` 并记录来源版本）；`main.tex` + `sections/*.tex` + `appendix/*.tex`；bib 合并自 `theory/references.bib`（加论文侧新引：Camurri 概率接触、Jenelten 滑地运动、深度接触估计器、Koning 在线置换、Cully Nature 2015——**全部按既有 verified 纪律核对后入库，核不到留可见 todo**）；`latexmk -pdf` 进 Makefile 目标 `make paper`。
- 作者占位：Kaiwen Yang + [advisor TBD] + [affiliation]；单盲，无需匿名。

### W2 五件套（每件单独 commit）
1. **标题**（工作版，冻结前可改）：主推 "GeoFDI: Fault and Slip Detection as Invariance Testing, with Finite-Sample Guarantees for Legged and Wheeled-Legged Robots"；备选两条列在 `paper/notes_title.md`。
2. **Fig. 1 概念图**（TikZ）：左列三种结构不变性（形态对称 C₂ 图示、步态时空对称 Σ、接触几何/滚动约束）→ 中列"破缺即故障/打滑"→ 右列三通道读数（R⁻/关节残差行/基座行 + InEKF innovation）与交付物（精确 FAR、anytime-valid、可检测域刻画）。单栏宽，一图讲完主命题。
3. **Algorithm 1（GeoFDI 检验器）**：八行流程，右侧边注**九条正确性条件**（原八条 + 新增第九条：定侧必须用带符号统计量——能量类结构上只排镜像对；每条标定理号或实验号）。用 algorithmic 环境 + 边注表。
4. **Introduction 完整初稿**：问题（接触与执行完整性，现有方法零保证——引 e03 的 52% 健康段误报与 B1 的 0.15/0.50 名义误报作现状证据）；空档（三社区查新结论一段）；主命题一段；**四条贡献**（①不变性作为原假设 + 精确 FAR 群随机化检验与 anytime-valid 序贯版〔N1-1, N1-3, N1-5〕；②可见域二层刻画与三通道架构〔N1-2 二层版, N3-3〕；③等变物理一致名义模型：必要性与 Π⁺-only 误差〔N3-1/2/3〕；④滚动接触精确保持 InEKF 结构〔Part 3〕+ 五层证据栈含双自家平台〕；结果预告用实数（11/11、0.049≈α、40.6% vs 1.9%、0.99 vs 0.03、两月 ν₀ ×1.18）。
5. **Claims 表**（Table I）：{GeoFDI-R⁻, GeoFDI-残差R⁻, 动量观测器+χ², Mahalanobis, AE, GRU} × {需故障数据, 需训练, 需动力学模型, 有限样本FAR, nuisance免疫, 序贯保证, 隔离粒度, 摩擦类可检}，每格标数据来源实验号。

### W3 正文组装（按 `docs/paper/outline.md`，逐节 commit）
- **Preliminaries**：从 Part 0 压缩（表示、Σ、数据元、H₀/H₀′）；**Theory**：定理陈述+直觉+实证锚，全部证明进 appendix（Part 0–3 的证明搬运，编号映射表写进 appendix 开头）；**Method**：三通道架构图、序贯层、**逐遍历校准规则**（R5 的部署规则单列小节，含 M1 173247 与 Go2 8/11 的统一机制叙述）、等变 DeLaN、πᵢ 门控；**Simulation**：图厂主图（可交换性、isotypic、nuisance 表、DK 曲线、低 SNR 最小可检表）；**External benchmark**：e03 四类表 + Mini Cheetah 八地形 + Leg-KILO/Street；**Hardware**：M1 三会话（H₀′ 图、滚动 InEKF 0.99 vs 0.03）+ Go2 十一会话（R1 图、R2 跨月表、LH−RH/LF−RF 对角加载发现含足惯板排除、xb4 切分结论、R4 的 40.6% vs 1.9%）+ **注入占位框**：可见的 framed box 逐项列明待填数据（单/双侧铺板、单侧配重、负载对角旋转、≥60s 连续段、LowState 录制解锁项、Gate 4）——实机日与占位框一一对应；**Limitations**（自列：R⁻ 只见反对称分量、镜像双侧盲由 R⁺ 补、激进步态相位配准限制〔B6〕、H₀′ 校准不可跨遍历移植、场地A表面切换的定性、model-free 版无摩擦可检性、机载数字为代理）；**Conclusion + 博士叙事一句**（πᵢ 前端）。
- 全文编译零 error；页数如实报告（超长不砍，v0 只求完整）。
**W 门**：`make paper` 出完整 PDF；五件套齐；占位框与实机日清单一一对应；进度文件记录页数与缺口清单。

---

## Block N — 机载代理基准（今日笔记本执行；NUC 文档留明日）

1. `scripts/bench_pipeline.py`：对一个已入库会话逐阶段计时（解析装载 → 直行段/相位 → 数据元构造 → 置换检验 M=512 → e-过程 → H₀′ → 可选 InEKF 传播+更新），单核锁定（`taskset -c 0`、BLAS 线程=1），报每阶段毫秒/周期、端到端每周期延迟、可持续 Hz vs 250 Hz 遥测率；输出 `results/bench/<hostname>/bench_table.csv` + markdown 表。
2. **今日执行**：本机（Legion Y9000P）上对一个 Go2 会话与一个 M1 会话各跑一遍，表进 review 包；论文 Limitations/实现节引用为"笔记本级 CPU 代理数字"。
3. **NUC 文档** `docs/protocol/bench_nuc.md`：NUC13（Ubuntu 22.04，无 GPU）的三步指引——clone + `make setup`（无 CUDA 分支）+ 用户 scp 一个会话目录过去 + `python scripts/bench_pipeline.py <session>`；机器配置 `env/machines/nuc13.yaml` 建好（data_root 指向本地目录占位）。明日用户自跑，结果表回填同一 CSV 格式。
**N 门**：笔记本两表出；NUC 文档 + 配置就绪。

---

## 收尾
三个 review 包；push；W 门后 tag `paper-draft-v0`。最后输出：全稿页数与目录、占位框清单（= 实机日购物清单的镜像）、bench 笔记本表、以及"冻结前仍开放的决定"清单（标题终选、拆分与否、advisor 署名）。
