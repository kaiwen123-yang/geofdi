# GeoFDI Sprint 8 — M1 真机数据落地 + 遗留问题清仓 + 预言实验 + 公开数据收割 — Claude Code 执行指令

你是本项目执行工程师。仓库 `~/research/geofdi`（Sprint 7 完成：sim-milestone-5、hw-ready、theory-part1-v1/part2-v1.1）。本冲刺头条：**用户在 `H:\m1_data`（WSL 下 `/mnt/h/m1_data`）放了四组 M1 轮足真机数据**——这是项目的第一批自家真机数据，第一优先级是审计、入库、跑通、出第一批真机图。其余按块清仓遗留问题。

## 防丢失机制（先做）
本文件存为 `docs/sprints/sprint8_spec.md` 并 commit；`docs/sprints/sprint8_progress.md` 建门项清单，**每块结束更新并 push**；会话中断则新会话先读进度文件续跑。每块一个 review 包（outbox 下一个空闲编号）。铁律沿用：数据经 `GEOFDI_DATA_ROOT`、零 `/mnt/g` 与 `/mnt/h` 字面量进代码（仅本 spec 与审计文档可引）、确定性 seed、图 ≤1600px、包 <20 MB、未兑现如实报告不调参、R⁻ 检验零可训练参数、预注册先于运行。

## 前置检查
`/mnt/h/m1_data` 若不存在：**停下**，打印给用户的挂载命令 `sudo mkdir -p /mnt/h && sudo mount -t drvfs H: /mnt/h`（sudo 需密码，你无法执行），等用户挂载后重跑。H 盘视为**只读源**：所有处理先经 `ingest_session.sh` 拷入 `$GEOFDI_DATA_ROOT/data/raw/m1/`（校验和 + meta.yaml），绝不修改源。

---

## Block D — M1 真机数据（头条；commit 按 d1/d2/d3 分）

### D1 审计（`docs/protocol/m1_h_data_audit.md`）
1. 列出四组的目录结构、格式（db3 bag / mcap / CSV / SDK dump）、体积、疑似采集时间。
2. **读 bag 用 `pip install rosbags`（纯 Python，直接解析 sqlite3/mcap + CDR，不需要 ROS 环境）**——这同时绕开已知的 zenoh 录包毒字段问题（该问题只影响 replay，不影响直接解析；若 rosbags 解析 metadata.yaml 报错，先修 yaml 的 rmw 字段副本再读，源不动）。
3. 每组产出一张话题/字段表：话题名、类型、频率（实测）、时长、样本字段。**定案清单**（逐项写"已定/待厂商/无此通道"）：关节数是否 16、`names` 字段原文（关节顺序真值！）、q/dq/efforts 有无与单位量级（efforts 的数值分布贴电流特征还是扭矩特征——只记观察不下结论）、轮编码器角度/速度与是否解缠、IMU 话题/频率/坐标系、cmd 有无、接触量有无、电机温度、定位输出话题（SLocalization 位姿——**若在，它就是参考真值**）、时间戳单调性与跨话题偏差。
4. 用 `names` 对照 `io/m1_mapping.yaml` 的候选映射（fl1..fl4→ABAD/HIP/KNEE/WHEEL）：吻合则 `unverified→false` 并 commit；不吻合则改映射并记录差异。
5. 每组给判定：能撑哪些实验（滚动 H₀′ / 滚动 InEKF / DeLaN 训练 / 仅存档），理由一句。

### D2 入库与直行段
- 合格组经 `ingest_session.sh` 入 `raw/m1/nominal/`（或 audit/），meta.yaml 按审计结果填。
- 直行段切分：有 cmd 用 cmd；无 cmd 用回退规则——IMU 偏航率 |ω_z|<阈 且 左右轮速差<阈 的连续段（阈值从数据分布取，记录）。

### D3 首批真机实验（**预注册先 commit**：`docs/protocol/m1_real_preregistration.md`——预期：朴素 H₀ 可能越界〔块相关 + 真实 ε_dyn，W 块已预言〕；H₀′ 差分在带内；单段内无报警）
1. `run_pipeline.sh <session> --robot m1 --mode rolling`：每组一份 report.md——**第一张真机 R⁻ H₀′ 图**从这里出（QQ、逐窗 p、e-过程轨迹）。块长 L 用 protocol 里的边界值起步，实测块相关后调整并记录。
2. 若 q/dq/轮速 + IMU 齐：**滚动 InEKF vs 固定足 RIEKF vs ESKF 在真机数据上重跑**（e10 的 0.71 m vs 13.4 m 那张图的真机版）；参考位姿优先用包内 SLocalization 话题，无则报相对指标（回环闭合差/直行段直线度）。
3. 若数据量够（≥20 min 名义滚动）：训练 M1 真机等变 DeLaN 一版（16 关节模板），残差 R⁻ 与 model-free R⁻ 并列一张表。
4. efforts 语义观察、真实 ε_dyn 候选（镜像残差稳定值）、H₀′ 校准段 ν₀ 的量级——全部写进审计文档，供理论 intake。
**D 门**：审计文档四组齐；映射定案；≥1 组跑通 pipeline 出 report；预注册先于运行；真机 H₀′ 图出。

---

## Block L — 遗留小修清仓（commit: `chore: leftover fixes`）
1. `make_review_pack.sh`：块内 MANIFEST 覆盖顶层模板（rp020–024 的 bug），并回补 rp020–024 的顶层 MANIFEST（从 code/ 拷正）。
2. Day-0 文档与 `run_pipeline.sh` 的 report 解读段加注：滚动模式主检验为 H₀′，朴素 H₀ 越界为预期行为（W 块发现）。
3. weld 世界补跑 e13a/e13b（功效 + δ_f 污染，R=50 即可）——RA-L 拆分素材，结果进 `docs/paper/split_option.md`。
4. e03 审计文档补注：镜像双侧外部检验 n=8 不定论，干净版待自采直行语料。
5. `docs/sprints/` 里登记：N2 定理正文化（稀疏校正一致性 + 签名=可观投影）为下一个理论冲刺项，本冲刺不做。

## Block T2 — N1-2 二层重写（commit: `feat(theory): N1-2 two-layer restatement`；tag `theory-part1-v1.1`）
按对抗审计结论重写 `02_n1_theorems.tex` 的 N1-2：
- **(I) 律级盲性二分**：Σ-固定故障 **且故障后对称吸引子唯一性（A5-under-fault）保持** ⟹ 故障后律 Σ-不变 ⟹ 一切水平-α 不变性检验功效 ≤ α；反之任何律级不变性破缺对一致统计量（能量距离类）可检。Remark：盲性有幅值天花板——严重双侧故障可经手性分岔变为可见（前向引用 P1）。
- **(II) 均值级功效刻画**：小均值响应故障非中心度 K‖Π⁻μ‖²、单腿足迹份额恰 ½（原证明保留）。
- **(III) 统计量一致性 Remark**：配对均值型对零均值律差异（如单侧噪声膨胀）盲，能量距离型可见（前向引用 P2）——部署应并用两型。
- 更新证伪条件与实证锚占位（P1/P2 跑完回填）。`make theory` 零 error。

## Block P — 三个预注册预言实验（`experiments/e15_predictions/`；预注册 `docs/protocol/e15_preregistration.md` **先 commit**）
- **P1 手性天花板**（stage a）：go2_urdf_sym 闭环，双侧镜像等 κ ∈ {0.7, 0.5, 0.4, 0.3}，R=30：R⁻ 功效 + 手性指标（占空比镜像残差的稳定非零性）。预言：κ=0.7 盲（≈α，已证）；某个更重的 κ 起手性分岔出现、R⁻ 转为可见。产出：功效与手性指标 vs κ 一张图，给出天花板估计。
- **P2 统计量分家**（stage b）：左腿编码器噪声方差 ×4（零均值），R=50：paired_energy 功效预言 ≈α，energy_distance 预言 >α。产出：两统计量功效对比 + 部署建议写进 protocol。
- **P3 打滑三 regime**（stage c）：Go2 trot——左侧足迹带低摩擦贴片（单侧持续滑）预言 R⁻ 响；全场均匀低 μ（双侧滑）预言 R⁻ 静 + InEKF innovation/NIS 响；M1 滚动——单轮 μ×0.5 vs 双轮 μ×0.5 同构预言。R=30。产出：regime × 双通道读数表——**盲性定理作为打滑分类器**的仿真实证。
**P 门**：预注册时间戳先于运行；三张图/表；结果回填 T2 的实证锚并重编译。

## Block G — 滑统计件 + πᵢ 门控（`detect/stance_event.py`、`estimate/pi_gating.py`；`experiments/e16_pi_gating/`）
1. **逐站立事件统计**：站立相条件化的逐事件 conformal p（每次触地一个检验窗，校准集 = 名义触地事件库），FAR 按事件计；e-值聚合为逐腿时变权重。轮足版：逐轮滚动约束残差（‖ḋ̂−R̂u‖ 的白化量）同构。
2. **πᵢ 门控**：逐腿 e-过程 → 权重 w_i（硬门 e≥1/α 剔除，软门协方差按 e 缩放，两版都实现）→ InEKF 量测更新。
3. **e16 三估计器对比**（P3 的单侧滑场景 + 名义）：无门控 / 文献标准（足速 0.4 m/s 阈值 + 协方差 ×10）/ GeoFDI-πᵢ（硬、软两版）：位姿 RMSE、NEES、门控触发时间线。预期 GeoFDI 在名义段零误剔（FAR 保证）而阈值法有误剔——如实报。
4. `docs/protocol/hw_slip_protocol.md`：实机打滑实验协议——材料清单（亚克力/PTFE 板、特氟龙胶带）、Go2 室外 Vision-RTK2 真值配置、M1 室内 SLocalization 参考、三段式会话设计（名义/单侧/双侧）、预注册模板。
**G 门**：逐事件 FAR 在名义仿真上 ≈α；三估计器表 + 时间线图；协议文档。

## Block PUB — 公开数据收割（`experiments/e17_public_realdata/`；预注册 `docs/protocol/e17_preregistration.md` 先 commit：预期真机朴素 H₀ 可能越界、H₀′ 在带、Mini Cheetah 跨地形 FAR 稳定）
1. **Mini Cheetah 接触数据集**：从 UMich-CURLY/deep-contact-estimator 的下载链接取（可开代理），入库 `raw/public/minicheetah-contact/`（记录 URL、许可、哈希）。字段：q/dq/τ_est/IMU/足位速/接触标签，1000 Hz 对齐，8 地形 3 步态 + 悬空序列。
2. Mini Cheetah 实验：**跨 8 地形的 R⁻ H₀′ FAR 表**（A3 的真机实证，主打表）；**真机残差 R⁻**（有 τ_est，解析模型用 MIT Cheetah URDF 或退化为 model-free+τ 通道，选择并记录）；悬空序列 = 真机腿-臂名义数据（weld 管线跑一遍）；可选 K₄ 演示（前后对称组，若时间允许）。
3. **Street A1 / legkilo Go1**：直行稳态段挖掘（相位估计器 + 偏航率筛）→ 真机 R⁻ H₀′ FAR 各一张；legkilo 上 **三估计器门控对比的真机版**（参考位姿：数据集自带者优先，无则 `pip install kiss-icp` 跑激光里程计作参考，记录）。
4. `docs/data_catalog.md` 与论文 outline 的证据栈段更新：五层证据结构成文。
**PUB 门**：Mini Cheetah 入库 + 地形 FAR 表 + 真机残差 R⁻；Street/legkilo 各一张真机 H₀′ 图；legkilo 门控对比表。

---

## 执行顺序与收尾
顺序：spec/progress → **D**（头条）→ L → T2 → P → G → PUB。每块 review 包 + 进度更新 + push；全部完成打 `real-data-milestone-1`。最后输出：真机数据审计摘要（四组各一句判定）、第一批真机图清单、遗留问题清仓状态表、以及"下一步唯一还缺的东西"清单（预期只剩：受控打滑/配重注入、Gate 4、efforts 厂商确认、N2 定理正文化、统稿五件套）。
