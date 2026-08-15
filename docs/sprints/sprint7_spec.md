# GeoFDI Sprint 7（总纲）— 轮足/点足双实机就绪 + e03 外部基准 + 低 SNR + 三通道 + N2 + 理论 Part 1 全文 + 加量项 — Claude Code 执行指令

你是本项目执行工程师。仓库 `~/research/geofdi`（当前：Sprint 6 完成，tag theory-part2-v1 与 sim-milestone-4；**Sprint 5 v2 从未执行**，其内容已全部并入本总纲并按当前状态重写）。这是一个大冲刺，允许跨多个会话完成。

## 防丢失机制（先做，commit: `docs: sprint7 spec and progress file`）
1. 把本文件原样存为 `docs/sprints/sprint7_spec.md` 并 commit——以后所有 sprint spec 都进这个目录，避免再出现"spec 未被看到"的情况。
2. 新建 `docs/sprints/sprint7_progress.md`：按下方 Block 列出门项清单，每完成一项打 ✓ 并写 commit 哈希；**每个 Block 结束必须更新并 push**。若会话中断，新会话第一件事是读这个文件并从未完成项继续。
3. 每 Block 一个 review 包（outbox 下一个空闲编号），块内按项 commit。铁律沿用：数据经 `GEOFDI_DATA_ROOT`、零 `/mnt/g` 字面量、确定性 seed、图 ≤1600px、包 <20 MB、未兑现如实报告不调参、R⁻ 检验零可训练参数。Go2 默认世界 `go2_urdf_sym`。

## 执行顺序与优先级
Block 0（回填）→ W（轮足 M1 + Go2 双实机就绪）→ E（e03 外部基准）→ T（理论 Part 1 全文 + Part 2 补 N3-3）→ P（低 SNR 完整版）→ I（三通道 + 异常诊断）→ S（序贯统一）→ N2（偏置增广 + 签名 + 滚动接触）→ A（Panda 机械臂化身，加量）→ F（图厂，加量）。**W 门过后即打 `hw-ready`**，不等后面。

---

## Block 0 — 回填（commit: `chore: liu/sprint6 backfills, decisions D006–D009`）
Liu 论文（`$GEOFDI_DATA_ROOT/lit/liu2025_grufd_ftc.pdf`，若不在则从 `/mnt/c/Users/ykw/Desktop/` 或 uploads 复制）已核实事实：
1. 数据集是 **Gazebo 仿真**（legged_control NMPC+WBC；论文 III.1 明文仅仿真数据）→ `docs/protocol/liu_a1_audit.md` provenance 定案；e03 对外表述"外部公开仿真基准"。
2. 故障模型 τ_real=η·τ_cmd（η 对角，传感器精确）= 我们的 `actuator_gain`。
3. 关节顺序官方：0–2 LF、3–5 LH、6–8 RF、9–11 RH（Hip,Thigh,Calf）→ 审计文档标签改正（镜像配对不变；数据推出的符号只用于符号约定）。
4. **查对角双故障**：论文 Case 3 用 η₅+η₈（LH calf + RF calf，非镜像）。扫 CSV 的 η 场；若存在对角双故障 episode，加入 e03 预注册第四类（对角破坏镜像 → R⁻ 可见）。
5. GRU 规格（Table I）：输入 57、隐层 256、输出 12（**回归 η̂，MSE**）、100 epoch、batch 32、lr 1e-4、层数按 1 层并标待核、推理 50 Hz、低通后 η̂<0.7 判故障 → `baselines/gru.py` 加 `mode: regression_eta`；`docs/protocol/baseline_protocol.md` 更新。
6. 延迟标杆 ~1 s（无 FAR/延迟分布/ROC）；episode 1–2 s ≈ 2–4 周期 → e03 必须逐周期/半周期序贯（Block E）。
7. 实机论据进 `docs/theory_intake.md`：Liu 实机 A1 持续右偏、老化、IMU 漂移 = 真实 A2/A5 失效案例（H₀′ 场景）；论文结论承认阈值检测器抓不到细微退化 = Block P 的靶子。

Sprint 6 遗留回填：
8. `docs/theory_intake.md` 追加三条（Block T 吸收）：**Corollary N3-3 候选**——等变模型的系统误差在对称名义轨迹上被镜像配对精确抵消、全部落 Π⁺，R⁻ 对模型质量不敏感而 R⁺ 吃 β_op（证据：e13a 中 Rminus_res_eq≈Rminus_res_an 而 Rplus_res_eq≪Rplus_res_an）；**污染在最好的普通模型上已饱和**（e13b δ_f=0.67 尺寸=1.0）→ 等变性是必要条件；**centring 陷阱**（减校准均值再精确检验尺寸≈1）进 protocol。
9. `docs/decisions.md`：D006 e05a 重定性（所有幅值通道在对称漂移下膨胀，只 R⁻ 静默）；D007 实机 M1=轮足 zgws，点足候选退役，STEP 不再投入；D008 里程碑编号：milestone-2/3 并入 4，不补打；D009 双实机 = Go2 点足（trot，Σ⊂G×S¹）+ M1 轮足（滚动 Σ=G，踏步 G×S¹）。
10. `docs/protocol/protocol_params.md` 追加：URDF 阻尼 0.01 保守性、力矩来源规则、地板（e-过程 3 周期 / R⁻ 窗 10 周期）、centring 陷阱、轮角度排除规则。

---

## Block W — 双实机就绪（轮足 M1 全量 + Go2 彩排）→ tag `hw-ready`

### W1 轮足 M1 世界（commit: `feat(sim): wheeled M1 (zgws) world, 16-joint manifest, GENISOM mapping`）
- 从 `~/research/third_party/MATRiX_Python_SDK/model/zgws/` 取模型；`sim/assets/m1/m1_wheeled.xml`（原版：base com_y 3.4 mm、RAR 膝 3.3 g、惯量积）与 `m1_wheeled_sym.xml`；ctrlrange（HIP/KNEE ±60、ABAD ±40、WHEEL ±20 起步，forcerange 保留 ±150，记录）；阻尼 0.05；IMU site 基座原点；网格不进仓库，`--with-meshes` 指向 third_party。
- manifest `sim/manifests/m1_wheeled.yaml`：LF,RF,LH,RH × ABAD,HIP,KNEE,WHEEL；ABAD roll −，HIP/KNEE/WHEEL pitch +（轮速伪向量、轮轴沿 y、反射后保号）；WHEEL 的 q `exclude`（无界）；IMU a→E、ω→−E；**复用 `groups/c2.py`**。
- 映射 `io/m1_mapping.yaml`（MJCF 顺序→GeoFDI；GENISOM 名 `fl1..fl4/fr1..fr4/bl1..bl4/br1..br4`→GeoFDI，1=ABAD,2=HIP,3=KNEE,4=WHEEL 候选，`unverified: true`）；`io/m1_sdk.py` 按 `names` 动态重排、缺通道置 NaN 并列出、`efforts_semantics: unknown|current_estimate|torque` 进 meta。
- t01：对称版 1e-10；原版记录 ε_dyn 候选。
- 滚动控制器 `sim/controller_wheeled.py`（腿 PD 站姿 + 轮速 PD 到 v/r，r=0.096；按构造等变；`asymmetry` 块）：0.5/1.0/2.0 m/s × 60 s smoke。
- 踏步模式：等变 PD trot 移植（轮锁定/轮速 0）试 30 s；稳则保留 `m1_stepping`，不稳记录跳过。

### W2 滚动模式 H₀ 机器 + e01-W（commit: `feat(detect): rolling-mode data elements; e01-W`）
- `phase/registration.py` `mode: rolling`：固定时长块 L（默认 1.0 s，N=64），配对相移 0，只切 cmd 直行段（去 warm-up 2 s）；翻转群/统计量/e-过程/H₀′ 复用；H₀′ 走两样本构造（禁 centring）。
- e01-W（R=200）：`m1_wheeled_sym` 三速度 QQ + 尺寸表；**L∈{0.5,1,2} s 扫掠给出可交换性最小 L**；原版世界列（ε_dyn 效应）；ε_ctrl（单侧轮速增益 1.02、单侧 HIP 1.02）下 H₀′ 尺寸恢复 + δ 翻倍报警。
- nuisance/fault 快照（R=30）：对称载荷 1 kg、横向偏载 0.5 kg、单侧轮摩擦 ×0.7、单侧轮电机 κ=0.8、单侧 HIP κ=0.8 的 R⁻ 时间线（轮胎磨损作为"正当的镜像破缺退化"写进讨论）。
- **Q4/e13d**：M1 滚动名义数据训练等变 DeLaN（16 关节模板，前/后两模板）；等变残差 R⁻ 的尺寸与单侧轮电机 κ=0.8 功效；nuisance 读数。

### W3 Go2 彩排（commit: `feat(pipeline): go2 lowstate mapping + synthetic go2 session`）
- 用户实机 Go2 走 CycloneDDS `LowState.motor_state[20]`：Unitree 顺序 FR,FL,RR,RL × hip,thigh,calf（0–11），IMU quaternion/gyro/accel，foot_force[4]。写 `io/go2_mapping.yaml` + `io/go2_lowstate.py`（bag/CSV 导出装载，`unverified: true` 待实机确认）；合成 Go2 会话（`go2_urdf_sym` trot 3 速度 × 双向 × 30 s，按 Unitree 顺序导出）经 `ingest_session.sh` 入 `raw/sim/go2_rehearsal/`。
- **运动学相位估计器** `phase/estimator.py`（膝角 Hilbert + 接触/电流事件校正）：Go2 仿真上与真值相位比误差 <5% 周期（e03 与 Go2 实机都需要）。

### W4 一键管线 + Day-0 文档（commit: `feat(pipeline): run_pipeline.sh + day0 docs`）
- `scripts/run_pipeline.sh <session_dir> --robot m1|go2 --mode rolling|trot [--residual off|analytic|delan_equiv]`：ingest 校验 → 装载器（重排、缺失清单）→ 段切分/相位配准 → 数据元 → R⁻ H₀（QQ + 尺寸 + e-过程轨迹）→ H₀′ → 三通道读数（若残差可用）→ `report.md`。**门：M1 合成滚动会话与 Go2 合成 trot 会话均零人工干预跑通。**
- Gate 1 估计量彩排：注入 ε_ctrl 后 state-matched 分布距离估回，误差 <30%。
- `docs/protocol/m1_day0_wheeled.md` 与 `docs/protocol/go2_day0.md`：通道普查（SDK `names` 原样存档 / `ros2 topic list -t` + hz + `bag record -a` 全量）、定案项（关节数、轮编码器、efforts 语义、IMU、接触量、温度、时钟）、符号实测（逐关节手动 + 拍照）、静置 10 min、名义语料（M1：直线 3 速 × 双向 × ≥10 × ≥30 s、原地转、若有踏步；Go2：trot 直线双向 ≥10、站立、若有慢走）、nuisance（对称载荷、温度扫掠）、Day-0 不注入故障、Gate 4（关节级指令，架高/趴箱后验证）、`ingest_session.sh` 入库、`run_pipeline.sh` 第一条命令。
- `docs/protocol/protocol_params.md` 追加 L 边界与相位估计器误差。

**W 门**：t01 两世界；滚动控制器 3 速稳；e01-W 出图 + L 边界；快照表；e13d 表；Go2 映射 + 合成会话；相位估计器 <5%；两条合成会话管线零干预跑通；Day-0 双文档。→ tag `hw-ready`。

---

## Block E — e03 外部基准 + 序贯重设计（`experiments/e03_liu_a1_headtohead/`）
- **E1 序贯层**：目标 R⁻ 延迟 ≤2 周期——校准集 ≥400 周期（min p→1/401，单周期 e≈10）+ **半周期数据元**（展开定义）+ 每元 e=½p^{-1/2} 连乘报警 1/α。Go2 仿真 e04a κ=0.7 上验证：延迟中位 ≤2 周期、名义 ARL ≥1/α。写 `detect/sequential.py` 统一接口（e-过程 / e-CUSUM / conformal-CUSUM 三种）。
- **E2 e03**：数据 `raw/public/liu-a1-fault/grufd-ftc_84ca180`（100 Hz，无 τ/接触）；相位用估计器；直行段 cmd 筛；预注册（含对角类，commit 先于运行）；检测器：R⁻ 半周期 e-过程（原始信号——Liu 数据无 τ 不能做残差，注明）、R⁺=跟踪误差 conformal + Mahalanobis、GRU 回归器（Table I 规格；按文件留一训练；泛化：η 0.4↔0.6、单→双；3 seed）；校准：H₀′ ν₀ 用每文件 30 s 前缀，conformal 用同速度跨文件汇总名义段；指标：episode 内检出率、延迟（周期/秒）、名义 FAR、定位；逐 episode 表 + 汇总表 + 四类图；许可：仅派生统计。
- **E 门**：E1 达标；预注册先于运行；四类结果表；GRU 泛化表；镜像双侧格 R⁻≈α（外部数据上的 N1-2 实证）。

---

## Block T — 理论 Part 1 全文 + Part 2 补遗（tag `theory-part1-v1`、`theory-part2-v1.1`）
- **Part 1**（`02_n1_theorems.tex` 替换 `02_part1_core.tex` 替身，**保留标签 thm:n1-1/thm:n1-2**）：§1 数据元与群（展开严格定义 + 回绕近似 Proposition 及失效机制；Assumption E 块可交换性）；§2 Theorem N1-1 完整证明 + Corollary 学生化 + Remark 离散性（K≥9）+ 证伪条件；§3 Theorem N1-2（非线性响应表述、双侧盲、"差值启发式"为何错、Corollary 双通道必要性）；§4 Proposition N1-5 序贯（引 E1 半周期结果）+ Remark 校准 e-CUSUM；§5 A5 手性吸引子命题 + H₀′ 正式化；§6 N1-3/N1-4 遗留清单（O(mδ²) conjecture、IPM 路线、n 周期累积）。每定理三件套（证明/证伪/实证锚：e01a/b/c/d、e04c/d、e13a/b/c 具体图表 id）。
- **Part 2 补遗**：Corollary N3-3 [Equivariant model error is Π⁺-only]（陈述 + 证明：等变系统误差在对称名义轨迹下被镜像配对逐点抵消，Π⁻ 只余随机部分；实证锚 e13a 的 eq/an 对比）；污染推论加"必要性"Remark（e13b 饱和）；centring 陷阱进 Lemma centring 正文。
- bib：Vovk & Wang e-values（Annals of Statistics 49(3):1736–1754, 2021——WebSearch 核对后入库）；其余已在库。`make theory` 零 error。
- **T 门**：Part 1 全文三件套；N3-3 入 Part 2；bib 零编造。

---

## Block P — 低 SNR 分离网格完整版（`experiments/e08_low_snr/`，Go2）
- e13a 已覆盖 gain/bias/friction 的残差对比；本块补齐：inertia_add {10,20,50} g；**噪声上探 ×{1,2,4}**（gain/bias 全幅值）；检测器全套 = R⁻ {原始, 解析残差, 等变 DeLaN 残差} × {半周期 e-过程, e-CUSUM} + rplus_resid + Mahalanobis + AE + GRU 回归（训练含最大两档幅值，测未见小幅值，**5 seed spread**）；统一 FAR 协议；nuisance（drift_sym、payload_sym）三档噪声下重跑。
- 产出：功效 vs 幅值曲线（每类一图 × 三档噪声）、**最小可检幅值总表**（合并 e13a）、GRU spread、R⁻ nuisance 静默确认。
- **P 门**：曲线不饱和；总表；spread；静默。

## Block I — 三通道隔离 + 异常诊断（`isolation/three_channel.py`；`experiments/e09_three_channel/`）
- **先诊断**：e13c 解析残差行的 LH-KFE 摩擦故障左右归属反转（真值模型不该输给学习模型）——查解析行隔离路径的符号/索引/腿映射；给出根因与修复；修复后重跑 e13c 隔离表。
- 读数向量 = (R⁻ 状态, 关节残差行逐腿能量份额, 浮动基残差 6 行均值偏移)；假设类（R=30）：单腿执行器故障（LF-KFE κ0.8；LF-HFE b0.5）、镜像双侧等 η、横向偏载 0.5/1 kg、对称载荷 1 kg、单侧小腿 +100 g、对称漂移、名义；**预注册决策规则**（`docs/protocol/e09_preregistration.md`，commit 先于运行）：R⁻响+单腿关节行+基座零→单腿故障；R⁻静+双腿镜像关节行+基座零→镜像双侧；R⁻响+关节行静+基座 fz/mx→横向偏载；R⁻静+关节行静+基座 fz、mx≈0→对称载荷；R⁻静+全幅值通道漂移+基座零→对称漂移；单侧加质量靠 N3 字典 inertia 类分（如实报）。
- 产出：7 类混淆矩阵（解析行 / 等变残差行两版）+ 逐类三行读数图 + **接触力 10%/20% 乘性误差敏感性**。
- **I 门**：根因 + 修复；预注册先于运行；混淆矩阵；敏感性表。

## Block S — 序贯统一（`experiments/e11_sequential/`）
- conformal-CUSUM / e-CUSUM / 裸 e-过程（含半周期版）同一 ARL₀ 目标 {1/α, 5/α}：ARL₀ 实测 + e08 中幅值故障延迟；ARL–延迟权衡曲线。
- 双通道互补图（论文图）：单腿 / 镜像双侧 / 横向偏载 / 对称漂移 × R⁻/R⁺ 报警时间线。
- **S 门**：权衡曲线；互补图。

## Block N2 — 偏置增广 + 签名重构 + 滚动接触（`inekf/`；`experiments/e10_n2_signatures/`）
- 两版增广 InEKF（IMU 偏置；+12 维编码器偏置随机游走）；等变单测同 S3。
- 签名重构（realistic regime，R=20）：滑移稳态方向 vs 伴随；编码器偏置 +0.05 rad 的站立相 innovation 差分方向 vs (J(q_t)−J(q_td))·b 及增广偏置状态收敛值；陀螺偏置 0.02 rad/s 阶跃响应；DK 在增广状态空间重算 + 混淆矩阵。
- **滚动接触**：`docs/decisions/n2_rolling_contact_memo.md`（ḋ_i=R·u_i、u_i 由 r·ω_wheel 与腿运动学给出；论证 group-affine 保持；量测模型；横向不滑）+ **实现 `inekf_rolling`** 并在 `m1_wheeled_sym` 上跑 NIS 一致性 smoke（realistic regime，InEKF vs ESKF 逐箱 FAR）。
- **N2 门**：等变单测；签名/偏置读出表；DK 重算；滚动接触备忘录 + smoke 图。

## Block A（加量）— Panda 机械臂化身（`sim/assets/panda/`；`experiments/e14_arm/`）
- MuJoCo Menagerie franka_emika_panda（vendored 无网格或 submodule）；固定基 7 自由度：G 平凡、只有 R⁺——协变动量残差（Pinocchio 模型）+ 等变结构退化为普通 DeLaN（模板=整臂）+ conformal 校准 + N3 字典（增益/偏置/摩擦/负载）+ DK 证书；轨迹族：三条周期轨迹 × 两种末端负载；焊腿（e06 weld）与 Panda 并列一张表（残差统计、隔离准确率、DK 证书一致性）。这是"腿=臂之桥"的臂端与拆分预案（RA-L）的内容基础。
- **A 门**：Panda 世界；残差 + DK 表；与焊腿并列表。

## Block F（加量）— 图厂（`scripts/make_paper_figures.py`）
- 一个脚本从 results/ 重生成 `docs/paper/figure_plan.md` 列出的 10 张主图与 10 张表（统一色板、双栏宽度、字号、χ²/带/阈值标注），输出 `docs/paper/figures/` 与 `docs/paper/tables/`（PDF+PNG+CSV）；figure_plan 状态列全部更新为"已生成/待实机"。
- **F 门**：脚本一键跑通；figure_plan 状态完整。

---

## 收尾
- 每 Block review 包 + 进度文件更新 + push；W 后 `hw-ready`，T 后两个 theory tag，全部完成后 `sim-milestone-5`。
- 每 Block MANIFEST 沿用格式，Findings 单列，`docs/theory_intake.md` 追加。
- 最后输出：完成度总结（N1/N2/N3 实证锚清单、已确认局限、仍需实机的问题）+ **两台实机各自的第一条命令**（`run_pipeline.sh` 完整调用）+ 拆分预案是否有变化。
