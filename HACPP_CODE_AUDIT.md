# HAC++ 代码审计与 Baseline 核验

审计日期：2026-08-25  
审计范围：规范 Step 1–Step 4；不实现 CoView Context，不修改模型结构或算法  
当前工作区提交：`8baca87ee3e61e021d88f2b15eb2c509f6ef07d4`  
当前工作区 `origin`：`https://github.com/Hualejie/HAC-plus.git`  
官方仓库当前 `main`：`46e3c4f4e2b99f98fc9d3da3f235b63aea8504f0`

## 1. 审计结论

1. 当前 checkout 的代码与 `YihangChen-ee/HAC-plus` 当前官方 `main` 在算法文件上完全一致；`git diff HEAD refs/remotes/upstream-audit/main` 仅有 `README.md` 的论文状态文字差异。因此本文对当前代码的结论也适用于审计时的官方 `main`。
2. HAC++ 有四个需要同步的 entropy parameter 使用路径：训练渲染属性量化、训练 5% rate sampling、`estimate_final_bits()`、真实 `conduct_encoding()`/`conduct_decoding()`。CoView 若只接其中一个路径，会造成训练目标或 codec 概率模型不一致。
3. 最合适的 CoView 接入层是 `calc_interp_feat(anchor_xyz)` 之后、feature/scaling/offset 参数被消费之前的统一 entropy parameter predictor。Phase 2 应以单一 helper 同时覆盖训练 rate、final estimate、encoder 和 decoder；不要从 rendering visible mask 推导 rate context，也不要修改 `Channel_CTX_fea` 的 chunk 顺序。
4. 当前名为 `calculate_morton_order()` 的函数并未计算真正的 Morton/Z-order code。它使用 `x @ base**[0,1,2]` 形成混合进制排序键。规范中所有“Morton order”应改称“当前 codec canonical sort（代码函数名仍为 `calculate_morton_order`）”，除非后续明确修改 baseline。
5. 当前 codec 不是 standalone decoder：MLP 权重、模型配置和部分状态不在 `bitstreams/` 中；更关键的是 `conduct_decoding()` 虽先读出 hash bitstream，却在 attribute 全部解码后才把 decoded hash 写回 `encoding_xyz`。attribute 解码时实际依赖同一进程中仍驻留的训练态 MLP/hash module。
6. Baseline 的静态控制流已确认是 `train -> estimate -> encode -> decode -> test -> save decoded model -> reload -> render -> evaluate`。但本机无法完成动态 baseline：仓库没有数据和已有输出，目标环境没有可用 PyTorch，CUDA 扩展未安装，`tmc3` 不在 PATH，官方 run shell 还是 Linux shell 语法。故本审计不报告虚构的 size/fidelity/time 数值。

## 2. 关键对象与真实数据流

### 2.1 GaussianModel、参数与 camera

- `GaussianModel` 在训练入口 `train.py:85-97` 实例化，在最终重载渲染入口 `train.py:421-434` 再实例化。
- `ModelParams.n_offsets` 默认值为 10（`arguments/__init__.py:47-52`），通过位置参数 `dataset.n_offsets` 传到 `GaussianModel.__init__()`，保存为 `self.n_offsets`（`scene/gaussian_model.py:237-265`）。构造函数自身默认值 5 不代表官方实验值。
- `Scene` 根据 `sparse/` 或 `transforms_train.json` 选择 loader（`scene/__init__.py:45-52`），将 `scene_info.train_cameras` 转成 `Camera` 列表（`scene/__init__.py:72-84`），由 `getTrainCameras()` 返回（`scene/__init__.py:103-104`）。
- 每个 `Camera` 已保存 `R/T/FoV/image size/znear/zfar/world_view_transform/projection_matrix/full_proj_transform/camera_center`（`scene/cameras.py:17-57`），但这些 camera metadata 当前没有进入 `GaussianModel`，也没有写入 entropy bitstream。

### 2.2 Anchor 的三种索引/顺序

训练期存在三个不可混淆的集合：

```text
当前全体 anchor index
    ├─ visible_mask ──> 当前 camera 的 rendering anchors
    └─ choose_idx(~5%) -> 与当前 camera 无关的 rate sampled anchors
```

- `prefilter_voxel()` 为全体 `pc.get_anchor` 生成 `visible_mask`（`gaussian_renderer/__init__.py:287-342`）。
- `generate_neural_gaussians()` 用该 mask 选择 rendering anchors（`gaussian_renderer/__init__.py:30-37`）。
- 训练 rate 分支独立地对全体 `pc.get_anchor` 采样约 5%（`gaussian_renderer/__init__.py:71-78`），不继承 `visible_mask`。

最终 codec 再建立另一套 canonical order：

```text
current anchors
  -> get_mask_anchor 过滤
  -> valid-anchor order
  -> round(anchor / voxel_size)
  -> calculate_morton_order() 的当前排序
  -> codec canonical order
```

若 Phase 1 需要把最终属性与原训练索引对齐，应显式保存：

```python
valid_original_idx = torch.nonzero(mask_anchor, as_tuple=False)[:, 0]
codec_original_idx = valid_original_idx[sorted_indices]
```

不能假定 current index、visible subset index、valid index 和 codec index 相同。

### 2.3 Hash context 与 mlp_grid

`calc_interp_feat(x)` 接收世界坐标/codec 坐标，不要求调用者预先归一化；函数内部使用 `x_bound_min/x_bound_max` 映射到 `[0,1]`，再查询 `encoding_xyz`（`scene/gaussian_model.py:525-531`）。当 `use_2D=True` 时，`encoding_xyz` 实际为 mixed 3D + XY/XZ/YZ hash encoder（`scene/gaussian_model.py:48-114,307-329`）。

`mlp_grid` 的输入是 mixed hash feature，输出按以下次序切分（`scene/gaussian_model.py:369-373,1145-1150`）：

| 输出段 | 维度 | 用途 |
|---|---:|---|
| `mean` | `feat_dim` | feature base Gaussian mean |
| `scale` | `feat_dim` | feature base Gaussian scale |
| `prob` | `feat_dim` | feature base mixture logits |
| `mean_scaling` | 6 | scaling Gaussian mean |
| `scale_scaling` | 6 | scaling Gaussian scale |
| `mean_offsets` | `3*n_offsets` | offset Gaussian mean |
| `scale_offsets` | `3*n_offsets` | offset Gaussian scale |
| `Q_feat_adj` | 1 | per-anchor feature quantization adjustment |
| `Q_scaling_adj` | 1 | per-anchor scaling quantization adjustment |
| `Q_offsets_adj` | 1 | per-anchor offset quantization adjustment |

官方 CLI 默认 `n_features=4`（`train.py:582-584`）。因此默认 mixed encoder 输出是 12 个 3D levels × 4，加上 3 组 4 个 2D levels × 4，共 96 维；默认 `mlp_grid` 输出是 225 维。源码中的若干旧注释仍写 32、48 或 56，不可据此推断实际 shape。

### 2.4 Channel_CTX_fea 与 feature mixture

非 synthetic 场景使用 `Channel_CTX_fea`，synthetic NeRF 使用 `Channel_CTX_fea_tiny`（`scene/gaussian_model.py:375-379`）。两者都把 50-D feature 固定切为 5 个 10-D chunk。

普通版本的依赖顺序是：

```text
chunk 0: base mean/scale/prob
chunk 1: decoded chunk 0 + base mean/scale/prob
chunk 2: decoded chunks 0..1 + base mean/scale/prob
chunk 3: decoded chunks 0..2 + base mean/scale/prob
chunk 4: decoded chunks 0..3 + base mean/scale/prob
```

对应实现为 `scene/gaussian_model.py:116-167`。encoder 虽把完整 quantized `feat` 传给 module，但每个 `MLP_dN` 只拼接先前 chunks；decoder 从全零 tensor 开始并在每轮写入刚解出的 chunk（`scene/gaussian_model.py:1464-1481`），所以两侧 autoregressive condition 一致。

synthetic tiny 版本中 chunk 0 是可学习常数，后续 chunk 只依赖先前 feature chunks，不使用传入的 base `mean_scale`（`scene/gaussian_model.py:169-219`）。规范应区分这一路径。

feature 最终使用两个 Gaussian component：hash/base component 与 intra-anchor adjusted component。`prob`/`prob_adj` 经过 softmax，estimated rate 调用 `Entropy_gaussian_mix_prob_2`，真实 codec 调用 mixed Gaussian arithmetic coder。

## 3. HAC++ 当前 entropy coding 完整调用链

### 3.1 训练 rate path

```text
train.py::training()
  -> random train camera
  -> prefilter_voxel(camera, gaussians)
  -> render(... visible_mask ...)
  -> generate_neural_gaussians()
       rendering branch:
         visible anchors
         -> calc_interp_feat()
         -> mlp_grid
         -> Q adjustments
         -> noisy quantization for rendering

       rate branch (step > 10000):
         all anchors --random ~5%--> anchor_chosen
         -> calc_interp_feat(anchor_chosen)
         -> mlp_grid -> base params + Q adjustments
         -> Channel_CTX_fea(_tiny) -> adjusted feature params
         -> softmax(base/adjusted logits)
         -> Entropy_gaussian_mix_prob_2(feature)
         -> Entropy_gaussian(scaling)
         -> Entropy_gaussian(offset)
         -> multiply mask_anchor / per-offset mask
         -> average bits per parameter
  -> reconstruction loss + lambda * (attribute rate + hash rate)
```

代码位置：`train.py:140-188`；`gaussian_renderer/__init__.py:46-119`；`utils/entropy_models.py:30-86`。

注意：rendering branch 和 rate branch 都调用 `mlp_grid`，但输入 anchor 集不同。CoView 是 entropy context，不应把 rate sampling 改成当前 camera visible anchors。

### 3.2 final estimated bits path

```text
training_report(final test iteration)
  -> GaussianModel.estimate_final_bits()
     -> get_mask_anchor 过滤最终有效 anchors
     -> get_anchor（voxel STE quantized position）
     -> calc_interp_feat -> mlp_grid
     -> Q-adjusted attribute quantization
     -> Channel_CTX_fea(_tiny)
     -> estimated feature/scaling/offset bits
     -> estimated hash/mask bits
     -> analytical MLP size
```

代码位置：`train.py:270-277`；`scene/gaussian_model.py:1129-1193`。

这里没有 canonical sort。当前 hash-only probability 是逐 anchor 的，排序不改变总估计；一旦引入依赖图或 causal neighbor context，该路径必须显式使用与 codec 相同的 final-valid canonical order。

另一个重要差异是 estimated anchor size 仅用 `N * 3 * anchor_round_digits`，其中 `anchor_round_digits=16`（`utils/encodings.py:11-12`），它不是 GPCC 的真实大小。

### 3.3 真实 encoding path

```text
training_report(final test iteration)
  -> conduct_encoding(bitstreams/)
     -> get_mask_anchor 过滤
     -> anchor_int = round(anchor / voxel_size)
     -> calculate_morton_order(anchor_int)
     -> 同步重排 anchor/feature/scaling/offset/mask
     -> GPCC encode sorted anchor_int -> xyz_gpcc.npz
     -> for each MAX_batch_size=3000 batch:
          calc_interp_feat(anchor_slice)
          -> mlp_grid -> entropy params + Q
          -> for feature chunk 0..4:
               Channel_CTX_fea(... to_dec=cc)
               -> encoder_gaussian_mixed_chunk
               -> encoder_gaussian_mixed
               -> arithmetic.calculate_cdf
               -> arithmetic.arithmetic_encode
          -> scaling:
               encoder_gaussian_chunk
               -> encoder_gaussian -> arithmetic coder
          -> valid offsets only:
               encoder_gaussian_chunk -> arithmetic coder
     -> hash embeddings:
          STE binary {-1,+1} -> {0,1}
          -> encoder() global Bernoulli arithmetic coder
     -> masks:
          encoder() global Bernoulli arithmetic coder
```

代码位置：`scene/gaussian_model.py:1196-1374`；`utils/encodings_cuda.py:177-250,319-380,439-468`。

真实 attribute 文件按 batch/chunk 分散写出，不存在单一 stream 内必须遵循的物理拼接顺序。encoder 的计算调度是 anchor，随后每 batch feature -> scaling -> offset，最后 hash -> mask。

### 3.4 真实 decoding path

```text
conduct_decoding(bitstreams/)
  -> load x_bound_min/x_bound_max
  -> GPCC decode anchor_int set
  -> calculate_morton_order(decoded anchor_int)
  -> anchor_decoded = anchor_int * voxel_size
  -> decode masks
  -> decode hash bits（暂存 local hash_embeddings）
  -> for same 3000-anchor batches:
       calc_interp_feat(anchor_sort) using currently resident encoding_xyz
       -> mlp_grid
       -> feature chunk 0..4 sequential decode
       -> scaling decode
       -> masked offset decode
  -> replace anchor/feature/scaling/offset/mask
  -> only now replace encoding_xyz params with decoded hash_embeddings
  -> decoded_version = True
```

代码位置：`scene/gaussian_model.py:1377-1560`；`utils/encodings_cuda.py:254-316,382-436,471-496`。

decoder 的文件读取调度是 anchor -> mask -> hash -> per-batch feature/scaling/offset，和 encoder 的计算调度不相同，但每种属性的 canonical anchor order、batch boundary 和 feature chunk order相同。

## 4. 规范第 58 节 18 个审计问题

| # | 问题 | 当前实现结论 |
|---:|---|---|
| 1 | `GaussianModel` 在哪里实例化？ | 训练：`train.py:85-97`；最终 reload/render：`train.py:421-434`。`Scene` 接收已构造实例。 |
| 2 | `ModelParams.n_offsets` 如何传入？ | `arguments/__init__.py:51` 默认 10，经 `dataset.n_offsets` 位置参数传入并赋给 `self.n_offsets`。构造函数默认 5 不生效。 |
| 3 | `Scene` 如何读取 train cameras？ | 根据 Colmap/Blender loader 得到 `scene_info.train_cameras`，可选 shuffle，调用 `cameraList_from_camInfos`，存入 `self.train_cameras[scale]`。 |
| 4 | `prefilter_voxel()` 是否只服务 rendering？ | 否。它直接服务 render acceleration/visible selection，同时其 `voxel_visible_mask` 传给 `training_statis()` 更新 densification/pruning 统计（`train.py:149-151,213-219`）。它还依赖 scaling/rotation，不适合作为 geometry-only CoView observation。 |
| 5 | visible anchor 与 rate sampled anchor 区别？ | visible anchor 由当前 camera + anchor scaling/rotation 的 rasterizer filter 得到；rate anchor 是从全体当前 anchors 独立 Bernoulli 采样约 5%。二者没有包含关系保证。 |
| 6 | `calc_interp_feat()` 输入已归一化吗？ | 调用者传世界/codec 坐标；函数内部才归一化。不能向它再传已归一化坐标。 |
| 7 | `mlp_grid` 各输出段？ | `mean, scale, prob, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj`，维度见 2.3。 |
| 8 | `Channel_CTX_fea` 5 chunk 如何用于 codec？ | encoder 按 0..4 编码，每个 chunk 只用先前 chunks；decoder 从零开始按 0..4 回填，形成同样条件。tiny 路径 chunk 0 为常量且不使用 base `mean_scale`。 |
| 9 | `get_mask_anchor` 如何决定最终 anchors？ | `get_mask` 的前 10 个 offset masks 求均值，只要至少一个为 1，anchor 就进入 bitstream。这里硬编码了 10。 |
| 10 | anchor 为什么排序？ | GPCC 不保证保留输入点顺序；属性又必须和 decoder 恢复的点序对齐，所以两侧都按坐标键恢复 canonical order。当前算法并非真正 Morton bit interleave。 |
| 11 | GPCC decode 坐标是否与 encoder 一致？ | GPCC 配置 `positionQuantizationScale=1`，设计上 lossless，故坐标集合应一致；代码没有 assert set equality。排序后顺序通常一致，但重复坐标/相同排序键的稳定性没有测试保护。 |
| 12 | encode/decode batch order 完全一致？ | 在坐标 canonical order 一致的前提下，二者都按 `MAX_batch_size=3000` 切 batch，batch 内 feature 0..4、scaling、offset 的元素顺序一致。全局 hash/mask 的调度顺序不同但文件独立。 |
| 13 | hash grid 参数如何编码？ | `get_encoding_params()` 将 3D/三组 2D params 拼接并 STE 二值化，映射到 `{0,1}`，用一个带全局 `P(1)` header 的 Bernoulli arithmetic coder 写 `hash.b`。仅默认 `ste_binary=True` 路径完整。 |
| 14 | mask 如何编码？ | 仅最终 valid anchors 的前 10 个 mask，经同一 canonical sort 后 flatten，用全局 Bernoulli arithmetic coder写 `masks.b`；decoder 根据 GPCC 点数推断 `N*n_offsets`。 |
| 15 | MLP size 如何计入 Total？ | `get_mlp_size()` 遍历所有 `named_parameters()`，名字包含字符串 `mlp` 的参数一律按 FP32（32 bit）计入。它是分析值，不是 `bitstreams/` 中实际文件大小。 |
| 16 | 新增 `mlp_coview` 后 Total 自动增加？ | 若它是 `GaussianModel` 注册子模块且参数名包含 `mlp`，分析 Total 会自动增加；optimizer、checkpoint save/load、codec 可用性都不会自动增加，必须显式接线。 |
| 17 | renderer 是否使用 decoded attributes？ | 是。最终迭代先在原 model 上 decode 并设置 `decoded_version=True`，随后保存 decoded PLY；后处理又以 `decoded_version=run_codec=True` 重新实例化并加载该 PLY/checkpoint。getter 直接返回 decoded scaling/mask/anchor。 |
| 18 | final test 是否在 decode 后执行？ | 是。最终 `training_report()` 中先 estimate/encode/decode，再计算 test/train fidelity；训练结束后又 reload decoded model、render test views 并运行 `evaluate()`。 |

## 5. Baseline train -> encode -> decode -> test 核验

### 5.1 静态流程确认

官方 run scripts 最终只调用一次 `train.py`。`train.py` 内部完成：

1. 训练至默认 30,000 iterations；
2. 最终 `training_report()` 调用 `estimate_final_bits()`；
3. `run_codec=True` 时调用 `conduct_encoding()`；
4. 紧接着调用 `conduct_decoding()`，替换内存 attributes；
5. 立即对 test cameras 和少量 train cameras 计算 L1/PSNR/SSIM/LPIPS；
6. 保存此时已 decoded 的 PLY 与 MLP/hash checkpoint；
7. 训练结束后 `render_sets()` reload 最新 iteration，渲染 test set；
8. `evaluate()` 从保存的 renders/gt 再计算最终 SSIM/PSNR/LPIPS 并写 `results.json`。

关键位置：`train.py:202-234,261-360,635-652`。

### 5.2 动态执行状态

结论：**本机未能启动 baseline，因此没有有效实验指标。** 这是环境阻塞，不是 baseline 成功或失败结论。

| 检查项 | 当前状态 | 证据/影响 |
|---|---|---|
| 数据集 | 缺失 | 仓库无 `data/`；在 `E:\3DGS\4.2MEGS` 范围也未找到 `transforms_train.json`。 |
| 已训练输出 | 缺失 | 仓库无 `outputs/`，`results/` 没有可用日志或指标。 |
| Python/PyTorch | 不可用 | base Python 3.11.14 无 `torch`；`c3dgs` conda env 没有可执行 Python。`conda run -n base python train.py --help` 在 import torch 处失败。 |
| CUDA | 仅驱动/toolkit 可见 | GTX 1660 Ti 6GB；driver 577.00；driver 显示 CUDA 12.9；本机 toolkit 12.1。仓库环境要求 Python 3.7/PyTorch 1.12.1/CUDA 11.6，README 声称 Ubuntu 20.04/CUDA 11.8。 |
| CUDA extensions | 未安装 | submodules 仍是 zip；`diff-gaussian-rasterization`、`simple-knn`、`gridencoder`、`arithmetic` 无可 import 的当前环境构建。 |
| GPCC | 缺失 | `tmc3` 不在 PATH，在本项目父目录范围也未发现 `tmc3.exe`；真实 encode/decode 必然失败。 |
| run shell 跨平台 | 当前 Windows 不可直接运行 | 脚本命令以 `CUDA_VISIBLE_DEVICES=0 python ...` 开头，`cmd.exe` 报该命令不存在；README 明确只报告 Ubuntu 测试环境。 |
| GPU 容量 | 风险 | 6GB 显存是否能跑官方 30k 配置未验证，尤其是 Mip-NeRF360；应先用一个 synthetic 小场景做 smoke/baseline。 |

### 5.3 Baseline 指标记录

| 指标 | 当前记录 |
|---|---:|
| Encoded total MB | N/A（未执行） |
| anchor MB | N/A（未执行） |
| feature MB | N/A（未执行） |
| scaling MB | N/A（未执行） |
| offset MB | N/A（未执行） |
| hash MB | N/A（未执行） |
| mask MB | N/A（未执行） |
| analytical MLP MB | N/A（未执行） |
| PSNR | N/A（未执行） |
| SSIM | N/A（未执行） |
| LPIPS | N/A（未执行） |
| training time | N/A（未执行） |
| encoding time | N/A（未执行） |
| decoding time | N/A（未执行） |

完成动态 baseline 前至少需要：准备一个官方数据场景、建立与 CUDA/toolchain 兼容的 `HAC_env`、解压并编译四个 CUDA submodules、安装并加入 `tmc3` PATH，以及把 run script 改为 PowerShell/跨平台启动方式或在 WSL/Linux 上运行。环境具备后应保存完整 `outputs.log`、`results.json`、`bitstreams/` 的逐文件物理大小和峰值显存。

### 5.4 报告 size 时必须区分的三种口径

1. `estimate_final_bits()`：概率模型估计值；anchor 是固定 16 bit/axis 近似。
2. `conduct_encoding()` log：attribute coder 返回值 + `xyz_gpcc.npz` 文件大小 + 按参数数目估算的 FP32 MLP size。
3. 实际交付 footprint：`bitstreams/` 全目录 + decoder 所需 checkpoint/config/metadata 的真实文件大小。

当前 `Encoded Total` 不是第 3 种口径：它没有按物理大小统计两个 `x_bound_*.pkl`，MLP 也不在 bitstreams 中；相反它额外加了注释所称的 24-byte xyz bounds。后续论文表格应明确口径，并优先同时报告真实目录大小。

## 6. CoView Context 最合适的实际接入点

推荐将接入点定义为统一的 base entropy parameter prediction，而不是直接散落修改现有调用：

```python
predict_entropy_params(
    anchor_xyz_codec_equivalent,
    coview_context=None,
)
```

内部次序：

```text
anchor xyz
  -> calc_interp_feat()                 # existing hash context
  -> mlp_grid()                         # existing base parameters
  -> optional small coview correction   # Phase 2, default disabled
  -> named/split entropy parameters
```

这个 helper 应成为以下位置的唯一参数入口：

- `generate_neural_gaussians()` 的训练 rate branch；
- `estimate_final_bits()`；
- `conduct_encoding()` 每个 codec batch；
- `conduct_decoding()` 每个 codec batch；
- 若 CoView 会修改 Q adjustment，再覆盖 rendering visible-attribute quantization branch；若 Phase 2 仅修正 feature base mean/scale，rendering quantization branch不应被无关修改。

为什么不是其他层：

- 不能接在 `prefilter_voxel()`：它是单 camera、依赖 scaling/rotation，且影响 densification。
- 不能只接当前 visible anchors：训练 rate sampling 与 visible set 分离。
- 不应先接 `Channel_CTX_fea`：这会改变现有 intra-anchor autoregressive 语义并扩大 codec 风险。
- 不能只在 encoder 计算 CoView：decoder 必须从相同 codec-equivalent positions、相同 train-camera metadata 和确定性算法复现完全相同的 context。

Phase 1 的图应基于最终 valid anchors 的 codec-equivalent coordinates 和当前 canonical sort。Phase 2 的 ViewSelf context 可以是非 causal 的 geometry-only per-anchor descriptor；若进入 Phase 3 使用邻居 attributes，必须限制为 canonical order 中已解码 anchors。

## 7. Phase 1 / Phase 2 预计文件清单

### 7.1 Phase 1：只做数据分析，不改算法

建议新增：

- `scene/coview_context.py`：geometry-only observation、sparse incidence、Top-K、canonical ordering utilities。
- `analysis/analyze_coview.py`：加载 trained scene/checkpoint，输出 overlap/correlation/controlled-distance 分析。
- `tests/test_coview_context.py`：投影 convention、camera permutation invariance、无 dense NxN、排序对齐测试。
- `utils/coview_cache.py`：可选；只保存版本化、带 scene/camera/anchor hash 的稀疏 cache。

Phase 1 原则上无需修改 `gaussian_renderer`、`mlp_grid`、`Channel_CTX_fea` 或 codec。若分析脚本无法干净复用参数解析，可给脚本独立 CLI，不要为了分析侵入训练主链。

### 7.2 Phase 2：ViewSelf/base parameter correction

预计修改：

- `scene/gaussian_model.py`：注册可关闭的 CoView module；统一 `predict_entropy_params()`；checkpoint save/load；optimizer/scheduler；estimate/encode/decode 同步。
- `gaussian_renderer/__init__.py`：训练 5% rate branch 改用统一 predictor；只在语义需要时同步 rendering quantization branch。
- `arguments/__init__.py`：`use_coview_context=False` 及最小超参数。
- `train.py`：从 `Scene.getTrainCameras()` 初始化/绑定确定性 camera metadata；日志、消融标识与 baseline compatibility。
- `scene/__init__.py`：仅当需要统一导出稳定 train-camera metadata 时修改；优先保持 Scene loader 不变。
- `scene/coview_context.py`：复用 Phase 1 utilities，增加 ViewSelf encoder/context builder。
- `utils/coview_cache.py`：版本、camera ID、anchor canonical order、x-bound/voxel-size fingerprint。
- `tests/test_coview_context.py`：训练/estimate/encode/decode 参数 parity、bit-exact round trip、开关关闭恢复 baseline。
- 可选新增 `tests/test_entropy_context_parity.py`：专门比较四条 predictor 路径。

不能依靠“名字包含 mlp”自动完成集成：`mlp_coview` 的 analytical size 可能自动计入，但 optimizer、eval/train mode、checkpoint、真实 decoder 依赖和 metadata 都必须显式处理。

## 8. 编解码一致性风险

按优先级排序：

1. **Standalone 状态缺失。** 当前 decoder 需要已实例化且权重正确的 rendering MLP、`mlp_grid`、`mlp_deform`、hash module、结构参数和 bounds；CoView 还会新增 train-camera metadata/context params。必须先定义 codec package 边界。
2. **decoded hash 安装过晚。** `hash_embeddings` 在 attribute 解码前读出，却在 attribute 解码后才写回。当前同进程 round trip 依靠驻留模型碰巧与 bitstream 一致。Phase 2 parity test 必须从 fresh model/process 解码，否则会掩盖问题。
3. **排序函数不是 Morton 且 tie 未保护。** encoder/decoder 对同一坐标集合通常得到相同顺序，但重复坐标或相同 key 的顺序没有显式 tie-break/checksum。CoView graph 的 node ID 不能仅凭函数名假设标准 Morton。
4. **camera metadata 未编码。** decoder 若没有完全相同的 train cameras、投影 convention、float dtype 和 canonical camera ID，就无法复现 geometry observation。
5. **当前 index 到 codec index 映射易错。** densification/pruning 改变当前 anchor tensor；`get_mask_anchor` 再过滤；codec 再排序。任何长期按训练早期 index 缓存的图都会错位。
6. **`n_offsets` 实际有硬编码 10。** `get_mask()`、decode mask assignment 等使用 `:10`。Phase 2 新代码必须使用 `self.n_offsets`，但不能假称 baseline 已支持任意值。
7. **四条 predictor 路径重复。** 当前 split/Q/scale clamp 逻辑散落在 renderer、estimate、encoder、decoder；局部改动极易产生概率或 tensor layout 差异。
8. **float/device determinism。** camera projection、Top-K tie-break、hash interpolation、`tanh` Q 和 arithmetic CDF 对微小数值差异敏感。必须固定 dtype、排序 tie-break 和 cache fingerprint。
9. **batch/chunk boundary。** codec 固定 outer batch 3000，feature coder另有 inner chunk size。causal CoView 不得跨 batch 引入 encoder 可见而 decoder 尚未解码的 future state。
10. **size 统计漏项。** camera/context cache、CoView MLP checkpoint、bounds/config 若未计入，会产生虚假的 total gain。
11. **synthetic tiny 分支不同。** 它不使用 base `mean_scale` 生成 intra component；Phase 2 不能只在普通 `Channel_CTX_fea` 上验证。
12. **Windows/Unix 启动差异。** 不先固定可复现环境，baseline timing 与 codec 文件行为无法比较。

## 9. 规范需要修正或收紧的内容

1. 将“真实 Morton order”改为“调用 `calculate_morton_order()` 得到的当前 codec canonical sort”；注明该实现不是 Morton/Z-order bit interleave。
2. 将“GPCC decode 后坐标与 encoder 完全一致”改为“lossless 配置预期坐标 multiset 一致，必须新增集合、数量、排序后逐点 assert；重复点/tie 需要测试”。
3. 将“`prefilter_voxel()` 只服务 rendering”改为“服务 rendering acceleration/selection，并参与 densification statistics；且依赖 scale/rotation，禁止作为 geometry-only observation”。
4. 将“`calc_interp_feat()` 输入是否已归一化”明确为“输入是 world/codec xyz，函数内部归一化”。
5. 补充训练期有两个 `mlp_grid` 使用：visible rendering attributes 的 adaptive Q 与全体 anchors 5% rate sampling。统一 predictor 时应按所修改的输出字段决定是否同时改前者。
6. 补充 `estimate_final_bits()` 的 anchor bits 是固定 16 bit/axis 估计，不是 GPCC bits，且不做 canonical sort。
7. 补充 `Channel_CTX_fea_tiny` 的真实语义：chunk 0 为 learned constant，后续只依赖 previous chunks，不使用 base `mean_scale`。
8. 将“所有代码不得硬编码 10”保留为新增代码要求，但注明 baseline 自身在 mask getter/decoder 中已硬编码 10，因此当前实现实际上只安全支持 `n_offsets=10`。
9. 修正“decoder decode hash 后用 decoded hash 预测 attributes”的描述：当前代码先解 hash 到 local tensor，attribute decode 仍使用 resident `encoding_xyz`，最后才安装 decoded hash。这是应在 Phase 2 前解决或至少用 fresh-process parity test 暴露的 baseline 风险。
10. 明确 `bitstreams/` 不是 standalone package。MLP/checkpoint/config、camera metadata、context cache与真实 bounds 文件必须纳入 decoder contract 和总大小。
11. 修正“MLP 自动计入 Total”的表述：仅注册参数名含 `mlp` 时会被 `get_mlp_size()` 按 FP32 分析计数；它不会自动保存、加载或出现在物理 bitstream 中。
12. 区分 encoder/decoder 的计算调度与元素顺序。二者 global hash/mask 调度不同，但 attribute canonical element/batch/chunk order必须相同。
13. 规范中的 shape 注释不应沿用源码旧注释；实际 mixed hash output 由 CLI `n_features`、3D/2D level 数共同决定，官方默认是 96。
14. Phase 1 cache key 必须包含：训练 camera 集合的 canonical fingerprint、final valid codec anchor coordinates、排序算法版本、`voxel_size`、projection convention 和代码提交号。

## 10. 本阶段边界与停止点

本阶段只新增本审计文档，并执行了只读代码/环境检查与 Python syntax compile。未实现 `mlp_coview`、未新增 CoView module、未修改 renderer/model/codec/参数结构，也未声称 baseline 指标已测得。

进入 Phase 1 前建议先由用户确认：

1. 是否以当前 checkout/官方同算法版本作为冻结 baseline；
2. baseline 数据场景与运行环境位置；
3. 是否先修复/定义 standalone decoder contract，还是在研究阶段明确限定为同进程 codec；
4. Phase 1 是否使用真正 Morton order 作为新实验排序，或严格保持当前 baseline 的 misnamed canonical sort。为保持 baseline，默认应选择后者。

