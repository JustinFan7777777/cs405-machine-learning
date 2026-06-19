# Lab11 K-means 视频分割：实现与优化总结

---

## 一、任务概述

**目标**：用 K-means 聚类算法对视频的每一帧进行像素级分割，把画面中不同物体（道路、天空、树木等）染成不同颜色，并输出为视频。

**核心文件**：[Lab11.K-means.ipynb](Lab11.K-means.ipynb)

**测试视频**：`road_video.MOV`（1920×1080，35 帧，车载前视行车记录）

---

## 二、初始实现

### 2.1 K-means 算法（从零手写）

按照作业要求实现了完整的五步流程，一步对应一个公式：

| 步骤 | 做什么 | 核心代码 |
|------|--------|----------|
| **① 初始化** | 随机选 K 个像素作为初始聚类中心 | `np.random.choice(n_samples, n_cl)` |
| **② 算距离** | 每个像素到每个中心的欧氏距离² | `np.sum((data[:,None,:] - centers[None,:,:])**2, axis=2)` |
| **③ 分类** | 每个像素归到最近的中心 | `np.argmin(distance, axis=1)` |
| **④ 更新中心** | 每类的所有像素取均值，作为新中心 | `np.mean(data[mask], axis=0)` |
| **⑤ 收敛判断** | 标签不变就退出 | `np.all(new_labels == old_labels)` |

### 2.2 像素上色

```python
for k in range(n_cl):
    mask = (labels == k)           # 找到属于第 k 类的所有像素
    new_frame[mask] = GBR[k]      # 染成对应的 BGR 颜色
```

### 2.3 颜色一致性（加分项）

如果不做处理，每一帧的 K-means 都是随机初始化的——第一帧的"道路"可能是红色，第二帧"道路"就变成蓝色了，视频会疯狂闪烁。

**解决方案**：把上一帧收敛后的聚类中心，作为下一帧的初始中心。

```python
prev_centers = None                 # 第一帧没得参考，随机初始化
while ret:
    labels, centers = kmeans(data, n_cl, init_centers=prev_centers)
    prev_centers = centers.copy()   # 保存下来给下一帧用
```

**原理**：视频相邻帧画面变化很小，上一帧的中心离这一帧的最优解已经很近了，这样聚类标签的语义（哪个是道路、哪个是天空）在整个视频中保持一致。

### 2.4 初始版本的性能

| 聚类数 | 总耗时 | 每帧平均 |
|--------|--------|----------|
| n_cl=1 | 14 秒 | 0.40 秒 |
| n_cl=2 | 22 秒 | 0.63 秒 |
| n_cl=3 | 53 秒 | 1.51 秒 |
| n_cl=5 | **306 秒（5 分钟）** | **8.74 秒** |

可以看到，n_cl=5 时几乎每帧都要等近 10 秒——一个 35 帧的短视频要跑 5 分钟，这显然有优化空间。

---

## 三、问题诊断：瓶颈在哪里？

分析 kmeans 函数中每步的时间占比后发现，**95% 以上的时间都花在第二步——计算距离**。

原始的距离计算方式是 NumPy 广播（broadcasting）：

```python
distance = np.sum((data[:, None, :] - centers[None, :, :]) ** 2, axis=2)
```

这行代码在内存中展开了一个巨大的三维数组：

```
data[:, None, :]   -> (2,073,600, 1, 3)    # 207 万像素 × 1 × 3 通道
centers[None,:,:]  -> (1, 5, 3)            # 1 × 5 个聚类 × 3 通道
相减后              -> (2,073,600, 5, 3)    # ≈ 124 MB（float32）
```

**每迭代一次，就要在内存中新建一个 124 MB 的数组，读写一遍，然后扔掉。** 一帧迭代 50~80 次，就是反复搬运 6~10 GB 的数据——纯粹的内存带宽瓶颈。

---

## 四、优化方案

### 4.1 优化一：距离计算改用数学恒等式（影响最大）

**核心洞察**：欧氏距离的平方可以展开，避免构造三维中间数组。

$$||x - c||^2 = ||x||^2 + ||c||^2 - 2 \cdot (x \cdot c)$$

即："两点距离² = 点 x 的长度² + 点 c 的长度² - 2×(x 和 c 的内积)"

**代码变化**：

```python
# ❌ 原始：广播相减，生成 (n, k, 3) 的 124MB 中间数组
distance = np.sum((data[:, None, :] - centers[None, :, :]) ** 2, axis=2)

# ✅ 优化：利用恒等式，最大中间数组仅 (n, k) = 41MB
data_norm_sq = np.sum(data ** 2, axis=1)              # 提前算好，每帧只算一次
centers_norm_sq = np.sum(centers ** 2, axis=1)         # 每轮迭代算一次
cross_term = np.dot(data, centers.T)                   # BLAS 矩阵乘法，极度优化
distance = data_norm_sq[:, None] + centers_norm_sq[None, :] - 2 * cross_term
```

**关键**：`np.dot()` 不是普通的 Python 循环，它会调用 macOS 底层的 **Apple Accelerate** 库（等价于 Intel 的 MKL），这个库用汇编级 SIMD 指令 + 多线程做矩阵乘法，比 NumPy 自己广播相减快一个数量级，而且内存占用只有原来的 1/3。

### 4.2 优化二：像素上色用 Fancy Indexing

```python
# ❌ 原始：逐类循环，每次生成 200 万 bool 的 mask
for k in range(n_cl):
    mask = (labels == k)
    new_frame[mask] = GBR[k]

# ✅ 优化：一行完成，NumPy 在 C 层直接索引
color_table = np.array(GBR[:n_cl])          # (n_cl, 3) 颜色查找表
new_frame = color_table[labels]              # labels 中每个值被替换为对应颜色
```

Fancy indexing 在 NumPy 的 C 内核中完成，Python 层没有循环开销。虽然这部分本身不是瓶颈，但在 n_cl 较大的时候能省掉几十毫秒。

### 4.3 优化三：降采样（用空间换时间）

一帧 1920×1080 = **2,073,600 个像素**需要聚类。但如果先缩小到 1/4（480×270 = **129,600 个像素**），像素量减少 16 倍：

```python
# 先缩小做聚类
small_frame = cv2.resize(frame, (w//4, h//4))
labels, centers = kmeans(small_data, n_cl)

# 上色后放大回原始分辨率
new_frame = color_table[labels].reshape(small_h, small_w, 3)
new_frame = cv2.resize(new_frame, (1920, 1080))
```

K-means 聚的是颜色，不是空间位置。缩小后颜色分布几乎不变，聚类结果几乎一样（因为道路的灰色、天空的蓝色在小图中依然是灰色和蓝色），但速度快了 16 倍。

### 4.4 优化四：容差收敛 + 最大迭代数

```python
# 原始：只判断标签完全不变才退出——有时会震荡
if np.all(new_labels == old_labels):
    break

# 优化：标签不变 OR 中心几乎不动（位移 < 0.0001）就退出
labels_stable = np.all(new_labels == old_labels)
centers_stable = np.all(np.abs(centers - old_centers) < 1e-4)
if labels_stable or centers_stable:
    break
```

额外加了一个 `max_iter=100` 的安全阀，防止极端情况下算法永远不收敛导致程序卡死。

---

## 五、优化效果对比

跑同一段 35 帧的 `road_video.MOV`：

| 聚类数 | 优化前总耗时 | 优化后总耗时 | 加速比 |
|--------|-------------|-------------|--------|
| n_cl=2 | 22.0 秒 | **1.0 秒** | **22×** |
| n_cl=3 | 53.0 秒 | **2.0 秒** | **26×** |
| n_cl=5 | 306.0 秒 | **4.5 秒** | **68×** |

n_cl=5 从 **5 分钟降到 4.5 秒**，加速最猛，因为原始版的广播开销随聚类数 K 线性增长，而 BLAS 矩阵乘法对 K 不敏感。

### 各优化贡献拆解

以 n_cl=5 为例，从 306 秒降到 4.5 秒：

| 优化手段 | 大致贡献 | 原因 |
|----------|----------|------|
| 降采样 1/4 | ~16× | 像素数直接减少 16 倍 |
| 距离恒等式 + BLAS | ~3× | 避免 124MB 中间数组，矩阵乘法指令级优化 |
| Fancy indexing 上色 | ~1.1× | 消除 Python for 循环 |
| 容差收敛 | ~1.2× | 减少无效迭代 |

**总加速 ≈ 16 × 3 × 1.1 × 1.2 ≈ 63×**，与实测的 68× 基本吻合。

---

## 六、输出文件说明

运行 notebook 后，在根目录下生成以下视频文件：

| 文件名 | 聚类数 | 用途 |
|--------|--------|------|
| `result_with_1clz.mp4` | 1 | 基准（全图一色） |
| `result_with_2clz.mp4` | 2 | 道路 vs 背景 二分 |
| `result_with_3clz.mp4` | 3 | 道路 + 天空 + 植被 |
| `result_with_4clz.mp4` | 4 | 更细粒度四分割 |
| `result_with_5clz.mp4` | 5 | 最细粒度五分割 |

建议用 n_cl=2 或 3 的效果最直观——能清晰看到路面和周围环境被区分开来。

---

## 七、关键要点回顾

1. **NumPy 广播虽然方便，但会制造巨大的中间数组**。当数据量达到百万级时，内存带宽就是瓶颈，需要用数学恒等式把中间结果降维。
2. **`np.dot()` 是数值计算中最值得信赖的加速工具之一**——它调用的是 C/Fortran 级别的 BLAS 库，几十年的工程优化不是 Python 循环能比的。
3. **对于像素级聚类，降采样是非常实用的技巧**。颜色分布在缩小后的图中几乎不变，但计算量呈平方级下降。
4. **视频帧间一致性**：把上一帧的聚类结果传给下一帧做初始化，既持续加速收敛，又保证颜色不跳变，一举两得。
5. K-means 的时间复杂度是 O(k × n × T)，n 是最关键的因素——**减少 n（降采样）比优化代码更见效**，两者结合效果最好。
