# 采样器 (Sampler) 详解

## 一、模块概述

`nanovllm/layers/sampler.py` 实现了从模型输出的 logits 概率分布中采样生成 token 的功能，使用 **Gumbel-Max 技巧**进行高效的温度采样。

### 1.1 核心功能

| 功能 | 说明 |
|------|------|
| **温度采样** | 通过温度参数控制分布的平滑程度 |
| **多项式采样** | 从概率分布中随机采样下一个 token |
| **Gumbel-Max 技巧** | 高效的采样实现，避免累积分布计算 |

### 1.2 在推理流程中的位置

```
完整推理流程:
─────────────────────────────────────────────────────────
输入 → Embedding → Decoder Layers → LM Head → Sampler → 输出
                                        │         │
                                        ▼         ▼
                                   logits     token IDs
                                   [batch,    [batch]
                                   vocab]
```

### 1.3 类结构

```python
class Sampler(nn.Module):
    """
    采样器模块
    
    从模型输出的 logits 中采样生成下一个 token。
    使用 Gumbel-Max 技巧实现高效的多项式采样。
    """
    
    # 核心方法
    __init__()              # 初始化
    forward()               # 采样前向传播 (使用 torch.compile 优化)
```

---

## 二、采样原理

### 2.1 标准多项式采样

```
标准采样方法：
1. 计算 softmax 概率：p = softmax(logits / temperature)
2. 计算累积分布：cumsum(p)
3. 生成均匀随机数：u ~ Uniform(0, 1)
4. 搜索：找到第一个 cumsum(p) > u 的位置

复杂度：O(vocab_size)
问题：需要搜索操作，效率较低
```

### 2.2 Gumbel-Max 技巧

```
Gumbel-Max 技巧：
sample = argmax(logits / temperature + Gumbel(0, 1))

其中 Gumbel(0, 1) = -log(-log(ε)), ε ~ Uniform(0, 1)

等价形式（代码实现）：
sample = argmax(softmax(logits / temperature) / g)
其中 g = -log(exponential(1))

优势：
- 只需要一次 argmax 操作
- 无需累积分布和搜索
- 完全向量化，GPU 友好
```

### 2.3 数学推导

```
目标：从多项式分布 Multinomial(π) 中采样，其中 π = softmax(θ)

标准方法：
1. 计算 π_i = exp(θ_i) / Σⱼ exp(θ_j)
2. 生成 u ~ Uniform(0, 1)
3. 找到 k 使得 Σᵢ<ₖ πᵢ < u ≤ Σᵢ≤ₖ πᵢ

Gumbel-Max 技巧：
1. 生成 Gumbel 噪声：g_i ~ Gumbel(0, 1)
2. 计算：k = argmax_i(θ_i + g_i)

证明：
P(k = argmax_i(θ_i + g_i)) = π_k = exp(θ_k) / Σⱼ exp(θ_j)

因此，argmax(θ + g) 等价于从 softmax(θ) 中采样！
```

### 2.4 Gumbel 分布

```
Gumbel 分布 (Type-I Extreme Value Distribution):

概率密度函数：
f(x) = exp(-(x + exp(-x)))

生成方法：
如果 ε ~ Uniform(0, 1)，则：
g = -log(-log(ε)) ~ Gumbel(0, 1)

代码中的等价形式：
g = 1 / exponential(1)

因为：
probs / g = probs * (1/g) = probs * exp(gumbel_noise)
log(probs / g) = log(probs) + gumbel_noise
argmax(log(probs / g)) = argmax(probs / g)
```

---

## 三、代码详解

### 3.1 完整代码

```python
class Sampler(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        """
        从 logits 中采样 token
        
        Args:
            logits: 模型输出的 logits，形状为 [batch_size, vocab_size]
            temperatures: 温度参数，形状为 [batch_size]
        
        Returns:
            sample_tokens: 采样的 token IDs，形状为 [batch_size]
        """
        # 步骤 1：应用温度缩放
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        
        # 步骤 2：计算 softmax 概率分布
        probs = torch.softmax(logits, dim=-1)
        
        # 步骤 3-4：Gumbel-Max 技巧采样
        sample_tokens = probs.div_(
            torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        ).argmax(dim=-1)
        
        return sample_tokens
```

### 3.2 逐步解析

#### 步骤 1：温度缩放

```python
logits = logits.float().div_(temperatures.unsqueeze(dim=1))
```

**图示**：
```
logits:     [batch_size, vocab_size]
            └─ 例如：[4, 32000]

temperatures: [batch_size]
              └─ 例如：[4]

temperatures.unsqueeze(dim=1): [batch_size, 1]
                               └─ 例如：[4, 1]

广播除法：
[4, 32000] / [4, 1] → [4, 32000]

每个序列的 logits 除以自己的温度值
```

**温度的作用**：
```
temperature > 1:  分布更平滑，增加随机性
                  ┌─────────────────────────────┐
                  │  原始：[0.6, 0.3, 0.1]      │
                  │  T=2:  [0.4, 0.35, 0.25]    │ ← 更均匀
                  └─────────────────────────────┘

temperature < 1:  分布更尖锐，减少随机性
                  ┌─────────────────────────────┐
                  │  原始：[0.6, 0.3, 0.1]      │
                  │  T=0.5: [0.8, 0.15, 0.05]   │ ← 更集中
                  └─────────────────────────────┘

temperature = 1:  原始分布
```

#### 步骤 2：Softmax 概率

```python
probs = torch.softmax(logits, dim=-1)
```

**计算**：
```
logits: [2.0, 1.0, 0.0]

softmax:
p₀ = exp(2) / (exp(2) + exp(1) + exp(0)) = 7.39 / 10.15 = 0.728
p₁ = exp(1) / (exp(2) + exp(1) + exp(0)) = 2.72 / 10.15 = 0.268
p₂ = exp(0) / (exp(2) + exp(1) + exp(0)) = 1.00 / 10.15 = 0.098

probs: [0.728, 0.268, 0.098]
验证：0.728 + 0.268 + 0.098 = 1.0 ✓
```

#### 步骤 3-4：Gumbel-Max 采样

```python
sample_tokens = probs.div_(
    torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
).argmax(dim=-1)
```

**逐步分解**：
```python
# 1. 生成指数分布噪声
gumbel_noise = torch.empty_like(probs).exponential_(1)
# 例如：[0.5, 2.3, 0.1]

# 2. 防止除零（数值稳定性）
gumbel_noise = gumbel_noise.clamp_min_(1e-10)
# 最小值为 1e-10

# 3. 概率除以噪声
scores = probs / gumbel_noise
# 例如：[0.728/0.5, 0.268/2.3, 0.098/0.1]
#      = [1.456, 0.117, 0.98]

# 4. 取 argmax
sample = argmax(scores)
# argmax([1.456, 0.117, 0.98]) = 0
# 采样得到 token 0
```

**为什么这样等价于多项式采样？**

```
数学原理：
如果 g ~ Exponential(1)，则 1/g 服从 Fréchet 分布

P(argmax_i(probs_i / g_i) = k) = probs_k

因此，argmax(probs / g) 等价于从 probs 中多项式采样！
```

---

## 四、温度参数详解

### 4.1 温度的数学作用

```
softmax 公式：
p_i = exp(logits_i / T) / Σⱼ exp(logits_j / T)

温度 T 的影响：
```

### 4.2 温度效果对比

```
假设 logits = [3.0, 2.0, 1.0]

T = 0.1 (极低温度):
─────────────────────────────────────────────────────────
logits / T = [30, 20, 10]
probs = [0.99995, 0.00005, 0.0]
效果：几乎确定选择 token 0（贪婪解码）

T = 0.5 (低温):
─────────────────────────────────────────────────────────
logits / T = [6, 4, 2]
probs = [0.88, 0.12, 0.01]
效果：高度倾向于 token 0，但仍有小概率选择其他

T = 1.0 (标准温度):
─────────────────────────────────────────────────────────
logits / T = [3, 2, 1]
probs = [0.67, 0.24, 0.09]
效果：标准多项式采样

T = 1.5 (高温):
─────────────────────────────────────────────────────────
logits / T = [2, 1.33, 0.67]
probs = [0.51, 0.26, 0.23]
效果：分布更均匀，随机性增加

T = 5.0 (极高温度):
─────────────────────────────────────────────────────────
logits / T = [0.6, 0.4, 0.2]
probs = [0.38, 0.33, 0.29]
效果：接近均匀分布，非常随机
```

### 4.3 温度选择建议

| 应用场景 | 推荐温度 | 原因 |
|---------|---------|------|
| **事实问答** | 0.1 - 0.3 | 需要准确性，接近贪婪解码 |
| **代码生成** | 0.2 - 0.5 | 需要语法正确性 |
| **创意写作** | 0.7 - 1.0 | 需要多样性和创造力 |
| **对话聊天** | 0.6 - 0.9 | 平衡有趣性和连贯性 |
| **头脑风暴** | 1.0 - 1.5 | 需要高度多样性 |

---

## 五、完整采样流程

### 5.1 从 logits 到 token

```mermaid
flowchart LR
    subgraph 输入
        Logits[logits<br/>batch × vocab]
        Temp[temperatures<br/>batch]
    end
    
    subgraph 温度缩放
        Div[除以温度]
        Scaled[scaled_logits<br/>batch × vocab]
    end
    
    subgraph 概率计算
        Softmax[Softmax]
        Probs[probs<br/>batch × vocab]
    end
    
    subgraph Gumbel 噪声
        Exp[Exponential(1)]
        Clamp[Clamp Min]
        Noise[gumbel_noise<br/>batch × vocab]
    end
    
    subgraph 采样
        Div2[概率 / 噪声]
        Scores[scores<br/>batch × vocab]
        ArgMax[argmax]
        Tokens[tokens<br/>batch]
    end
    
    Logits --> Div
    Temp --> Div
    Div --> Scaled --> Softmax --> Probs
    Probs --> Div2
    Exp --> Clamp --> Noise --> Div2
    Div2 --> Scores --> ArgMax --> Tokens
```

### 5.2 数值示例

```
输入：
logits = [[3.0, 2.0, 1.0],    # 序列 0
          [1.0, 3.0, 2.0]]    # 序列 1
temperatures = [0.5, 1.0]

步骤 1：温度缩放
─────────────────────────────────────────────────────────
序列 0: logits / 0.5 = [6.0, 4.0, 2.0]
序列 1: logits / 1.0 = [1.0, 3.0, 2.0]

步骤 2：Softmax
─────────────────────────────────────────────────────────
序列 0: probs = [0.88, 0.12, 0.01]
序列 1: probs = [0.12, 0.67, 0.21]

步骤 3：生成 Gumbel 噪声
─────────────────────────────────────────────────────────
序列 0: noise = [0.3, 1.5, 0.8]
序列 1: noise = [2.1, 0.4, 1.2]

步骤 4：计算 scores = probs / noise
─────────────────────────────────────────────────────────
序列 0: scores = [0.88/0.3, 0.12/1.5, 0.01/0.8]
              = [2.93, 0.08, 0.01]
序列 1: scores = [0.12/2.1, 0.67/0.4, 0.21/1.2]
              = [0.06, 1.68, 0.18]

步骤 5：ArgMax
─────────────────────────────────────────────────────────
序列 0: argmax([2.93, 0.08, 0.01]) = 0
序列 1: argmax([0.06, 1.68, 0.18]) = 1

输出：
tokens = [0, 1]
```

---

## 六、关键问题解答

### ❓ 多次相同的 prompt 生成过程会不会生成相同的回答？

**答案：不一定，取决于温度参数。**

#### 情况 1：temperature = 0（或极低，如 0.01）

```
行为：接近贪婪解码（Greedy Decoding）
结果：每次生成相同的输出

原因：
- 温度极低时，softmax 分布趋近于 one-hot
- 概率最高的 token 几乎总是被选中
- 采样变成确定性过程

示例：
prompt: "Hello, how are"
输出 1: "Hello, how are you?"
输出 2: "Hello, how are you?"
输出 3: "Hello, how are you?"
        └─ 完全相同
```

#### 情况 2：temperature > 0（如 0.6-1.0）

```
行为：多项式采样（Multinomial Sampling）
结果：每次生成可能不同

原因：
- 从概率分布中随机采样
- 即使 logits 相同，采样结果也可能不同
- 这是真正的随机过程

示例：
prompt: "Hello, how are"
temperature = 0.8

输出 1: "Hello, how are you today?"
输出 2: "Hello, how are you doing?"
输出 3: "Hello, how are you?"
        └─ 可能不同
```

#### 实验验证

```python
import torch
from nanovllm.layers.sampler import Sampler

sampler = Sampler()

# 相同的 logits（模拟相同 prompt 的输出）
logits = torch.randn(1, 32000)  # [1, vocab_size]

# 不同温度下的采样结果
print("Temperature = 0.1:")
for i in range(5):
    temps = torch.tensor([0.1])
    token = sampler(logits.clone(), temps)
    print(f"  Sample {i+1}: {token.item()}")
# 可能输出：[1234, 1234, 1234, 1234, 1234] ← 几乎相同

print("\nTemperature = 0.8:")
for i in range(5):
    temps = torch.tensor([0.8])
    token = sampler(logits.clone(), temps)
    print(f"  Sample {i+1}: {token.item()}")
# 可能输出：[1234, 5678, 1234, 9012, 3456] ← 可能不同

print("\nTemperature = 1.5:")
for i in range(5):
    temps = torch.tensor([1.5])
    token = sampler(logits.clone(), temps)
    print(f"  Sample {i+1}: {token.item()}")
# 可能输出：[1234, 5678, 9012, 3456, 7890] ← 更加多样
```

#### 为什么会有随机性？

```
随机性来源：Gumbel 噪声

sampler.py 代码：
torch.empty_like(probs).exponential_(1)
                    └─ 生成指数分布随机数

每次调用都会生成新的随机噪声：
Run 1: noise = [0.5, 2.3, 0.1, ...]
Run 2: noise = [1.2, 0.8, 3.5, ...]
Run 3: noise = [0.3, 1.9, 0.7, ...]

不同的噪声导致不同的采样结果
```

#### 如何获得确定性输出？

**方法 1：使用极低温度**
```python
temperatures = torch.tensor([0.01])  # 接近贪婪解码
```

**方法 2：使用贪婪解码（修改 sampler）**
```python
def greedy_decode(logits):
    return logits.argmax(dim=-1)  # 直接取最大概率的 token
```

**方法 3：设置随机种子**
```python
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# 现在每次运行都会得到相同的"随机"结果
# 因为随机数生成器被重置到相同状态
```

#### 实际应用中的考虑

| 应用 | 是否需要确定性 | 推荐方案 |
|------|---------------|---------|
| **模型测试** | 是 | temperature=0.01 或固定随机种子 |
| **产品演示** | 是 | 固定随机种子 |
| **创意写作** | 否 | temperature=0.8-1.0 |
| **对话系统** | 否 | temperature=0.7-0.9 |
| **代码生成** | 部分 | temperature=0.2-0.5 |

---

## 七、性能优化

### 7.1 Torch Compile

```python
@torch.compile
def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
    ...
```

**优势**：
- JIT 编译优化
- 融合多个操作
- 减少 Python 开销
- 加速 20-50%

### 7.2 原地操作

```python
logits = logits.float().div_(temperatures.unsqueeze(dim=1))
                    └─ 原地除法，减少内存分配

probs = torch.softmax(logits, dim=-1)

sample_tokens = probs.div_(...)  # 原地除法
                    └─ 复用 probs 内存
```

**优势**：
- 减少内存分配
- 降低内存带宽压力
- 提升缓存效率

### 7.3 数值稳定性

```python
.clamp_min_(1e-10)
```

**作用**：
- 防止除零错误
- 保证数值稳定性
- 不影响采样结果（概率极小）

---

## 八、与其他采样方法的对比

### 8.1 采样方法对比

| 方法 | 公式 | 随机性 | 速度 | 质量 |
|------|------|--------|------|------|
| **贪婪解码** | `argmax(logits)` | 无 | 最快 | 一般 |
| **Gumbel-Max** | `argmax(logits/T + gumbel)` | 高 | 快 | 好 |
| **Top-K 采样** | `multinomial(top_k(logits))` | 中 | 中 | 好 |
| **Nucleus 采样** | `multinomial(top_p(logits))` | 中 | 中 | 最好 |

### 8.2 扩展：Top-K 采样

```python
def top_k_sampling(logits, k=50, temperature=0.8):
    # 1. 温度缩放
    logits = logits / temperature
    
    # 2. 只保留 top-k 个 logits
    indices_to_remove = logits < torch.topk(logits, k)[0][..., -1, None]
    logits[indices_to_remove] = float('-inf')
    
    # 3. 多项式采样
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

### 8.3 扩展：Nucleus (Top-P) 采样

```python
def nucleus_sampling(logits, p=0.9, temperature=0.8):
    # 1. 温度缩放
    logits = logits / temperature
    
    # 2. 排序并计算累积概率
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    
    # 3. 移除累积概率超过 p 的部分
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    # 4. 还原原始顺序并屏蔽
    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove
    )
    logits[indices_to_remove] = float('-inf')
    
    # 5. 多项式采样
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

---

## 九、总结

### 9.1 核心要点

| 概念 | 说明 |
|------|------|
| **Gumbel-Max 技巧** | 高效的 multinomial 采样，避免累积分布计算 |
| **温度参数** | 控制分布平滑度，T>1 增加随机性，T<1 减少随机性 |
| **随机性来源** | Gumbel 噪声（指数分布随机数） |
| **确定性输出** | 使用极低温度或固定随机种子 |

### 9.2 关键公式

```python
# 温度缩放
scaled_logits = logits / temperature

# Softmax 概率
probs = softmax(scaled_logits)

# Gumbel-Max 采样
gumbel_noise = -log(exponential(1))
sample = argmax(probs / gumbel_noise)
      = argmax(scaled_logits + gumbel_noise)
```

### 9.3 使用建议

```python
# 创意写作
sampler(logits, temperatures=0.8)

# 事实问答
sampler(logits, temperatures=0.2)

# 代码生成
sampler(logits, temperatures=0.3)

# 需要确定性输出
sampler(logits, temperatures=0.01)
# 或设置随机种子
torch.manual_seed(42)
```

### 9.4 与其他模块的关系

```
完整推理链：
─────────────────────────────────────────────────────────
输入 → VocabParallelEmbedding → Qwen3DecoderLayer × N
                                    │
                                    ▼
                              ParallelLMHead
                                    │
                                    ▼
                               logits (rank=0)
                                    │
                                    ▼
                                  Sampler
                                    │
                                    ▼
                              next_token_ids
```

</content>