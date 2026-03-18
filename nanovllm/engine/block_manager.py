"""
KV Cache 块管理器模块

本模块定义了 BlockManager 类，负责管理 GPU 上的 KV Cache 内存块。
核心功能包括：
- 块的分配与回收
- 前缀缓存（Prefix Caching）：通过哈希去重，复用相同前缀的 KV Cache
- 引用计数：支持多个序列共享相同的 KV Cache 块
"""

from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:
    """
    KV Cache 块类
    
    表示一个物理 KV Cache 块，用于存储序列的 token 对应的 KV 值。
    每个块有固定容量（block_size，默认 256 tokens）。
    
    属性：
        block_id (int): 物理块的唯一 ID（0 到 num_blocks-1）
        ref_count (int): 引用计数，表示有多少个序列正在使用此块
        hash (int): 块的哈希值，用于前缀缓存匹配
        token_ids (list[int]): 块中存储的 token IDs，用于哈希冲突时验证
    """

    def __init__(self, block_id: int):
        """
        初始化 KV Cache 块
        
        Args:
            block_id: 物理块的唯一 ID
        """
        self.block_id = block_id      # 物理块 ID
        self.ref_count = 0            # 引用计数：0 表示空闲，>0 表示被占用
        self.hash = -1                # 哈希值：-1 表示未计算/无效
        self.token_ids = []           # 块中存储的 token IDs

    def update(self, hash: int, token_ids: list[int]):
        """
        更新块的哈希值和 token IDs
        
        当块被填充完整（达到 block_size）时调用，用于注册到哈希表中
        
        Args:
            hash: 计算得到的哈希值
            token_ids: 块中的 token IDs 列表
        """
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        """
        重置块状态，准备重新分配
        
        当块从空闲池分配出来时调用，初始化引用计数为 1（表示当前分配者持有）
        """
        self.ref_count = 1       # 引用计数重置为 1（新分配）
        self.hash = -1           # 哈希值清零
        self.token_ids = []      # token IDs 清空


class BlockManager:
    """
    KV Cache 块管理器
    
    管理所有物理 KV Cache 块，支持：
    - 块的分配与回收（基于引用计数）
    - 前缀缓存优化（通过哈希去重复用）
    - 动态扩展（当序列增长时分配新块）
    
    核心数据结构：
        - blocks: 所有物理块的数组
        - hash_to_block_id: 哈希值 → 物理块 ID 的映射（用于前缀缓存查找）
        - free_block_ids: 空闲块 ID 队列（FIFO 分配）
        - used_block_ids: 已分配块 ID 集合（用于快速查询）
    
    前缀缓存原理：
        1. 对每个完整的块计算哈希值（基于 token IDs）
        2. 相同哈希值的块可以共享 KV Cache，避免重复计算
        3. 使用引用计数管理共享块的生命周期
    """

    def __init__(self, num_blocks: int, block_size: int):
        """
        初始化块管理器
        
        Args:
            num_blocks: 物理块的总数量（决定 KV Cache 总容量）
            block_size: 每个块可容纳的 token 数量（默认 256）
        """
        self.block_size = block_size  # 块大小：每个块容纳的 token 数
        
        # 所有物理块，索引即 block_id
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        
        # 哈希值到物理块 ID 的映射，用于前缀缓存查找
        # key: 哈希值，value: block_id
        self.hash_to_block_id: dict[int, int] = dict()
        
        # 空闲块 ID 队列（FIFO），新分配时从队首取出
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        
        # 已分配块 ID 集合，用于快速判断块是否在使用中
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1) -> int:
        """
        计算 token 序列的哈希值
        
        使用 xxhash 算法（64 位），支持前缀哈希链式计算：
        - 当前块的哈希 = hash(前一块哈希 + 当前块 token_ids)
        - 这种链式结构确保相同前缀的序列产生相同哈希
        
        Args:
            token_ids: 当前块的 token IDs 列表
            prefix: 前一块的哈希值（用于链式计算，-1 表示第一块）
            
        Returns:
            64 位哈希值（整数）
        """
        h = xxhash.xxh64()
        if prefix != -1:
            # 将前缀哈希写入，形成链式哈希
            h.update(prefix.to_bytes(8, "little"))
        # 将 token IDs 转为字节串并更新哈希
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        """
        分配一个空闲块（内部方法）
        
        从空闲池中取出指定 block_id 的块，重置其状态并标记为已使用
        
        Args:
            block_id: 要分配的块 ID
            
        Returns:
            分配后的 Block 对象
            
        Raises:
            AssertionError: 如果块的引用计数不为 0（应确保分配空闲块）
        """
        block = self.blocks[block_id]
        assert block.ref_count == 0, f"Block {block_id} ref_count should be 0"
        
        # 重置块状态（ref_count=1, hash=-1, token_ids=[]）
        block.reset()
        
        # 从空闲队列移除，加入已使用集合
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        """
        回收一个块（内部方法）
        
        当块的引用计数降为 0 时，将其返回到空闲池
        
        Args:
            block_id: 要回收的块 ID
        """
        # 确保引用计数已为 0
        assert self.blocks[block_id].ref_count == 0
        
        # 从已使用集合移除，加入空闲队列（队尾）
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        """
        检查是否有足够的空闲块分配给序列
        
        Args:
            seq: 要分配块的序列
            
        Returns:
            True 如果有足够空闲块，否则 False
        """
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        """
        为序列分配 KV Cache 块
        
        核心逻辑（支持前缀缓存优化）：
        1. 遍历序列的每个逻辑块
        2. 计算块的哈希值
        3. 查找哈希表，如果命中且 token_ids 匹配，则复用该物理块（cache hit）
        4. 否则从空闲池分配新块（cache miss）
        5. 更新块的引用计数和哈希表
        
        前缀缓存命中条件：
        - 哈希值相同（快速筛选）
        - token_ids 完全匹配（避免哈希冲突）
        
        Args:
            seq: 要分配块的序列
        """
        # 确保序列尚未分配块表
        assert not seq.block_table, "Sequence already has block_table"
        
        h = -1  # 前缀哈希，初始为 -1（第一块无前缀）
        cache_miss = False  # 标记是否发生缓存未命中
        
        # 遍历序列的每个逻辑块
        for i in range(seq.num_blocks):
            # 获取第 i 个逻辑块的 token IDs
            token_ids = seq.block(i)
            
            # 计算哈希值（仅当块完整时计算，用于前缀缓存）
            if len(token_ids) == self.block_size:
                h = self.compute_hash(token_ids, h)  # 链式哈希
            else:
                h = -1  # 不完整的块，哈希无效
            
            # 查找哈希表，尝试复用已有块
            block_id = self.hash_to_block_id.get(h, -1)
            
            # 检查缓存是否命中：
            # 1. block_id == -1：哈希表中不存在
            # 2. token_ids 不匹配：哈希冲突，实际内容不同
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            
            if cache_miss:
                # 缓存未命中：从空闲池分配新块
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                # 缓存命中：复用已有块
                seq.num_cached_tokens += self.block_size  # 更新缓存 token 计数
                
                if block_id in self.used_block_ids:
                    # 块已被其他序列使用：增加引用计数（共享）
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    # 块在哈希表中但当前空闲：重新分配
                    block = self._allocate_block(block_id)
            
            # 更新块的哈希和 token_ids（如果哈希有效）
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id  # 注册到哈希表
            
            # 将物理块 ID 加入序列的块表
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        """
        回收序列占用的所有 KV Cache 块
        
        使用引用计数管理：
        - 每个块的 ref_count 减 1
        - 当 ref_count 降为 0 时，真正回收块到空闲池
        
        Args:
            seq: 要回收块的序列
        """
        # 逆序遍历块表（确保依赖关系正确）
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            
            # 引用计数减 1
            block.ref_count -= 1
            
            # 当引用计数为 0 时，真正回收块
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        
        # 重置序列的缓存相关状态
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        """
        检查是否可以追加新块
        
        当序列长度增长到需要新块时调用（例如：256→257 tokens）
        
        Args:
            seq: 要追加块的序列
            
        Returns:
            True 如果有空闲块可用，否则 False
        """
        # 当 len(seq) % block_size == 1 时，需要新块
        # 例如：block_size=256，len=257 时需要第 2 个块
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        """
        尝试为序列追加新块（如果必要）
        
        三种情况：
        1. len(seq) % block_size == 1：需要分配新块（跨块边界）
        2. len(seq) % block_size == 0：最后一个块刚好填满，计算其哈希
        3. 其他情况：块内追加，无需操作
        
        Args:
            seq: 要追加块的序列
        """
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]  # 最后一个物理块
        
        if len(seq) % self.block_size == 1:
            # 情况 1：跨块边界，需要分配新块
            # 例如：block_size=256，len 从 256→257
            assert last_block.hash != -1, "Last block should have valid hash"
            
            # 从空闲池分配新块
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
            
        elif len(seq) % self.block_size == 0:
            # 情况 2：最后一个块刚好填满，计算其哈希并注册
            # 例如：block_size=256，len=256, 512, ...
            assert last_block.hash == -1, "Last block hash should be -1 before computing"
            
            # 获取最后一个块的 token IDs
            token_ids = seq.block(seq.num_blocks - 1)
            
            # 计算前缀哈希（前一块的哈希）
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            
            # 计算当前块的哈希
            h = self.compute_hash(token_ids, prefix)
            
            # 更新块并注册到哈希表
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
            
        else:
            # 情况 3：块内追加，无需操作
            # 例如：len=257→300，仍在第 2 个块内
            assert last_block.hash == -1, "Incomplete block should have hash=-1"
