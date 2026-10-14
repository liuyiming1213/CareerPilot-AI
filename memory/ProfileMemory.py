"""画像记忆：复用原项目的 Redis Hash 设计。

作用域：L4 用户画像层。以"用户"（user_id）为粒度保存长期稳定的属性，
例如目标岗位、城市、技能方向。生产环境使用 Redis 的 Hash（一个 key 下存
多个字段），Redis 不可用时降级到进程内字典（_fallback），便于本地调试。
"""

import os

# 尝试导入 redis 库；没安装时置为 None，整个类自动走内存 fallback
try:
    import redis
except ImportError:
    redis = None

# 尝试加载 .env 中的环境变量；没有 python-dotenv 时忽略，不阻塞启动
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class ProfileMemory:
    """保存稳定的用户属性，例如目标岗位、城市和技能方向。"""

    # 类级别的内存降级存储：user_id -> {字段: 值}
    # 用类变量而不是实例变量，这样不同会话对象之间也能共享同一份画像数据
    _fallback: dict[str, dict[str, str]] = {}

    def __init__(self, user_id: str, ttl: int | None = None):
        self.user_id = str(user_id)
        # Redis key 统一加 "chat_profile:" 前缀，与其它记忆的 key 命名区分
        self.key = f"chat_profile:{self.user_id}"
        # 画像有效期，默认 7 天（7*24*3600 秒），到期自动过期
        self.ttl = ttl or 7 * 24 * 3600
        self.client = None
        # 与窗口记忆保持一致：显式开启 ENABLE_REDIS_MEMORY=1 后才连接 Redis。
        # 这样没有启动 Redis 时项目仍能立即启动并使用内存 fallback。
        if redis is not None and os.getenv("ENABLE_REDIS_MEMORY", "0") == "1":
            try:
                self.client = redis.Redis(
                    host=os.getenv("REDIS_HOST", "localhost"),
                    port=int(os.getenv("REDIS_PORT", "6379")),
                    db=int(os.getenv("REDIS_DB", "0")),
                    password=os.getenv("REDIS_PASSWORD") or None,
                    # 让 Redis 返回 Python 字符串而非 bytes，省去手动解码
                    decode_responses=True,
                    # Redis 未启动时快速降级，避免 API 启动被网络连接阻塞。
                    socket_connect_timeout=0.3,
                    socket_timeout=0.3,
                )
                # 立即 ping 一次确认连接真的可用，失败则回退到内存存储
                self.client.ping()
            except Exception:
                self.client = None
        # 保证内存 fallback 中已有该用户的空字典，方便后续直接读写
        self._fallback.setdefault(self.user_id, {})

    def add(self, field: str, value: str) -> bool:
        """新增或覆盖一个画像字段。

        参数 field：字段名（如 job、city）；value：字段值。
        返回值：True 表示是"新增"（原来没有这个字段），False 表示覆盖已有字段。
        """
        if self.client:
            # Redis hset：字段已存在时返回 0，新建时返回 1，正好用它判断是否新增
            created = self.client.hset(self.key, field, value)
            # 每次写入都刷新过期时间，让"活跃用户"的画像一直存活
            self.client.expire(self.key, self.ttl)
            return created == 1
        # 内存 fallback：先判断字段是否已存在，再写入（统一转成字符串存储）
        created = field not in self._fallback[self.user_id]
        self._fallback[self.user_id][field] = str(value)
        return created

    def batch_add(self, data: dict[str, str]) -> int:
        """批量更新画像字段。

        参数 data：{字段名: 值} 字典，一次性写入多个字段。
        返回值：本次实际"新增"的字段数量。
        """
        if self.client:
            # 逐字段位置传参调用 hset（部分 redis-py 版本不支持 mapping= 关键字，
            # 会报 "wrong number of arguments for 'hset'"）。返回"本次真正新增"的字段数。
            created = 0
            for k, v in data.items():
                # Redis hset 对已存在字段返回 0、新建返回 1
                created += 1 if self.client.hset(self.key, str(k), str(v)) == 1 else 0
            # 每次写入都刷新过期时间，让"活跃用户"的画像一直存活
            self.client.expire(self.key, self.ttl)
            return created
        # 内存 fallback：统计 data 中有多少字段原先不存在，视为"新增"
        count = sum(1 for key in data if key not in self._fallback[self.user_id])
        # 统一把 key 和 value 都转成字符串后合并进 fallback 字典
        self._fallback[self.user_id].update({str(k): str(v) for k, v in data.items()})
        return count

    def get(self, field: str) -> str:
        """读取单个画像字段。

        参数 field：字段名。返回值：字段值字符串；字段不存在时返回空串。
        """
        if self.client:
            # hget 可能返回 None（字段不存在），用 or "" 统一成空串
            return self.client.hget(self.key, field) or ""
        return self._fallback[self.user_id].get(field, "")

    def load(self) -> dict[str, str]:
        """读取完整画像，返回字典。

        返回值：{字段名: 值} 字典，供 PromptBuilder 拼进提示词。
        """
        if self.client:
            # hgetall 返回 dict 类型，显式转一次保证返回类型一致
            return dict(self.client.hgetall(self.key))
        # 返回副本，避免外部修改污染内部存储
        return dict(self._fallback[self.user_id])

    def delete(self, field: str) -> bool:
        """删除一个字段。

        参数 field：要删除的字段名。返回值：True 表示确实删除了一个字段。
        """
        if self.client:
            # hdel 返回删除的字段个数，>0 说明真的删掉了
            return self.client.hdel(self.key, field) > 0
        # pop 带默认值 None，字段不存在时返回 None，存在时返回被删的值
        return self._fallback[self.user_id].pop(field, None) is not None

    def clear(self) -> None:
        """清空画像。

        删除该用户的整个画像（包括 Redis key 和内存 fallback 中的记录）。
        """
        if self.client:
            self.client.delete(self.key)
        self._fallback.pop(self.user_id, None)
