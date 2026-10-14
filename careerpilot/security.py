"""密码和 JWT 工具。

示例项目中使用标准库 hashlib 做密码哈希，避免把密码明文放进数据库。生产
环境可以进一步换成 Argon2/Bcrypt，但接口层不需要改变。

作用域说明：
- 本模块是 CareerPilot 的安全基础件，提供两类能力：
  1) 密码哈希与校验（hash_password / verify_password）：用于用户密码落库与登录校验；
  2) JWT 的签发与解析（create_access_token / decode_access_token）：用于登录后
     给用户签发令牌，以及在后续请求里验证身份。
- 上层（auth.py、API 路由）只调用这几个函数，不关心底层具体算法。
"""

import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone

import jwt

from careerpilot.config import settings


def hash_password(password: str) -> str:
    """使用随机盐和 PBKDF2 生成不可逆密码摘要。

    安全要点：
    - 每个密码都用独立的 16 字节随机盐（secrets 是加密安全随机源），
      避免相同的密码产生相同的摘要，也抵抗彩虹表攻击；
    - PBKDF2 迭代 12 万次，增加暴力破解成本；
    - 返回值按 `算法$迭代次数$盐$摘要` 的格式打包，方便以后换算法也能识别。

    - 参数：password 明文密码。
    - 返回：可直接存入数据库的摘要字符串。
    """
    # 生成 16 字节（32 个十六进制字符）的随机盐
    salt = secrets.token_hex(16)
    # PBKDF2-HMAC-SHA256：把密码和盐一起做 12 万次迭代派生，
    # 结果对攻击者不可逆；只能用 verify_password 验证
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 120_000)
    # 用 $ 分隔的编码格式把算法/迭代次数/盐/摘要存进一条字符串，
    # 以后即使升级算法，也能根据这个字符串判断当时用的是什么参数
    return f"pbkdf2$120000${salt}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """按照保存时的算法重新计算并安全比较。

    - 参数：password 待验证的明文密码；stored 数据库中保存的摘要字符串。
    - 返回：密码匹配返回 True，否则返回 False；格式非法时也返回 False。
    """
    try:
        # 把存储串按 $ 拆成 4 段：算法 / 迭代次数 / 盐 / 期望的摘要。
        # split("$", 3) 的 3 表示最多切 3 刀，避免盐里出现 $ 时被多切
        algorithm, iterations, salt, expected = stored.split("$", 3)
        # 用与保存时完全相同的参数重新计算摘要。
        # "pbkdf2" 在存储串里是算法标识，映射回 hashlib 用的 "sha256" 名称
        actual = hashlib.pbkdf2_hmac(
            algorithm.replace("pbkdf2", "sha256"), password.encode(), salt.encode(), int(iterations)
        ).hex()
        # hmac.compare_digest 是恒定时间比较，避免"短路径提前返回"带来的
        # 时序侧信道攻击（普通 == 比较可被逐字符猜出哈希）
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        # 存储串格式不对（段数不够/迭代次数不是整数等）时当作验证失败，
        # 而不是抛异常让调用方崩溃
        return False


def create_access_token(user_id: int) -> str:
    """生成包含 user_id 的 JWT。

    - 参数：user_id 要签发给哪个用户。
    - 返回：签好名的 JWT 字符串。
    """
    # 过期时间 = 当前 UTC 时间 + 配置里的有效期（分钟）
    exp = datetime.now(timezone.utc) + timedelta(minutes=settings.access_token_minutes)
    # sub 是 JWT 标准里的"主题"，这里放用户 ID 字符串；
    # exp 声明过期时间；再用 jwt_secret 签名，防止 token 被篡改
    return jwt.encode({"sub": str(user_id), "exp": exp}, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> int:
    """解析 JWT，失败时抛出异常，由 API 层转换为401。

    - 参数：token 客户端提交的 JWT。
    - 返回：JWT 中携带的用户 ID（int）。
    - 抛出：token 过期、签名不合法或被篡改时抛出 jwt 相关异常。
    """
    # jwt.decode 会自动校验签名和过期时间（exp），
    # algorithms=[...] 明确只接受配置里指定的算法，防止算法混淆攻击
    payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    # sub 里存的是字符串形式，这里转回 int 作为用户 ID 返回
    return int(payload["sub"])

