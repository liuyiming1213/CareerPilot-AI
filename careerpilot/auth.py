"""认证依赖。

HTTP 请求的 Authorization 头格式为：Bearer <token>。依赖函数会验证 JWT，
并返回当前用户 ID。所有涉及简历、JD、记忆的接口都应该使用它做用户隔离。

作用域说明：
- 本模块定义 FastAPI 的"依赖（dependency）"：把 `current_user_id` 声明为路由
  参数时，FastAPI 会自动解析 Authorization 头、验证 JWT，并把当前用户 ID 注入
  到接口函数里；
- 这是多用户数据隔离的第一道闸门：所有涉及用户私有数据（简历、JD、记忆）的
  接口都必须用它拿到当前用户 ID，再按这个 ID 过滤数据。
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from careerpilot.security import decode_access_token


# HTTPBearer 用于从请求头中解析 "Authorization: Bearer <token>"。
# auto_error=False 表示"没有带凭证时不自动抛 401"，而是把 credentials 置为 None，
# 交给下面的函数自己判断并给出更友好的错误信息。
bearer = HTTPBearer(auto_error=False)


def current_user_id(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)) -> int:
    """提取并验证当前用户。

    作为 FastAPI 依赖使用：`def api(user_id: int = Depends(current_user_id))`。

    - 参数：credentials 由 Depends(bearer) 自动注入的 Bearer 凭证对象。
    - 返回：解析出的当前用户 ID（来自 JWT 里的 subject 字段）。
    - 抛出：未携带凭证或 JWT 无效/过期时抛 401 HTTPException。
    """
    # 没有带 Authorization 头或格式不对时，credentials 为 None，直接拒绝访问
    if not credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="请先登录")
    try:
        # 对 token 做签名校验 + 过期校验，成功则返回用户 ID。
        # 这一步在 security.decode_access_token 中完成。
        return decode_access_token(credentials.credentials)
    except Exception as exc:
        # 任何解密/验签/过期异常都统一转成 401，
        # 不把底层异常细节暴露给客户端（避免信息泄露）。
        # `from exc` 保留原始异常链，方便服务端日志排查。
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="登录凭证无效或已过期") from exc

