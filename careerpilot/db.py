"""数据库连接和 Session 管理。

项目要求摘要记忆使用 PostgreSQL，因此生产环境只要设置 DATABASE_URL 为
PostgreSQL 地址，所有 SQLAlchemy 模型就会自动使用 PostgreSQL。为了让新手
可以先运行项目，本地没有配置时默认落到 data/careerpilot.db。

本文件的职责：
1. 定义所有 ORM 表的基类 Base（各模型继承它）。
2. 根据配置创建数据库引擎 engine。
3. 提供 SessionLocal（会话工厂）与 get_db（FastAPI 依赖），以及建表入口 init_database。
"""

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from careerpilot.config import settings


class Base(DeclarativeBase):
    """所有 ORM 表的基类。"""
    # 各表模型（User、Resume 等）都继承 Base；Base.metadata 汇总了全部表定义，
    # 后续建表、迁移都基于它。


# SQLite 允许多线程访问时需要关闭线程检查；PostgreSQL 不需要这个参数，传空字典即可。
connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
# 创建引擎：pool_pre_ping=True 表示每次取连接前先做一次存活探测，
# 避免拿到已经失效的连接（尤其在生产数据库重启后很常见）。
engine = create_engine(settings.database_url, pool_pre_ping=True, connect_args=connect_args)
# 会话工厂：autoflush=False（不自动把改动提前刷库）、autocommit=False（必须显式提交）、
# expire_on_commit=False（提交后对象属性仍可用，避免随后访问属性触发惰性加载报错）。
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def init_database() -> None:
    """创建当前模型对应的表。

    这里只 create_all，不执行 drop_all。这样开发服务器重启不会删除数据。
    正式项目可以再加入 Alembic 迁移，这里先保持初学者容易理解。
    """
    # 局部导入 models：确保所有模型类被导入并注册到 Base.metadata（import 本身有副作用）。
    from careerpilot import models  # noqa: F401 - 通过导入确保模型已注册到 Base

    # 按照 Base.metadata 里登记的表定义，在引擎对应的数据库里建表（已存在的表会跳过）。
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI 依赖：每个请求获取一个数据库 Session，请求结束后关闭。"""
    db = SessionLocal()
    try:
        # 把会话 yield 出去给路由函数使用；FastAPI 在请求结束后继续执行 finally 收尾。
        yield db
    finally:
        # 无论请求成功还是抛异常，都确保会话被关闭、连接被归还连接池。
        db.close()
