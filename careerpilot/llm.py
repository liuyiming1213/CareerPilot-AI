"""可选的大模型客户端。

项目没有配置 DASHSCOPE_API_KEY 时，CareerPilot 仍可运行规则型 fallback；
配置密钥后，后续 Agent 可以复用 get_llm() 接入 Qwen/OpenAI 兼容接口。

本文件是"大模型接入"的唯一封装点：外部代码只调 get_llm()，不用关心具体
用哪家厂商、哪个 base_url。拿不到客户端时返回 None，调用方据此走规则兜底。
"""

import os

from careerpilot.config import settings


def get_llm():
    """返回 ChatOpenAI 兼容客户端；没有密钥时返回 None。"""
    # 没有配置 API 密钥时直接返回 None，调用方据此走规则型 fallback。
    if not settings.llm_api_key:
        return None
    try:
        # langchain_openai 提供 OpenAI 兼容协议的客户端；阿里云百炼（DashScope）
        # 也提供兼容接口，因此可以直接复用，无需单独 SDK。
        from langchain_openai import ChatOpenAI

        # 注意：这里直接读环境变量 DASHSCOPE_API_KEY（与 settings.llm_api_key 同值），
        # base_url 和模型名是硬编码的；将来建议改为读取 settings 里的对应字段，更统一。
        return ChatOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            streaming=True,  # 支持 /chat/stream 的逐 token 输出；invoke 仍会聚合完整结果
            # MaaS qwen3 系列在 thinking=true 时不接受 Agent 结构化工具选择参数。
            extra_body={"enable_thinking": settings.llm_enable_thinking},
        )
    except ImportError:
        # 没安装 langchain_openai 时同样退化为 None，保证项目仍可正常启动。
        return None
