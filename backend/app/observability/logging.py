import logging


def configure_logging() -> None:
    """配置结构简洁的应用日志，不记录请求正文、密钥或私密文档。"""
    # 只定义时间、级别、模块和消息，避免自动展开整个请求对象。
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
