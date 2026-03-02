from .base import BaseBrokerAdapter
from .gtja import GtjaAdapter

# 券商适配器注册表
BROKER_REGISTRY = {
    "gtja": GtjaAdapter,  # 国泰君安
}

def get_adapter(broker_name: str) -> BaseBrokerAdapter:
    """
    根据券商标识获取对应的适配器实例。
    如果未找到，抛出 ValueError。
    """
    cls = BROKER_REGISTRY.get(broker_name.lower())
    if cls is None:
        raise ValueError(
            f"Unknown broker: '{broker_name}'. "
            f"Available brokers: {list(BROKER_REGISTRY.keys())}"
        )
    return cls()
