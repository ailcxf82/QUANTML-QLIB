"""
数据更新模块（Data Updater）
==============================
职责：从外部数据源下载最新行情，写入 Qlib provider_uri 格式。

层级位置: 外部数据源 → DataUpdater → Qlib 二进制 → DataHandlerLP

使用方式:
    from data.updater import BaoStockUpdater
    updater = BaoStockUpdater(provider_uri="D:/qlib_data/qlib_data")
    updater.run(date="2026-05-05")
"""
from .updater import BaoStockUpdater, DataUpdater

__all__ = ["DataUpdater", "BaoStockUpdater"]
