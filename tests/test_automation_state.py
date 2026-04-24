import logging
from datetime import datetime, timezone

from tg_signer.automation.models import RuleStateStore


def test_rule_state_store_roundtrip(tmp_path):
    """状态文件写入后应能按 rule/trigger 维度完整回读。"""
    path = tmp_path / "state.json"
    store = RuleStateStore(path, logging.getLogger("test"))
    dt = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)

    store.set_rule_vars("r1", {"a": 1})
    store.set_trigger_next_run("r1", "t1", dt)
    store.save(force=True)

    store2 = RuleStateStore(path, logging.getLogger("test"))
    assert store2.get_rule_vars("r1")["a"] == 1
    assert store2.get_trigger_next_run("r1", "t1") == dt
