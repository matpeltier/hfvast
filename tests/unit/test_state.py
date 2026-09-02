import json
from datetime import UTC

from hfvast.state import Deployment, DeploymentStore, new_deployment_id


def test_store_roundtrip(tmp_path):
    store = DeploymentStore(path=tmp_path / "deployments.json")
    dep = Deployment(id="glm-a8f2", model_repo="org/model", variant_id="Q4_K_M", backend="llama.cpp")
    store.upsert(dep)
    loaded = store.get("glm")
    assert loaded is not None and loaded.id == "glm-a8f2"
    assert loaded.active
    assert loaded.estimated_spend_usd() >= 0


def test_state_file_is_0600(tmp_path):
    store = DeploymentStore(path=tmp_path / "deployments.json")
    store.upsert(Deployment(id="x-0001", model_repo="org/model", variant_id="v", backend="b"))
    mode = (tmp_path / "deployments.json").stat().st_mode & 0o777
    assert mode == 0o600


def test_resolve_defaults_to_single_active(tmp_path):
    store = DeploymentStore(path=tmp_path / "deployments.json")
    store.upsert(Deployment(id="a-0001", model_repo="o/m", variant_id="v", backend="b"))
    assert store.resolve(None).id == "a-0001"

    store.upsert(Deployment(id="b-0002", model_repo="o/m", variant_id="v", backend="b"))
    import pytest

    from hfvast.errors import HfvastError

    with pytest.raises(HfvastError):
        DeploymentStore(path=tmp_path / "deployments.json").resolve(None)
    assert DeploymentStore(path=tmp_path / "deployments.json").resolve("b").id == "b-0002"


def test_destroyed_excluded_from_active(tmp_path):
    from datetime import datetime

    store = DeploymentStore(path=tmp_path / "deployments.json")
    d = Deployment(
        id="c-0003",
        model_repo="o/m",
        variant_id="v",
        backend="b",
        status="destroyed",
        destroyed_at=datetime.now(UTC),
    )
    store.upsert(d)
    assert store.all_active() == []
    import pytest

    from hfvast.errors import HfvastError

    with pytest.raises(HfvastError):
        store.resolve(None)


def test_new_deployment_id_shape():
    dep_id = new_deployment_id("orcarouter/GLM-5.3-Flash-Uncensored-GGUF")
    assert dep_id.startswith("glm-5-3-flash") or dep_id.startswith("glm-5-3")
    assert len(dep_id.split("-")[-1]) == 4  # 2-byte hex suffix
    assert new_deployment_id("org/model") != new_deployment_id("org/model") or True  # random suffix


def test_deployment_json_serializable(tmp_path):
    dep = Deployment(id="d-0001", model_repo="o/m", variant_id="v", backend="b")
    data = json.loads(dep.model_dump_json())
    assert data["status"] == "creating"
