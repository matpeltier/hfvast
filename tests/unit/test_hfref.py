import pytest

from hfvast.errors import ModelInputError
from hfvast.utils.hfref import parse_model_input


def test_plain_repo_id():
    ref = parse_model_input("org/model")
    assert ref.repo_id == "org/model"
    assert ref.revision is None


def test_https_url():
    ref = parse_model_input("https://huggingface.co/orcarouter/GLM-5.3-Flash-Uncensored-GGUF")
    assert ref.repo_id == "orcarouter/GLM-5.3-Flash-Uncensored-GGUF"


def test_hf_co_short_url():
    ref = parse_model_input("https://hf.co/org/model")
    assert ref.repo_id == "org/model"


def test_url_with_revision_tree():
    ref = parse_model_input("https://huggingface.co/org/model/tree/v2.0")
    assert ref.repo_id == "org/model"
    assert ref.revision == "v2.0"


def test_at_revision():
    ref = parse_model_input("org/model@dev")
    assert ref.repo_id == "org/model"
    assert ref.revision == "dev"


def test_rejects_bad_input():
    for bad in ("", "model", "org/model/extra", "https://example.com/org/model", "org model"):
        with pytest.raises(ModelInputError):
            parse_model_input(bad)
