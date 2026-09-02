from hfvast.models.offers import OfferQuery
from hfvast.providers.vast.offers import SnapshotProvider, build_bundle_filters, load_snapshot, normalize_offer


def test_normalize_offer_units():
    raw = {
        "id": 12345,
        "gpu_name": "RTX A6000",
        "num_gpus": 4,
        "gpu_ram": 49140,  # MB per GPU (Vast REST units)
        "cpu_ram": 262144,
        "cpu_cores": 64,
        "disk_space": 1024,
        "disk_bw": 600,
        "inet_down": 782,
        "inet_up": 490,
        "inet_down_cost": 0.08,
        "dph_base": 1.72,
        "dph_total": 1.82,
        "storage_cost": 0.04,
        "reliability": 0.9944,
        "verified": True,
        "dlperf": 132.0,
        "gpu_mem_bw": 512,
        "pcie_bw": 15.2,
        "bw_nvlink": 0,
        "geolocation": "Falkenstein, DE",
    }
    offer = normalize_offer(raw)
    assert offer.offer_id == 12345
    assert offer.gpu_count == 4
    assert abs(offer.per_gpu_vram_gb - 48.0) < 0.1  # MB → GiB
    assert abs(offer.total_vram_gb - 192.0) < 0.2
    assert offer.cpu_ram_gb == 256.0
    assert offer.inet_down_mbps == 782
    assert offer.hourly_total_usd == 1.82
    assert offer.label == "4× RTX A6000"


def test_bundle_filters_shape():
    query = OfferQuery(
        min_total_vram_gb=195.0,
        disk_gb=232.0,
        max_gpus=4,
        min_per_gpu_vram_gb=46.0,
        min_download_mbps=300.0,
        min_reliability=0.98,
        max_hourly_usd=3.0,
    )
    filters = build_bundle_filters(query)
    assert filters["rentable"] == {"eq": True}
    assert filters["num_gpus"] == {"gte": 1, "lte": 4}
    assert filters["gpu_ram"] == {"gte": int(46.0 * 1024)}  # MB in REST API
    assert filters["disk_space"] == {"gte": 232}
    assert filters["reliability"] == {"gte": 0.98}
    assert filters["inet_down"] == {"gte": 300}
    # Caps are intentionally NOT part of the server-side query (post-ranking
    # enforcement gives us a "cheapest compatible" message instead).
    assert "dph_total" not in filters
    assert filters["type"] == "on-demand"
    assert filters["allocated_storage"] == 232
    assert filters["order"] == [["dph_total", "asc"]]


def test_snapshot_loads_and_filters():
    snapshot = load_snapshot()
    assert snapshot["_meta"]["captured_at"] is None  # explicitly not live data
    assert len(snapshot["offers"]) >= 10

    provider = SnapshotProvider()
    query = OfferQuery(
        min_total_vram_gb=190.0,
        disk_gb=232.0,
        max_gpus=4,
        min_per_gpu_vram_gb=46.0,
        min_download_mbps=300.0,
        min_reliability=0.98,
    )

    import asyncio

    offers = asyncio.run(provider.search_offers(query))
    assert offers, "4×48GB-class offers should satisfy the query"
    assert all(o.disk_gb >= query.disk_gb for o in offers)
    assert all(o.reliability >= 0.98 for o in offers)
    assert all(o.per_gpu_vram_gb >= 46.0 for o in offers)
    assert provider.data_source == "sample"
