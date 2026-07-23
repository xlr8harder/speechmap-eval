from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.select_judge_gpu import normalize_gpu_type, rank_offers


def profile_document(*profiles):
    return {"schema_version": 1, "profiles": list(profiles)}


def profile(gpu="H100 80GB", *, rows_per_second=2.0, confidence="measured"):
    return {
        "profile_id": gpu.lower().replace(" ", "_"),
        "gpu_type_regex": f"^{gpu}$",
        "socket": "SXM5",
        "source_provider": "provider-a",
        "rows_per_second": rows_per_second,
        "startup_seconds": {"cold": 600, "warm": 100, "none": 0},
        "provider_startup_seconds": {"provider-a": {"cold": 500, "warm": 80, "none": 0}},
        "confidence": confidence,
    }


def offer(*, price=2.0, spot=False, provider="provider-a", gpu="H100 80GB", socket="SXM5"):
    return {
        "id": "offer-1",
        "gpu_type": f"{gpu} (Spot)" if spot else gpu,
        "gpu_count": 1,
        "gpu_memory": 80,
        "socket": socket,
        "provider": provider,
        "location": "US",
        "stock_status": "Available",
        "price_value": price,
        "is_spot": spot,
    }


def test_normalize_gpu_type_only_removes_spot_decoration():
    assert normalize_gpu_type("  H100  80GB (Spot) ") == "H100 80GB"
    assert normalize_gpu_type("RTX PRO 6000B 96GB") == "RTX PRO 6000B 96GB"


def test_cost_separates_inference_and_startup():
    ranked, _ = rank_offers(
        {"gpu_resources": [offer()]},
        profile_document(profile()),
        rows=3600,
        startup_mode="cold",
        allow_spot=True,
        spot_attempts_used=0,
        max_spot_attempts=2,
    )
    result = ranked[0]
    assert result.inference_seconds == 1800
    assert result.startup_seconds == 500
    assert result.marginal_cost == pytest.approx(1.0)
    assert result.startup_cost == pytest.approx(2 * 500 / 3600)


def test_spot_is_allowed_twice_then_excluded():
    availability = {"gpu_resources": [offer(price=0.5, spot=True), offer(price=2.0)]}
    profiles = profile_document(profile())

    first, _ = rank_offers(
        availability,
        profiles,
        rows=100,
        startup_mode="none",
        allow_spot=True,
        spot_attempts_used=1,
        max_spot_attempts=2,
    )
    assert first[0].offer["is_spot"] is True

    fallback, stats = rank_offers(
        availability,
        profiles,
        rows=100,
        startup_mode="none",
        allow_spot=True,
        spot_attempts_used=2,
        max_spot_attempts=2,
    )
    assert fallback[0].offer["is_spot"] is False
    assert stats["spot_policy_excluded"] == 1


def test_estimated_and_wrong_socket_profiles_are_not_used_by_default():
    availability = {"gpu_resources": [offer(socket="PCIe")]}
    ranked, stats = rank_offers(
        availability,
        profile_document(profile(confidence="estimated")),
        rows=100,
        startup_mode="none",
        allow_spot=True,
        spot_attempts_used=0,
        max_spot_attempts=2,
    )
    assert ranked == []
    assert stats["missing_profile"] == 1


def test_failed_offer_or_provider_can_be_excluded_during_reselection():
    availability = {
        "gpu_resources": [
            offer(price=0.5, provider="provider-a"),
            {**offer(price=0.7, provider="provider-b"), "id": "offer-2"},
            {**offer(price=0.9, provider="provider-c"), "id": "offer-3"},
        ]
    }
    profiles = profile_document(profile())
    ranked, stats = rank_offers(
        availability,
        profiles,
        rows=100,
        startup_mode="none",
        allow_spot=True,
        spot_attempts_used=0,
        max_spot_attempts=2,
        excluded_offer_ids=frozenset({"offer-1"}),
        excluded_providers=frozenset({"provider-b"}),
    )
    assert [item.offer["id"] for item in ranked] == ["offer-3"]
    assert stats["explicitly_excluded"] == 2


def test_total_cost_ceiling_filters_expensive_offers():
    ranked, stats = rank_offers(
        {"gpu_resources": [offer(price=2.0)]},
        profile_document(profile()),
        rows=3600,
        startup_mode="cold",
        allow_spot=True,
        spot_attempts_used=0,
        max_spot_attempts=2,
        max_total_cost=1.2,
    )
    assert ranked == []
    assert stats["cost_threshold_excluded"] == 1
