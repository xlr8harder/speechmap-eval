from __future__ import annotations

import argparse

from tools.reserve_judge_gpu import (
    create_pod_command,
    parse_created_pod_id,
    polling_delay,
    validate_args,
)


def test_parse_created_pod_id() -> None:
    output = "Successfully created pod 0123456789abcdef0123456789ABCDEF\n"
    assert parse_created_pod_id(output) == "0123456789abcdef0123456789abcdef"
    assert parse_created_pod_id("provider returned HTTP 400") is None


def test_create_command_uses_explicit_resource_shape() -> None:
    args = argparse.Namespace(
        name="judge-test",
        disk_size=180,
        vcpus=16,
        memory=64,
        image="ubuntu_22_cuda_12",
    )
    command = create_pod_command(args, "offer123")
    assert command == [
        "prime",
        "pods",
        "create",
        "--plain",
        "--id",
        "offer123",
        "--name",
        "judge-test",
        "--disk-size",
        "180",
        "--vcpus",
        "16",
        "--memory",
        "64",
        "--image",
        "ubuntu_22_cuda_12",
        "--yes",
    ]


def test_validate_args_rejects_invalid_polling() -> None:
    args = argparse.Namespace(
        rows=2120,
        max_total_cost=1.0,
        poll_interval=61,
        poll_timeout=0,
        max_create_attempts_per_poll=5,
        spot_attempts_used=0,
        max_spot_attempts=2,
    )
    assert validate_args(args) == "--poll-interval must be between 0 and 60 seconds"


def test_zero_poll_interval_is_single_shot() -> None:
    args = argparse.Namespace(poll_interval=0, poll_timeout=0)
    assert polling_delay(args, 0) is None
