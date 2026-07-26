import os
import stat
import subprocess
import textwrap
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = REPO_ROOT / "examples" / "launch-gce.sh"


def write_executable(path: Path, content: str) -> None:
    path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def fake_cloud_env(tmp_path: Path, **overrides: str) -> dict[str, str]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log_path = tmp_path / "gcloud.log"

    write_executable(
        fake_bin / "gcloud",
        r"""
        #!/usr/bin/env bash
        set -euo pipefail

        echo "gcloud $*" >> "$GCE_FAKE_LOG"

        if [[ "$1" == "auth" && "$2" == "list" ]]; then
          echo "${GCE_FAKE_ACCOUNT:-tester@example.com}"
          exit 0
        fi
        if [[ "$1" == "auth" && "$2" == "activate-service-account" ]]; then
          exit 0
        fi
        if [[ "$1" == "config" && "$2" == "get-value" && "$3" == "project" ]]; then
          echo "${GCE_FAKE_PROJECT:-test-project}"
          exit 0
        fi
        if [[ "$1" == "billing" && "$2" == "projects" && "$3" == "describe" ]]; then
          echo "${GCE_FAKE_BILLING_ENABLED:-True}"
          exit 0
        fi
        if [[ "$1" == "services" && "$2" == "list" ]]; then
          printf '%s\n' "${GCE_FAKE_SERVICES:-compute.googleapis.com}"
          exit 0
        fi
        if [[ "$1" == "compute" && "$2" == "zones" && "$3" == "describe" ]]; then
          exit 0
        fi
        if [[ "$1" == "compute" && "$2" == "regions" && "$3" == "describe" ]]; then
          cat <<'JSON'
        {"quotas":[
          {"metric":"NVIDIA_L4_GPUS","limit":2.0,"usage":0.0},
          {"metric":"NVIDIA_T4_GPUS","limit":2.0,"usage":0.0}
        ]}
        JSON
          exit 0
        fi
        if [[ "$1" == "compute" && "$2" == "instances" && "$3" == "list" ]]; then
          printf '%b' "${GCE_FAKE_INSTANCES:-}"
          exit 0
        fi
        if [[ "$1" == "compute" && "$2" == "instances" && "$3" == "create" ]]; then
          exit 0
        fi
        if [[ "$1" == "compute" && "$2" == "instances" && "$3" == "describe" ]]; then
          echo "${GCE_FAKE_IP:-203.0.113.10}"
          exit 0
        fi
        if [[ "$1" == "compute" && "$2" == "instances" && "$3" == "delete" ]]; then
          exit 0
        fi
        if [[ "$1" == "compute" && "$2" == "firewall-rules" && "$3" == "describe" ]]; then
          [[ "${GCE_FAKE_FIREWALL_EXISTS:-0}" == "1" ]]
          exit $?
        fi
        if [[ "$1" == "compute" && "$2" == "firewall-rules" && "$3" == "create" ]]; then
          exit 0
        fi
        if [[ "$1" == "compute" && "$2" == "firewall-rules" && "$3" == "delete" ]]; then
          exit 0
        fi

        echo "unexpected gcloud invocation: $*" >&2
        exit 64
        """,
    )
    write_executable(
        fake_bin / "curl",
        r"""
        #!/usr/bin/env bash
        set -euo pipefail

        echo "curl $*" >> "$GCE_FAKE_LOG"
        exit 0
        """,
    )

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
            "GCE_FAKE_LOG": str(log_path),
        }
    )
    env.update(overrides)
    return env


def run_launcher(args: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(LAUNCHER), *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def cloud_log(env: dict[str, str]) -> str:
    return Path(env["GCE_FAKE_LOG"]).read_text(encoding="utf-8")


def test_cli_validation_fails_before_cloud_preflight() -> None:
    cases = [
        (["--ttl"], "ERROR: --ttl requires a value."),
        (["--mode", "--ttl", "2"], "ERROR: --mode requires a value."),
        (["--bogus"], "ERROR: unknown option: --bogus"),
        (["--version", "latest"], "ERROR: unsupported image version 'latest'."),
        (["--version", "v0.3.0"], "ERROR: unsupported image version 'v0.3.0'."),
        (["--no-public-ip"], "ERROR: --no-public-ip is not supported"),
    ]

    for args, expected_error in cases:
        result = run_launcher(args)

        assert result.returncode != 0
        combined_output = result.stdout + result.stderr
        assert expected_error in combined_output
        assert "unbound variable" not in combined_output
        assert "gcloud" not in combined_output


def test_stream_deploy_uses_pinned_image_public_port_and_locked_vm_credentials(tmp_path: Path) -> None:
    env = fake_cloud_env(tmp_path)

    result = run_launcher(["--mode", "stream", "--version", "0.3.0", "--ttl", "2"], env=env)

    assert result.returncode == 0, result.stderr
    assert "URL:     http://203.0.113.10:8001" in result.stdout
    assert "python3 benchmarks/scripts/bench_stream.py" in result.stdout
    assert "--server ws://203.0.113.10:8001" in result.stdout

    log = cloud_log(env)
    assert "gcloud compute instances create hpfasr-eval-" in log
    assert "ghcr.io/beastoin/highperfasr-stream:0.3.0" in log
    assert "PORT=8001" in log
    assert "-p \"${PORT}:8000\"" in log
    assert "-e HPFASR_CORS_ORIGINS='*'" in log
    assert "--no-service-account" in log
    assert "--no-scopes" in log


def test_batch_deploy_uses_batch_image_and_batch_port(tmp_path: Path) -> None:
    env = fake_cloud_env(tmp_path)

    result = run_launcher(["--mode", "batch", "--gpu", "t4"], env=env)

    assert result.returncode == 0, result.stderr
    assert "URL:     http://203.0.113.10:8000" in result.stdout
    assert "python3 benchmarks/scripts/bench_batch.py" in result.stdout
    assert "--server http://203.0.113.10:8000" in result.stdout

    log = cloud_log(env)
    assert "ghcr.io/beastoin/highperfasr-batch:0.3.0" in log
    assert "PORT=8000" in log
    assert "--machine-type=n1-standard-4" in log
    assert "--accelerator=type=nvidia-tesla-t4,count=1" in log


def test_billing_disabled_fails_before_api_quota_or_vm_checks(tmp_path: Path) -> None:
    env = fake_cloud_env(tmp_path, GCE_FAKE_BILLING_ENABLED="False")

    result = run_launcher(["--mode", "stream"], env=env)

    assert result.returncode != 0
    assert "ERROR: billing is not enabled for project test-project." in result.stderr
    assert "GPU VMs require an active billing account." in result.stderr
    assert "https://console.cloud.google.com/billing/linkedaccount?project=test-project" in result.stderr

    log = cloud_log(env)
    assert "gcloud billing projects describe test-project --format=value(billingEnabled)" in log
    assert "gcloud services list" not in log
    assert "gcloud compute zones describe" not in log
    assert "gcloud compute regions describe" not in log
    assert "gcloud compute instances create" not in log


def test_teardown_deletes_existing_eval_vm_and_firewall(tmp_path: Path) -> None:
    env = fake_cloud_env(
        tmp_path,
        GCE_FAKE_INSTANCES="hpfasr-eval-123\tus-central1-a\t203.0.113.10\tRUNNING\n",
        GCE_FAKE_FIREWALL_EXISTS="1",
    )

    result = run_launcher(["teardown"], env=env)

    assert result.returncode == 0, result.stderr
    assert "Deleting hpfasr-eval-123 (us-central1-a, RUNNING)" in result.stdout
    assert "Deleting firewall rule allow-highperfasr-eval" in result.stdout

    log = cloud_log(env)
    assert "gcloud compute instances delete hpfasr-eval-123" in log
    assert "gcloud compute firewall-rules delete allow-highperfasr-eval" in log
