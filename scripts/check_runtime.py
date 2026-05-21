from __future__ import annotations

from pcos_ssl.utils.runtime import configure_runtime


def main() -> None:
    runtime = configure_runtime(seed=42, cpu_threads=18)
    print(f"torch_version={runtime.torch_version}")
    print(f"device={runtime.device_name}")
    print(f"mps_available={runtime.mps_available}")
    print(f"cpu_threads={runtime.cpu_threads}")


if __name__ == "__main__":
    main()

