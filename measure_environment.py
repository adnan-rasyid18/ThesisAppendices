"""
A.6 The measurement environment
=====================================================================
The timings reported in benchmark.py depend on the machine that
produced them, so the specification of that machine is recorded here.
This reads it from the running session rather than from
documentation, so that the figures given in Table 4.3 belong to the
session in which the measurements were actually taken.

The last check confirms that no hardware accelerator is attached. The
implementation is single-threaded and uses only the processor, so an
accelerator would have no effect on the measurements; the check is
included so that the runtime setting is recorded rather than assumed.
"""

import platform
import sys
import subprocess
import datetime
import zoneinfo
import re


def find(text, key):
    for line in text.splitlines():
        if line.strip().startswith(key):
            return line.split(":", 1)[1].strip()
    return "?"


if __name__ == "__main__":
    lscpu = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
    free = subprocess.run(["free", "-h"], capture_output=True, text=True).stdout
    osrel = open("/etc/os-release").read()

    print("Execution environment :", "Google Colab (CPU runtime, no accelerator)")
    print("Processor model  :", find(lscpu, "Model name"))
    print("Architecture  :", platform.machine())
    print("vCPU  :", find(lscpu, "CPU(s)"))
    print("Cores per socket  :", find(lscpu, "Core(s) per socket"))
    print("Threads per core  :", find(lscpu, "Thread(s) per core"))
    print("Memory  :", free.splitlines()[1].split()[1])
    print("Operating system  :", re.search(r'PRETTY_NAME="(.+?)"', osrel).group(1))
    print("Kernel  :", platform.release())
    print("Python version  :", sys.version.split()[0])
    print("Date  :",
          datetime.datetime.now(zoneinfo.ZoneInfo("Asia/Jakarta")).strftime("%-d %B %Y"))

    # confirm no accelerator is active
    try:
        gpu = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        gpu_present = gpu.returncode == 0
    except FileNotFoundError:
        gpu_present = False
    print("Accelerator  :",
          "GPU DETECTED - check the runtime!" if gpu_present else "none")

    # Expected output (will vary by machine/session):
    # Execution environment : Google Colab (CPU runtime, no accelerator)
    # Processor model  : Intel(R) Xeon(R) CPU @ 2.20GHz
    # Architecture  : x86_64
    # vCPU  : 2
    # Cores per socket  : 1
    # Threads per core  : 2
    # Memory  : 12Gi
    # Operating system  : Ubuntu 24.04.5 LTS
    # Kernel  : 6.6.122+
    # Python version  : 3.13.15
    # Date  : 20 September 2026
    # Accelerator  : none
