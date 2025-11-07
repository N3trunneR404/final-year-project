#!/usr/bin/env python3
"""
Generate N heterogeneous node descriptors into ./nodes/
Usage:
  python3 sim/gen_nodes.py --count 100 --seed 42
"""
import argparse, random, math
from pathlib import Path
from typing import Dict, Any, List
import yaml

# -----------------------------
# Tunables
# -----------------------------
DEVICE_CLASS_WEIGHTS = {
    "phone":        0.06,
    "sbc":          0.22,  # Pi, Jetson, RK3588, VisionFive
    "laptop":       0.12,
    "workstation":  0.18,
    "gaming_rig":   0.18,
    "server":       0.16,
    "hpc":          0.08,
}

ARCH_WEIGHTS_BY_CLASS = {
    "phone":       {"arm64": 0.98, "amd64": 0.01, "riscv64": 0.01},
    "sbc":         {"arm64": 0.85, "riscv64": 0.1, "amd64": 0.05},
    "laptop":      {"amd64": 0.85, "arm64": 0.14, "riscv64": 0.01},
    "workstation": {"amd64": 0.95, "arm64": 0.04, "riscv64": 0.01},
    "gaming_rig":  {"amd64": 0.98, "arm64": 0.02, "riscv64": 0.00},
    "server":      {"amd64": 0.7, "arm64": 0.29, "riscv64": 0.01},
    "hpc":         {"amd64": 0.7, "arm64": 0.25, "riscv64": 0.05},
}

WIFI_STANDARDS = ["802.11ac", "802.11ax", "802.11be"]
FABRICS = ["ethernet", "infiniband", "wifi", "lte"]
FORMATS = ["native", "wasm", "cuda", "npu", "fpga", "asic"]

# Some realistic accelerator “catalog” entries (sim-friendly)
GPU_CATALOG = [
    # vendor, model, vram, accel_score, compute_capability, hbm_gb, hbm_bw_tbps, nvlink, tdp_w
    ("nvidia", "RTX-3060",      12, 4.0,  "8.6", 0,   0.0, False, 170),
    ("nvidia", "RTX-4060",       8, 4.5,  "8.9", 0,   0.0, False, 115),
    ("nvidia", "RTX-5060-Ti",   16, 8.0,  "9.0", 0,   0.0, False, 160),
    ("nvidia", "RTX-A2000",     12, 5.5,  "8.6", 0,   0.0, False,  70),
    ("nvidia", "Jetson Orin",    8, 4.0,  "8.7", 0,   0.0, False,  15),
    ("amd",    "RX-6800",       16, 6.0,  "",    0,   0.0, False, 250),
    ("nvidia", "H100-SXM",      80, 50.0, "9.0", 80,  3.35, True,  700),
    ("amd",    "MI300X",       192, 48.0, "",   192,  5.32, False, 750),
]

NPU_KINDS = [("coral_usb", 4), ("ncs2", 4), ("ascend", 20)]
FPGA_CARDS = ["alveo_u55c", "agilex7", "other"]
ASIC_KINDS = ["tpu_v4", "inferentia2", "trainium", "gaudi2"]

# -----------------------------
def choice_weighted(d: Dict[str, float]) -> str:
    items = list(d.items())
    r = random.random() * sum(w for _, w in items)
    s = 0.0
    for k, w in items:
        s += w
        if r <= s:
            return k
    return items[-1][0]

def rnd(a: float, b: float, nd: int = 2) -> float:
    return round(random.uniform(a, b), nd)

def pick_arch(cls: str) -> str:
    return choice_weighted(ARCH_WEIGHTS_BY_CLASS[cls])

def maybe(p: float) -> bool:
    return random.random() < p

def pick_gpu(for_class: str):
    # Higher chance of serious GPUs on gaming/server/hpc
    pool = GPU_CATALOG.copy()
    if for_class in ["gaming_rig", "server", "hpc", "workstation", "sbc"]:
        return random.choice(pool)
    return None

def pick_npu(for_class: str):
    if for_class in ["sbc", "phone", "laptop"] and maybe(0.25):
        return random.choice(NPU_KINDS)
    return None

def pick_fpga(for_class: str):
    if for_class in ["server", "hpc"] and maybe(0.15):
        return random.choice(FPGA_CARDS)
    return None

def pick_asic(for_class: str):
    if for_class in ["server", "hpc"] and maybe(0.10):
        return random.choice(ASIC_KINDS)
    return None

def pick_fabric(for_class: str):
    if for_class in ["hpc"]:
        return "infiniband"
    if for_class in ["server", "workstation"]:
        return "ethernet"
    if for_class in ["phone", "laptop", "sbc", "gaming_rig"]:
        return random.choice(["ethernet", "wifi"])
    return "ethernet"

def cpu_profile(for_class: str, arch: str):
    # cores, base, turbo, tdp
    if for_class == "phone":
        cores = random.choice([4, 6, 8])
        return cores, rnd(1.8, 2.6), rnd(2.4, 3.0), rnd(4, 10)
    if for_class == "sbc":
        cores = random.choice([4, 6, 8])
        return cores, rnd(1.5, 2.6), rnd(2.0, 2.8), rnd(6, 20)
    if for_class == "laptop":
        cores = random.choice([8, 12, 16])
        return cores, rnd(2.0, 3.0), rnd(3.5, 5.0), rnd(15, 65)
    if for_class in ["workstation", "gaming_rig"]:
        cores = random.choice([8, 12, 16, 24])
        return cores, rnd(2.8, 4.2), rnd(4.5, 5.7), rnd(65, 200)
    if for_class == "server":
        if arch == "arm64":
            cores = random.choice([64, 80, 96])
        else:
            cores = random.choice([24, 32, 48, 64])
        return cores, rnd(2.2, 3.0), rnd(3.2, 4.0), rnd(120, 300)
    if for_class == "hpc":
        cores = random.choice([48, 64, 96, 128])
        return cores, rnd(2.2, 3.2), rnd(3.0, 4.0), rnd(180, 350)
    # fallback
    return 4, 2.5, 3.5, 35

def mem_profile(for_class: str):
    if for_class == "phone":
        return random.choice([4, 6, 8, 12]), False, None
    if for_class == "sbc":
        return random.choice([2, 4, 8, 16]), False, None
    if for_class == "laptop":
        return random.choice([8, 16, 32]), False, "ddr5"
    if for_class in ["workstation", "gaming_rig"]:
        return random.choice([16, 32, 64, 128]), False, "ddr5"
    if for_class == "server":
        return random.choice([64, 128, 256, 512]), maybe(0.6), "ddr5"
    if for_class == "hpc":
        return random.choice([128, 256, 512, 1024]), True, "ddr5"
    return 8, False, None

def storage_profile(for_class: str):
    if for_class == "phone":
        return "emmc", random.choice([64, 128, 256]), None
    if for_class == "sbc":
        if maybe(0.5):
            return "microsd", random.choice([32, 64, 128]), None
        else:
            return "nvme", random.choice([128, 256, 512]), random.choice(["pcie3", "pcie4"])
    if for_class == "laptop":
        return "nvme", random.choice([256, 512, 1024]), "pcie4"
    if for_class in ["workstation", "gaming_rig"]:
        return "nvme", random.choice([512, 1024, 2048]), random.choice(["pcie4", "pcie5"])
    if for_class in ["server", "hpc"]:
        t = random.choice(["nvme", "zns", "sata"])
        size = random.choice([1024, 2048, 4096, 8192])
        gen = "pcie4" if t in ("nvme", "zns") else None
        return t, size, gen
    return "sata", 512, None

def net_profile(for_class: str, fabric: str):
    if fabric == "infiniband":
        return dict(fabric=fabric, speed_gbps=400, base_latency_ms=0.05, base_bandwidth_mbps=400000,
                    jitter_ms=0.01, loss_pct=0.0, ecn=False, mtu_bytes=9000, rocev2=False)
    if fabric == "wifi":
        std = random.choice(WIFI_STANDARDS)
        phy = 23.0 if std == "802.11be" else (9.6 if std == "802.11ax" else 3.5)
        return dict(fabric=fabric, speed_gbps=phy, base_latency_ms=rnd(8, 25),
                    base_bandwidth_mbps=int(phy*1000*0.35), jitter_ms=rnd(2, 10),
                    loss_pct=rnd(0.0, 1.0), ecn=False, mtu_bytes=1500, rocev2=False,
                    wifi_standard=std, mlo=(std=="802.11be"))
    if fabric == "lte":
        return dict(fabric=fabric, speed_gbps=0.2, base_latency_ms=rnd(25, 60),
                    base_bandwidth_mbps=200, jitter_ms=rnd(5, 20), loss_pct=rnd(0.0, 1.5),
                    ecn=False, mtu_bytes=1500, rocev2=False)
    # ethernet
    speed = random.choice([1, 2.5, 10, 25, 40, 100])
    return dict(fabric="ethernet", speed_gbps=speed, base_latency_ms=rnd(0.2, 1.5),
                base_bandwidth_mbps=speed*1000, jitter_ms=rnd(0.02, 0.3),
                loss_pct=rnd(0.0, 0.05), ecn=maybe(0.2), mtu_bytes=random.choice([1500, 9000]),
                rocev2=maybe(0.15))

def health_profile(for_class: str):
    base_crash = 0 if for_class in ["server", "hpc"] else random.choice([0,0,1,2])
    thermal = rnd(0.0, 0.12 if for_class not in ["server","hpc"] else 0.06)
    ram_err = rnd(0.0, 0.05 if for_class in ["server","hpc"] else 0.15)
    return dict(
        ram_errors_million_hours=ram_err,
        last_week_crashes=base_crash,
        thermal_derate=thermal,
        last_service_days=random.choice([30,60,120,240,480]),
        temperature_c=rnd(28, 75),
        fan_fault=maybe(0.03 if for_class in ["server","hpc"] else 0.01)
    )

def battery_profile(for_class: str):
    if for_class in ["phone","laptop"]:
        return dict(present=True,
                    level_pct=random.randint(10,100),
                    cycles=random.randint(50,900),
                    health_pct=random.randint(70,100))
    return dict(present=False, level_pct=100, cycles=0, health_pct=100)

def labels_profile(for_class: str):
    zone = random.choice(["lab","edge","home","cloudlet","rackA","rackB","rackC"])
    trust = str(rnd(0.60, 0.98))
    return dict(zone=zone, rack=random.choice(["A1","A2","B1","B2","C1","C2"]),
                trust=trust, wasm=random.choice(["enabled","disabled"]),
                owner=random.choice(["dept", "student", "lab", "org"]))

def formats_supported(gpu: Any, npu: Any, fpga: Any, asic: Any):
    fmts = {"native", "wasm"}
    if gpu: fmts.add("cuda")
    if npu: fmts.add("npu")
    if fpga: fmts.add("fpga")
    if asic: fmts.add("asic")
    return sorted(list(fmts))

def accelerators_block(npu, fpga, asic) -> Dict[str, Any]:
    acc = {
        "npu": "none", "npu_tops": 0, "jetson_cuda": False,
        "fpga_card": "none", "fpga_hbm_gb": 0, "fpga_net_gbps": 0,
        "fpga_bitstream": "", "fpga_toolchain": "",
        "asic_kind": "none", "asic_memory_gb": 0,
        "asic_interconnect": "none", "asic_dtypes": []
    }
    if npu:
        acc["npu"] = npu[0]
        acc["npu_tops"] = npu[1]
    if fpga:
        acc["fpga_card"] = fpga
        acc["fpga_hbm_gb"] = random.choice([8, 16, 32])
        acc["fpga_net_gbps"] = random.choice([100, 200])
        acc["fpga_bitstream"] = random.choice(["graph_analytics_v1","regex_accel","stream_proc"])
        acc["fpga_toolchain"] = random.choice(["vitis_2024.2","oneapi_2024.1","other"])
    if asic:
        acc["asic_kind"] = asic
        acc["asic_memory_gb"] = random.choice([32, 64, 96, 128, 192])
        acc["asic_interconnect"] = random.choice(["tpu_torus","neuronlink","rocev2"])
        acc["asic_dtypes"] = random.sample(["fp32","bf16","fp16","fp8","int8","int4"], k=random.randint(2,4))
    return acc

def dpu_block(for_class: str):
    if for_class in ["server","hpc"] and maybe(0.25):
        t = random.choice(["bluefield3","pensando_elba","intel_ipu_e2100"])
        return dict(type=t, ports_gbps=random.choice([100,200,400]),
                    protocol=random.choice(["rocev2","infiniband","tcp"]),
                    offloads=random.sample(["tls","ipsec","nvmeof","vxlan","geneve","compression","telemetry"],
                                           k=random.randint(2,5)))
    return dict(type="none")

def cxl_block(for_class: str):
    if for_class in ["server","hpc"] and maybe(0.20):
        return dict(type="type3_memory", capacity_gb=random.choice([256,512,1024]),
                    link_gtps=random.choice([32,64]), added_ns_latency=random.choice([120,150,180]))
    return dict(type="none")

def gpu_block(for_class: str):
    if for_class in ["gaming_rig","workstation","server","hpc","sbc"] and maybe(0.75 if for_class!="sbc" else 0.35):
        v, m, vram, score, cc, hbm_gb, hbm_bw, nvlink, tdp = random.choice(GPU_CATALOG)
        return dict(
            type="real", vendor=v, model=m, vram_gb=vram, cuda_cores=random.choice([2048,3072,3584,4352,16384]),
            tensor_cores=random.choice([64,96,112,528]), rt_cores=random.choice([0,20,24,28,80]),
            compute_capability=cc, hbm_gb=hbm_gb, hbm_bw_tbps=hbm_bw, nvlink=nvlink, tdp_w=tdp,
            accel_score=float(score)
        )
    return dict(type="none", vendor="none", model="none", vram_gb=0, cuda_cores=0,
                tensor_cores=0, rt_cores=0, compute_capability="", hbm_gb=0, hbm_bw_tbps=0.0,
                nvlink=False, tdp_w=0, accel_score=0.0)

def os_block(for_class: str, arch: str):
    if for_class in ["phone"]:
        return dict(distro="postmarketOS", version="stable", kernel="6.6", container_runtime="podman")
    if for_class in ["sbc"]:
        distro = random.choice(["raspbian","ubuntu","debian"])
        return dict(distro=distro, version="24.04", kernel=random.choice(["5.15","6.1","6.6"]), container_runtime="docker")
    if for_class in ["server","hpc"]:
        return dict(distro="ubuntu", version="24.04", kernel="6.8", container_runtime="containerd")
    # laptops/workstations/gaming
    return dict(distro="ubuntu", version="24.04", kernel="6.8", container_runtime="docker")

def isa_exts(arch: str) -> List[str]:
    if arch == "amd64":
        opts = ["avx2","avx512","sse4_2","sha","aes"]
    elif arch == "arm64":
        opts = ["neon","sve","sve2","sha","aes"]
    else:  # riscv64 (treat as minimal SIMD support in sim)
        opts = ["aes","sha"]
    k = random.randint(2, min(5, len(opts)))
    return sorted(random.sample(opts, k=k))

def name_for(cls: str, idx: int) -> str:
    prefix = {
        "phone": "phone",
        "sbc": "sbc",
        "laptop": "laptop",
        "workstation": "ws",
        "gaming_rig": "grig",
        "server": "srv",
        "hpc": "hpc"
    }[cls]
    return f"{prefix}-{idx:03d}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--outdir", type=str, default="nodes")
    args = ap.parse_args()

    random.seed(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    devices = []
    for i in range(1, args.count+1):
        cls = choice_weighted(DEVICE_CLASS_WEIGHTS)
        arch = pick_arch(cls)
        name = name_for(cls, i)
        cores, base, turbo, tdp = cpu_profile(cls, arch)
        ram_gb, ecc, ddr = mem_profile(cls)
        stype, ssize, igen = storage_profile(cls)
        fab = pick_fabric(cls)
        net = net_profile(cls, fab)
        health = health_profile(cls)
        battery = battery_profile(cls)
        labels = labels_profile(cls)
        osb = os_block(cls, arch)
        cpu = dict(uarch=random.choice(["Zen4","Zen3","GoldenCove","NeoverseV1","NeoverseN2","P-core/E-core"]),
                   isa_ext=isa_exts(arch), cores=cores, base_ghz=base, turbo_ghz=turbo, tdp_w=tdp,
                   numa_nodes=random.choice([1,1,1,2,4]) if cls in ["server","hpc"] else 1)
        memory = dict(ram_gb=ram_gb, ecc=bool(ecc), channels=random.choice([2,4,8]) if cls in ["server","hpc"] else random.choice([2,2,2,4]),
                      ddr_gen=(ddr if ddr else random.choice(["ddr4","ddr5","lpddr4","lpddr5"])))
        storage = dict(type=stype, interface_gen=igen, size_gb=ssize,
                       tbw_pct_used=rnd(0, 70 if cls!="hpc" else 40),
                       smart_health=rnd(0.85, 1.0), qd_cap=random.choice([32,64,128,256]))
        gpu = gpu_block(cls)
        npu = pick_npu(cls)
        fpga = pick_fpga(cls)
        asic = pick_asic(cls)
        acc = accelerators_block(npu, fpga, asic)
        dpu = dpu_block(cls)
        cxl = cxl_block(cls)
        io = dict(has_usb3=maybe(0.8), csi_cam=maybe(0.3), m2_socket=maybe(0.6), pcie_lanes=random.choice([4,8,16,32]))

        fmts = formats_supported(gpu if gpu["type"]=="real" else None, npu, fpga, asic)

        node = {
            "name": name,
            "class": cls,
            "arch": arch,
            "role": "worker",
            "os": osb,
            "power": {
                "supply_w": 15 if cls in ["phone","sbc","laptop"] else random.choice([65,120,300,800]),
                "poe": (cls=="sbc" and maybe(0.2)),
                "battery_present": (cls in ["phone","laptop"]),
                "tdp_total_w": (gpu.get("tdp_w",0) if gpu else 0) + tdp
            },
            "cpu": cpu,
            "memory": memory,
            "storage": storage,
            "battery": battery,
            "gpu": gpu,
            "accelerators": acc,
            "dpu": dpu,
            "cxl": cxl,
            "io": io,
            "network": net,
            "formats_supported": fmts,
            "labels": labels,
            "health": health,
            "data_locality": []  # fill later if you want dataset hints
        }

        # Minor cleanups for schema friendliness
        if node["gpu"]["type"] == "none":
            node["gpu"]["vram_gb"] = 0
            node["gpu"]["accel_score"] = 0.0

        devices.append(node)
        # Write file
        with open(outdir / f"{name}.yaml", "w") as f:
            yaml.safe_dump(node, f, sort_keys=False)

    # Inventory summary
    from collections import Counter
    c_class = Counter([d["class"] for d in devices])
    c_arch  = Counter([d["arch"] for d in devices])
    c_gpu   = Counter(["yes" if d["gpu"]["type"]=="real" else "no" for d in devices])
    c_npu   = Counter([d["accelerators"]["npu"] for d in devices])
    print(f"Generated {len(devices)} nodes in ./nodes/")
    print("By class:", dict(c_class))
    print("By arch:", dict(c_arch))
    print("With GPU:", dict(c_gpu))
    print("NPU kinds:", dict(c_npu))
    print("Example file:", (outdir / f"{devices[0]['name']}.yaml").as_posix())
    print("Tip: run `python3 tools/validate_nodes.py` to lint against your schema.")
    

if __name__ == "__main__":
    main()

