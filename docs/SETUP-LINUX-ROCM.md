# Ubuntu Dual-Boot + ROCm Setup for the RX 6800M

Step-by-step procedure to get DDERA training locally on an **AMD Radeon RX 6800M**
(Navi 22, `gfx1031`, 12 GB) in an ASUS ROG Strix G15 Advantage Edition.

**Why Linux at all.** ROCm does not support this GPU on Windows or WSL2 — AMD's matrices
there list RDNA3/RDNA4 only. Native Linux is the only path to the card. See
[`decisions.md`](../decisions.md) ADR-009.

**Why this can work despite `gfx1031` being "unsupported".** `gfx1030` (Navi 21) *is* an
officially packaged target in current ROCm and is built into the PyTorch ROCm wheels.
`HSA_OVERRIDE_GFX_VERSION=10.3.0` makes the `gfx1031` card load those shipped `gfx1030`
code objects. The two are close enough that this normally works — but it is unsupported, so
Part 7's verification gate exists to prove it rather than assume it.

> **Time budget.** Parts 0–2 take 1–2 hours (mostly waiting). Parts 3–7 take about an hour.
> Do Part 0 carefully; it is the only part that can lose data.

---

## Part 0 — Windows preparation ⚠️ *do not skip*

### 0.1 BitLocker — check this first

Repartitioning a BitLocker-encrypted drive without the recovery key can lock you out of
Windows permanently.

```powershell
# Run in an ADMIN PowerShell
manage-bde -status C:
```

- **"Protection Off" / "Fully Decrypted"** → you are fine, continue.
- **"Protection On"** → either suspend it, or save the recovery key somewhere off-machine:

```powershell
manage-bde -protectors -disable C: -RebootCount 0    # suspend until you re-enable
manage-bde -protectors -get C:                       # SAVE the 48-digit recovery key
```

### 0.2 Back up anything irreplaceable

Partitioning is usually uneventful. "Usually" is not "always".

### 0.3 Disable Fast Startup

Fast Startup leaves the Windows filesystem in a hibernated state; Linux mounting it then can
corrupt it.

*Control Panel → Power Options → Choose what the power buttons do → Change settings that are
currently unavailable → untick **Turn on fast startup** → Save.*

### 0.4 Disable Secure Boot

`amdgpu-dkms` builds unsigned kernel modules. With Secure Boot on they will not load, and
the failure looks like "GPU not detected" rather than anything informative.

Reboot into BIOS (**F2** during the ASUS splash) → *Security* / *Boot* → **Secure Boot →
Disabled** → save and exit.

> While in BIOS, if there is a **VMD / RAID** option under storage, set it to **AHCI /
> Disabled**. With VMD enabled the Ubuntu installer cannot see the NVMe drive at all.

### 0.5 Free up space

DDERA needs roughly:

| Item | Size |
|---|---|
| Ubuntu + ROCm (ROCm alone is ~30 GB installed) | ~45 GB |
| CheXpert-v1.0-small | ~11 GB |
| Cached encoder features (224k × 1024 × fp16) | <1 GB |
| Checkpoints across all variants | ~3 GB |
| Working headroom | ~20 GB |

**Allocate 80 GB.** You have ~130 GB free on C:, so this leaves Windows about 50 GB — tight
but workable. If you would rather not squeeze Windows, put `data/` on an external SSD and
allocate 60 GB instead.

Shrink the Windows partition:

*Right-click Start → Disk Management → right-click C: → **Shrink Volume** → enter `81920` MB
→ Shrink.* Leave the freed space **unallocated** — the Ubuntu installer will use it.

---

## Part 1 — Create the installer

1. Download **Ubuntu 24.04 LTS Desktop** from <https://ubuntu.com/download/desktop>.
   Use 24.04, not 25.x: ROCm supports 24.04.x and 22.04.5, and nothing newer.
2. Write it to an 8 GB+ USB stick with [Rufus](https://rufus.ie) (GPT / UEFI) or
   [balenaEtcher](https://etcher.balena.io).

---

## Part 2 — Install Ubuntu

1. Reboot, press **F8** (or **Esc**) for the ASUS boot menu, choose the USB stick in **UEFI**
   mode.
2. *Try or Install Ubuntu* → **Install**.
3. At *Installation type*, pick **"Install Ubuntu alongside Windows Boot Manager"** if
   offered — it will use the unallocated space automatically.

   If you prefer manual control, choose *Something else* and create, inside the free space:

   | Mount | Size | Type |
   |---|---|---|
   | `/` | remainder (~72 GB) | ext4 |
   | `swap` | 8 GB | swap |

   Do **not** create a new EFI partition — select the existing one Windows already uses.
4. Complete the install and reboot. GRUB should now offer both Ubuntu and Windows.

> **If it boots straight to Windows:** BIOS → *Boot* → move **ubuntu** above *Windows Boot
> Manager* in the boot priority list.

---

## Part 3 — First boot

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git curl wget build-essential python3-venv python3-dev

# Confirm the kernel can see BOTH GPUs (Cezanne = iGPU, Navi 22 = RX 6800M)
lspci | grep -Ei 'vga|display|3d'
```

You should see two AMD devices. That is expected and is dealt with in Part 5.

---

## Part 4 — Install ROCm

Commands below follow AMD's current
[install-on-linux quick start](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html).
**Check that page for the current installer version before running** — the `7.2.4` in the
URL moves over time, and using a stale one is the most common way this goes wrong.

```bash
# 4.1  Register AMD's repository (verify the version number against the docs first)
wget https://repo.radeon.com/amdgpu-install/7.2.4/ubuntu/noble/amdgpu-install_7.2.4.70204-1_all.deb
sudo apt install ./amdgpu-install_7.2.4.70204-1_all.deb
sudo apt update

# 4.2  Kernel driver
sudo apt install -y "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)"
sudo apt install -y amdgpu-dkms

# 4.3  ROCm itself (large download, ~30 GB installed)
sudo apt install -y python3-setuptools python3-wheel
sudo usermod -a -G render,video $LOGNAME
sudo apt install -y rocm

# 4.4  Reboot — required, not optional
sudo reboot
```

### 4.5 Confirm ROCm sees the card

```bash
rocminfo | grep -E 'Name|gfx'
rocm-smi
```

You are looking for **`gfx1031`** in the output. You will probably also see `gfx90c` — that
is the Ryzen iGPU, and Part 5 handles it.

> If `rocminfo` reports no agents: confirm Secure Boot is off (`mokutil --sb-state`), that
> `dkms status` shows amdgpu built for your kernel, and that you have logged out and back in
> since the `usermod` (group membership only applies to new sessions).

---

## Part 5 — Configure for `gfx1031` ⚠️ *the part people get wrong*

Two environment variables, and the second matters more than most guides admit.

```bash
# Identify which index is the discrete GPU
rocminfo | grep -E 'Agent|Name|gfx'
```

Then add to `~/.bashrc`:

```bash
# Load the officially-packaged gfx1030 kernels on this gfx1031 card
export HSA_OVERRIDE_GFX_VERSION=10.3.0

# Expose ONLY the discrete RX 6800M to ROCm.
# This is essential on this laptop: HSA_OVERRIDE_GFX_VERSION applies to EVERY visible
# agent, so without this the Ryzen iGPU (gfx90c, a Vega part) would also be told it is
# gfx1030. It is not, and the result is a crash or silently wrong numbers.
# Set the index to whichever agent rocminfo reports as gfx1031.
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0
```

```bash
source ~/.bashrc
rocminfo | grep gfx        # should now show only gfx1031
```

---

## Part 6 — PyTorch and DDERA

Ubuntu 24.04 ships Python 3.12, which is what AMD validates for ROCm PyTorch. DDERA supports
3.11–3.12, so use the system Python here rather than installing 3.11.

```bash
git clone https://github.com/SriramWorkSpace/DDERA-Data-Driven-Explainable-Radiological-Analytics.git
cd DDERA-Data-Driven-Explainable-Radiological-Analytics

python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip

# PyTorch for ROCm. Match the index URL to your installed ROCm minor version.
# Check https://pytorch.org/get-started/locally/ (Linux + pip + ROCm) for the current URL.
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.2

pip install -r requirements/base.txt -r requirements/dev.txt
pip install -e .
```

Quick sanity check before the real gate:

```bash
python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available())"
```

Expect a version string, a HIP version (**not** `None`), and `True`. `torch.cuda` is the
correct API on ROCm — the name is historical.

---

## Part 7 — Run the verification gate 🔒

This is the ADR-009 gate. **No training begins until it passes.**

```bash
python scripts/verify_gpu.py            # checks 1-6 and 8, a few minutes
python scripts/verify_gpu.py --full     # adds the 30-minute soak test
```

It tests the real workload, not just device detection: matmul correctness, **conv2d
forward+backward through MIOpen** (where unsupported targets usually break), DenseNet-121
under AMP, masked BCE, a 200-step overfit, sustained load, and a VRAM headroom probe that
tells you the largest usable batch size.

When it passes:

```bash
python scripts/verify_gpu.py --full --json reports/gpu_verification.json
```

Paste the output into [`decisions.md`](../decisions.md) under **ADR-009 → Verification
result**, together with the exact ROCm, PyTorch and kernel versions:

```bash
apt list --installed 2>/dev/null | grep -E '^rocm-core|^amdgpu-dkms'
python -c "import torch; print('torch', torch.__version__, 'hip', torch.version.hip)"
uname -r
```

Then run the test suite to confirm the environment is sound end to end:

```bash
pytest tests/ -q          # expect 195 passed
ruff check src/ tests/ scripts/
```

---

## Part 8 — If the gate fails

Work the ADR-009 ladder **in order**, and record each failure in `decisions.md` before
stepping down. Do not jump to an old ROCm version on the strength of a forum post; the
packaging has changed and current releases should be tried first.

| Rung | Action |
|---|---|
| 1 | Host install above (this document) |
| 2 | `docker run -it --device=/dev/kfd --device=/dev/dri --group-add video rocm/pytorch:latest` — isolates the stack and lets you try versions without reinstalling |
| 3 | Step ROCm minor versions **down** via `rocm/pytorch` Docker tags, newest first |
| 4 | Windows + `torch-directml` — approved fallback, flagged as degraded |

### Common failures

| Symptom | Cause and fix |
|---|---|
| `torch.cuda.is_available()` is `False` | You installed the CPU or CUDA wheel. Reinstall from the ROCm index URL. |
| `rocminfo` shows no agents | Secure Boot still on, `amdgpu-dkms` not built, or you have not re-logged in since `usermod`. |
| Crash the moment a conv runs | `HSA_OVERRIDE_GFX_VERSION` unset, or the iGPU is still visible — set `HIP_VISIBLE_DEVICES` (Part 5). |
| Check 3 (conv2d) fails or SIGSEGVs | The known `gfx1031` risk. Record it, then move to rung 2. |
| Out of memory at batch 32 | Normal on 12 GB at 320 px. Use the batch size check 8 reports. |
| Non-finite gradients under AMP | Prefer bf16 over fp16; `device.py` already selects bf16 where supported. |

---

## Everyday workflow, once this is done

```bash
cd ~/DDERA-Data-Driven-Explainable-Radiological-Analytics
source .venv/bin/activate
python scripts/verify_gpu.py --quick     # 10-second sanity check
```

The environment variables from Part 5 are in `~/.bashrc`, so they apply automatically. If you
ever move to a different machine or GPU, re-run the full gate and add a new ADR — per
Invariants 7 and 8 the backend may change, but the methodology may not.
