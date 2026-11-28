# Installing PolyStrata

Everything about installing, starting, updating and configuring PolyStrata, on Windows and Linux, with an NVIDIA or an
AMD graphics card. The short version is in the [README](../README.md#install); an AI coding assistant can do all of
this for you with [AI_SETUP.md](AI_SETUP.md). A Mac: [MAC.md](MAC.md).

Apple Silicon Macs use the same shell launcher and complete model catalog, with native Metal engines.
Newly ported paths are experimental: see [coverage and validation](MAC.md#model-coverage).

> **On this page:** [What you need](#what-you-need) · [Windows](#windows) · [Linux](#linux) ·
> [AMD cards](#amd-cards) · [Several cards](#two-or-three-cards) · [Older CPUs](#older-cpus-experimental) ·
> [The terminal](#the-terminal) · [Updating](#updating) · [Where things are stored](#where-things-are-stored) ·
> [Setup's questions](#setups-questions) · [Tuning](#tuning-for-your-pc) · [All options](#options-without-questions)

## What you need

This is for Qwen3.8-Flash-Next. GLM 5.3 Flash needs Linux, one NVIDIA card and far more RAM:
[GLM.md](GLM.md#what-you-need). The smaller models of PolyStrata's own engine (Qwen3.6-35B-A3B, Ornith 1.5, Gemma 4
26B-A4B and the others) need Linux or Windows, one NVIDIA card and less of everything: [BACKENDS.md](BACKENDS.md#install).

| | |
| --- | --- |
| **GPU** | **NVIDIA** RTX 20, 30, 40 or 50 series, **12 GB VRAM or more** (8 GB runs, slowly). Measured on an RTX 5070 and an RTX 3090; RTX 20 (Turing, since 0.1.27) was tested by a contributor on an RTX 2070. Older NVIDIA cards have experimental paths: [OLDER_GPUS.md](OLDER_GPUS.md). **AMD** Radeon RX 7900 XT / XTX, RX 9070 / 9070 XT and Radeon AI PRO R9700 (validated), RX 7800 XT / 7700 XT and RX 9060 XT (validated by their owners), RX 6800 / 6900 series (community-reported), with 12 GB of VRAM or more. See [AMD cards](#amd-cards). |
| **RAM** | Enough for the size you pick ([which model](MODELS.md#pick-by-memory)); **64 GB** runs every size. A big GPU makes up for less RAM - the [low-RAM mode](MODELS.md#a-big-graphics-card-and-little-ram). |
| **CPU** | x86-64 with AVX2 (any Intel/AMD desktop CPU from the last ~8 years). AVX-512 (Ryzen 7000/9000) is a bit faster. Older CPUs without AVX2 are experimental and slow: [Older CPUs](#older-cpus-experimental). |
| **Disk** | ~70-80 GB free for the model, ~6 GB for the MTP layer (+1 GB with images). **Q2_0 on an AVX-512 CPU** also writes a one-time ~40 GB copy of its experts for the fast CPU kernel. On Linux with an AMD card, ROCm takes ~10 GB more when setup installs it. An NVMe SSD is strongly recommended: it makes the first start much faster. |
| **OS** | Windows 10/11, or Linux (Ubuntu 22.04/24.04 and Debian 12/13 get everything installed automatically). |
| **Driver** | **NVIDIA:** version 580 or newer for the default CUDA 13 engine ([nvidia.com/drivers](https://www.nvidia.com/drivers) or the NVIDIA App). The experimental CUDA 12 engine runs with 528 or newer on Windows and 525 or newer on Linux ([OLDER_GPUS.md](OLDER_GPUS.md)). **AMD:** on Linux the kernel's amdgpu driver (no ROCm install needed); on Windows a current AMD Software: Adrenalin Edition driver ([amd.com/support](https://www.amd.com/en/support)). |

The driver is the only thing you install yourself. Everything else - Python, the engine, the model - is set up for
you the first time: Python 3.12 if you have none (for your user account, no admin), a private Python environment in
`terminal/.venv`, the engine, the model and the MTP draft layer. On NVIDIA it uses the ready-made engine for RTX
20/30/40/50 and NVIDIA's CUDA libraries from pip (~0.4 GB); if no ready-made engine fits your PC, it offers to install
the build tools (Visual Studio Build Tools + CUDA Toolkit on Windows, `build-essential` + CUDA on Ubuntu) and compiles
the engine for your GPU (asks first; 20-40 minutes once). More: [details](DETAILS.md#windows).

## Windows

1. [Download this project](https://github.com/simplexai-labs/PolyStrata/archive/refs/heads/main.zip) and unzip it (or
   `git clone` it).
2. Double-click **`START-HERE.bat`**.
3. Answer a few questions - or just press Enter each time for the recommended choice ([the questions](#setups-questions)).

Then it downloads everything (the model is ~70 GB, so the first time takes a while - you can stop and it picks up
where it left off) and **starts the model**. When it is ready, its API is at `http://127.0.0.1:8080/v1`. If you want an
app to chat with it, we recommend [Linger](https://github.com/simplexai-labs/linger); any app that speaks the OpenAI or
Anthropic API works too.

> **While the model starts, your PC can be slow or stop responding for 1-3 minutes** (longest the first time): PolyStrata
> loads 35-55 GB into your RAM and locks part of it for the graphics card. That's normal - wait, and don't close the
> window. The window tells you what it is doing.

**Next time**, just double-click `START-HERE.bat` again: it starts right away (30-90 s to load the model), nothing is
downloaded twice. Close its window to stop the model. `SETUP.bat` (`START-HERE.bat --setup`) installs another model or changes the
settings. Starting PolyStrata from Task Scheduler at logon needs two task settings, or the start is 24x slower:
[Running it at startup](DETAILS.md#running-it-at-startup-task-scheduler).

## Linux

```bash
./setup.sh
```

The same automatic install (it uses `sudo apt`, `dnf` or `pacman` for Python and, only if it has to compile, for
the build tools), and the same start: `http://127.0.0.1:8080`. In a terminal, `./setup.sh` first presents **model → quantization →
context**. Enter a listed number or name; Enter accepts the default, and `q` cancels. Installed choices are the
defaults on later runs. Use `terminal/scripts/run-<model>.sh` (or `python terminal/polystrata.py start <name>`, or `./setup.sh --yes`) to start a saved model directly; `./setup.sh --setup` installs another model or changes the settings (`--no-scripts` skips writing per-model scripts). Other distributions, WSL and compiling: [details](DETAILS.md#linux). Nix: [SOURCE_INSTALL.md](SOURCE_INSTALL.md).

`./setup.sh --list-models` lists the platform's models, quantizations and context choices without installing them. `--hf-token TOKEN` (or `HF_TOKEN`) authenticates Hugging Face downloads for the current run without saving the token on disk.

On both Linux and macOS, `setup.sh` follows `START-HERE.bat`'s launch sequence: reuse a working `terminal/.venv`, otherwise
find 64-bit Python 3.10 or newer, install Python if needed, and create the private environment. Its shared menu
(`terminal/tools/setup_launcher.py`) reads the existing model catalogs and passes `--family`, `--model`, and `--context`
to `engine/metal/setup_mac.py` on macOS or `terminal/setup.py` on Linux. Mac menus include only the families and quantizations
supported by Metal. The installer still checks hardware and handles its other settings.

Explicit choices skip their corresponding questions. For example, `./setup.sh --family lfm25` asks only for
quantization and context before handing off. Every original command-line argument is retained. `--yes`,
`--check`, `--help`, updates, and noninteractive calls bypass the menu; `--yes` keeps the installer's defaults.
Supplying all three choices also needs no menu input:

```sh
./setup.sh --family lfm25 --model UD-Q3_K_XL --context 2048 --yes --no-start
```

On a Mac, `strata-poly` defaults to optimized expert dispatch, balanced memory management, and Q8 KV
for compatible Qwen35MoE models (FP16 otherwise). Explicit controls are `--metal-optimize` /
`--no-metal-optimize`, `--metal-memory legacy|balanced`, and `--poly-kv fp16|q8_0`.
Command-line choices override saved settings, which override defaults. Explicit Q8 KV requires `qwen36`,
`ornith`, `tiel` or `minicpmv`. Setup reports header-derived KV estimates for those models, including
context, draft and parallel allocations; unknown memory components are labeled rather than added to an
exact total. See [Mac tuning options](MAC.md#smaller-model-metal-defaults) for behavior and limits.

An incomplete `.venv` is recreated on the next run. Errors wait for Enter only when both input and output are
terminals; redirected or noninteractive commands exit with the failure status immediately.

Linux installs missing Python packages with `apt-get`, `dnf` or `pacman`, using sudo unless already running as
root. Apple Silicon macOS uses a checksum-verified standalone Python in `terminal/.python` when a suitable interpreter
is unavailable; it needs neither sudo nor Homebrew. Other distributions get instructions to install Python
manually. A working environment is reused without running any of these installation steps.

On Nix/NixOS, you can use the flake:

```bash
nix develop -c ./setup.sh
```

NixOS uses uv to create a managed Python environment. On other systems, `STRATA_UV=1 ./setup.sh` opts into the
same path. Install uv first; ordinary Linux and macOS launches use Python's standard `venv` module.

## AMD cards

The steps are the same as with NVIDIA: `START-HERE.bat` on Windows, `./setup.sh` on Linux. Setup finds the Radeon
card and chooses the AMD (HIP) engine by itself on a PC with no NVIDIA card PolyStrata can use; `--backend hip` chooses
it on a PC that has both. Supported cards: RX 7900 XT / XTX (gfx1100), RX 7800 XT / 7700 XT (gfx1101), RX 9060 XT
(gfx1200), RX 9070 / 9070 XT and Radeon AI PRO R9700 (gfx1201), and the RX 6800 / 6900 series (gfx1030). Integrated
Radeon GPUs are listed as not supported, except the Ryzen AI Max ([STRIX_HALO.md](STRIX_HALO.md)).

On Windows the engine is a ready-made HIP package, so the AMD driver is all you need; `--build` compiles it on the PC
instead (Visual Studio 2022 Build Tools with the C++ workload, and git). On Linux setup uses a system ROCm 7 when there
is one, or installs ROCm into `terminal/.venv` from AMD's wheels (~10 GB, no sudo), and compiles the engine on your PC
for the card (10-20 minutes, once; it needs a C++ compiler and git: `sudo apt install build-essential git`). Several AMD
cards share the model with `--gpus`, as on NVIDIA.

Pictures are read by the image encoder on the CPU under 24 GB of VRAM and on the card from 24 GB, as on NVIDIA.
Measurements per card, the build by hand and the tuning tables: [AMD_HIP.md](AMD_HIP.md).

## Two or three cards

**Two or three NVIDIA cards?** Just run `START-HERE.bat`: it lists your cards, says which ones PolyStrata can use, and
asks whether to share the model across them (recommended when two can). An install made on one card asks once at
its next start. Or choose yourself: `START-HERE.bat --gpus 0,2` (both, remembered), `--gpus all`, or `--gpu 0` (one
card, this start only). PolyStrata puts the faster card last, and the last card is the Main one. Each card keeps the
experts of its own layers, and prompts flow through the cards in a pipeline: on an RTX 5080 + RTX 3090 prompts were
read 18-20% faster than on the 5080 alone, decoding on par. Every card must be an RTX 20 series or newer with 8 GB or
more. See [MULTI_GPU.md](MULTI_GPU.md).

**Several AMD cards** work the same way, numbered as setup lists them. A split pays only when no single card holds
the model's experts ([AMD_HIP.md](AMD_HIP.md#rdna4-gfx1201)).

## Older CPUs (experimental)

PolyStrata's ready-made engine needs AVX2 (Intel Haswell 2013, AMD Zen 2017 or newer). Since 0.1.39 an older CPU - AVX
only (Sandy Bridge / Ivy Bridge, Xeon E5 v1/v2, AMD Bulldozer) or SSE4.2 only (Nehalem / Westmere, Xeon X5600) - runs
as an **experimental** build that setup compiles on that PC. Run setup as usual: it says the CPU has no AVX2, warns
that this is experimental and slow, and compiles the engine (10-20 minutes, once; Windows needs the Visual Studio Build
Tools and the CUDA Toolkit, which setup offers to install). It does not stop. The ready-made engine still refuses such
a CPU, and says so.

What differs in that build (`STRATA_ISA_FLOOR=avx` or `none`, chosen from the CPU; an option of `engine/CMakeLists.txt`):

- The CPU's share of the experts runs on code compiled for AVX or SSE4.2 instead of AVX2. PolyStrata's own AVX2 /
  AVX-512 kernels stay in the program but are used only on a CPU that has them.
- Speed: the GPU part is unchanged and the CPU part is slower, so a GPU with more VRAM (more experts cached there)
  helps most. With the older paths forced on a Ryzen 5 7600 + RTX 5070 and the expert cache held at 1500 slots, greedy
  decode went from 26 to 11-17 tok/s with the AVX build and to 3.5-3.8 tok/s with the SSE4.2 build (Coder and IQ3_XXS,
  10 prompts each, one run). An old Xeon with DDR3 will be slower than these.
- Untested on a real old CPU by us. It was checked here by forcing the older paths on a Ryzen 5 7600
  (`STRATA_FORCE_ISA=avx` or `sse`): the answers were sane and passed the engine's checks, but not the same token for
  token as the normal build. Contributors ran earlier versions of the same approach on a Xeon E5-2680 (2.87 tok/s with
  a GTX 1080 Ti) and on Xeon E5-2687W / X5690 machines.
- AMD cards: Linux only; on Windows the AMD engine is the ready-made one, which needs AVX2. NVIDIA on Windows and Linux.
- Two-socket Xeons: memory on the other socket is slow; keep the RAM on one CPU if you can.

By hand: `cmake ... -DSTRATA_ISA_FLOOR=avx` (or `none`) builds it; `STRATA_ISA_FLOOR=avx` set for setup builds it on
any PC (to try it). Problems and results on real hardware are welcome as GitHub issues.

## The terminal

The same models and the same server, from a terminal in the PolyStrata folder:

```
python polystrata.py --list-models                 the installed models, and which one is in memory
python polystrata.py start <name>                  start an installed model's server directly (no script needed)
python polystrata.py --model <name>                chat with one; /model in the chat changes it
python polystrata.py benchmark --model <name>      speeds, memory and cache figures, as a table (--json FILE too)
```

With a server already running on the port, `--model` switches it to that model; without one, a server is started and
stopped again when the chat ends. More: [BACKENDS.md](BACKENDS.md#changing-the-model).

## Updating

**`UPDATE.bat`** (Linux: `./setup.sh --update`; on Windows also `START-HERE.bat --update`) updates PolyStrata without
starting the model - for when the GPU is busy with something else, or you just want the new version ready. In a
`git clone` it runs `git pull`, then does what `START-HERE.bat` does before a start: the engine (a new ready-made one
when the new version needs it; on Linux a compiled engine is compiled again when its source changed), the Python
packages, and each installed model's settings. The model files are not touched (at most a new engine is downloaded)
and no question is asked. Close the model's window first (a running engine cannot be replaced); start the model later
with `START-HERE.bat` as usual. In a copy that was downloaded as a zip it says to download the new zip (below): it
cannot fetch new files itself. `START-HERE.bat --rollback-engine` puts back the engine an update replaced.

Or by hand: download the new version and unzip it anywhere (or `git pull`), then run `START-HERE.bat` (Linux:
`./setup.sh`) in it. The model files are kept in a `PolyStrata-data` folder next to your PolyStrata folder, so a new copy
finds them and sets itself up the same way - nothing big is downloaded again. On Linux after a `git pull`, setup
compiles the engine again when its source changed (a minute or two for the changed files); if that compile fails, it
says so and starts the engine you had.

### Coming from 0.1.45-poly.10 or older

Since 0.2.19-poly.1 PolyStrata has a folder for each part (`engine/`, `terminal/`, `docs/`, `tools/`).
`START-HERE.bat`, `SETUP.bat`, `UPDATE.bat` and `setup.sh` stay at the top and work as before. What was at the top besides them
is now in `terminal/`:

- setup (`setup.py`), the server (`serve/`), `strata.py` and the Python environment (`.venv`)
- what setup wrote for each model: `terminal/configs/strata-<model>.json`, `terminal/scripts/run-<model>.bat` / `.sh` and `terminal/logs/strata-<model>.log`
- the engines setup installed (`engine/` is now `terminal/engine/`, `engine-poly/` is now `terminal/engine-poly/`), and
  `models/`, `packs/` and `mtp/` if they were inside the PolyStrata folder

The first start or `UPDATE.bat` after the update moves these files once and changes the paths inside the run configs
and run scripts (flat files in `terminal/` can also be moved with `START-HERE.bat --migrate` or `strata models migrate`). Files outside the PolyStrata folder (a llama-swap config, a service unit, your own scripts) are not
changed. If one of them calls these files, change the path:

- setup: `python setup.py ...` becomes `terminal\.venv\Scripts\python terminal\setup.py ...` (Linux:
  `terminal/.venv/bin/python terminal/setup.py ...`), or run `START-HERE.bat` / `./setup.sh` with the same options.
- the server: `serve/server.py --config strata-<model>.json` becomes
  `terminal/serve/server.py --config terminal/configs/strata-<model>.json` (older unmigrated `terminal/strata-<model>.json` still works), run with the Python in `terminal/.venv`. For
  llama-swap: [LLAMA_SWAP.md](LLAMA_SWAP.md).
- a run config, run script or engine: it is under `terminal/` (`configs/`, `scripts/`, `logs/`) now.

To go back to 0.1.45-poly.10 or older:

- In a second folder: put the older version next to this one and run its `START-HERE.bat --setup` (Linux:
  `./setup.sh --setup`) for your model. Like any new copy, it uses the model files in `PolyStrata-data`.
- In the same folder: after `git checkout v0.1.45-poly.10`, the older setup looks for its files at the top again.
  Move `.venv`, `strata-*.json`, `run-*.bat` / `run-*.sh` and `engine-poly` from `terminal/` back to the top and the
  files in `terminal/engine/` back to `engine/`, then remove `terminal\` (`terminal/`) from the paths inside the
  `.json` and run files. Or run `START-HERE.bat --setup` (`./setup.sh --setup`) for your model, which writes those
  files again with the old paths.

## Where things are stored

- **How the model starts:** `terminal/configs/strata-<model>.json` (context, GPUs, host, API key, ...), written by setup;
  `terminal/scripts/run-<model>.bat` / `.sh` (unless `--no-scripts` was passed), the log
  `terminal/logs/strata-<model>.log` and, when shared settings were saved (`POST /settings`),
  `terminal/configs/strata-<model>.shared-settings.json`. Older unmigrated installs with flat files directly in
  `terminal/` still work read-only and can be moved with `START-HERE.bat --migrate` (`./setup.sh --migrate`) or
  `strata models migrate`. Running setup again for the same model (another context, say)
  rewrites the keys setup writes (`exe`, `args`, `port`, `gpu`, `host`, `api_key`, `vision`, ...) and keeps the
  ones you added (`sampling`, `mcp_servers`, `mcp`, `cors_origins`, ...); the earlier file is kept beside it as
  `strata-<model>.json.bak`. Engine options you added to `"args"` by hand are not carried over: setup names them, and
  you add them again.
- **The model files** (`models/`, `packs/`, `mtp/`, 70-120 GB): in **`PolyStrata-data` next to the PolyStrata folder**, or
  wherever `--data-dir` put them.
- **Where that data folder is remembered:** `%APPDATA%\PolyStrata\settings.json` on Windows,
  `~/.config/polystrata/settings.json` on Linux ([details](DETAILS.md#windows)).
- **The program itself:** in the `terminal` folder: `.venv/`, the engines setup installed (`engine/`, and
  `engine-poly/` for PolyStrata's own models) and `third_party/`. The `engine` folder at the top of PolyStrata is the
  engine's source.

## Setup's questions

- **Which model and size?** The original, Swift 1.5, the Coder or Unsloth's files, and Q2_0, IQ2_XS, IQ3_XXS or
  IQ3_S - see [which model](MODELS.md). The other models are chosen with `--family` ([All options](#options-without-questions)).
- **How much context?** How much text it can keep in mind at once (it suggests one for your card). 384K and
  512K (experimental) extend the model past its trained 262K by rope scaling - the setup turns it on itself (yarn and a
  covering factor; `--rope-scaling`/`--rope-scale` override) ([details](DETAILS.md#context-beyond-262k-rope-scaling-experimental-off-unless-you-pick-it)).
- **Images?** Whether it should also read pictures. The image encoder (0.9 GB) runs on the GPU (0.1-0.5 s per
  picture, ~1.4 GB of VRAM kept free for it) or on the CPU (about 3-13 s per picture, nothing on the GPU); setup picks
  the CPU under 24 GB of VRAM and the GPU from 24 GB, on NVIDIA and AMD alike. [Details](DETAILS.md#images-vision).
- **Experimental speed projection?** Off unless you say yes - [read what it does](DETAILS.md#experimental-speed-projection-experimental-off-by-default)
  first. It changes how the model answers, and only the original model offers it.

With `--yes` setup takes the recommended answer to every question.

## Tuning for your PC

Every PC is different: `START-HERE.bat --calibrate` (Linux: `./setup.sh --calibrate`) measures a few engine settings
on yours and keeps the fastest (about 15-30 minutes, longer on a slow card; on an RTX 5070 with a Ryzen 5 7600 it made the Coder 7% faster).
It keeps a setting only when it is more than 3% faster, and the result is remembered per PC and model, so updates
keep it. Measuring the CPU worker count needs a fresh engine, so the model is loaded more than once: the PC is
busy, and can stop responding for a minute or two, once per restart. When it finishes it **starts the model**,
so the server is already running when it returns - do not start it a second time.
It tunes Qwen3.8-Flash-Next's engine: PolyStrata's own models have none of these settings, and setup says so. An AMD
card under Windows keeps the default settings. [What it measures](DETAILS.md#tuning-for-your-pc---calibrate).

## Options without questions

For authenticated Hugging Face downloads, pass a read token for this run:

```sh
SETUP.bat --hf-token hf_example
./setup.sh --hf-token hf_example
```

`--hf-token=hf_example` also works, including with `setup.py` and `metal/setup_mac.py` directly.
Alternatively, set `HF_TOKEN` in the environment; the command-line option takes precedence.
The token is not saved in settings or start scripts. It authenticates model, vision, tokenizer,
and draft downloads from `https://huggingface.co`, including resumed downloads. It is not sent
to third-party `HF_ENDPOINT` mirrors, ModelScope, or redirected CDN hosts. Gated repositories
still require access approval for the token's account. This is separate from the server's `--api-key`.


```
START-HERE.bat --setup                          install another model, or change context / images (what SETUP.bat runs)
START-HERE.bat --model IQ2_XS --context 32768 --vision yes --yes     no questions
START-HERE.bat --setup --family qwen36          another model (every family: START-HERE.bat --help)
START-HERE.bat --gguf-dir D:\models\IQ2_XS      use GGUF files you already have
START-HERE.bat --custom-gguf D:\models          install one custom GGUF: a file, or a folder of its shards
START-HERE.bat --inspect D:\models\some.gguf    what a GGUF really holds and whether PolyStrata runs it (no download)
START-HERE.bat --data-dir E:\PolyStrata-data    keep the model files somewhere else
START-HERE.bat --port 8081                      another port
START-HERE.bat --gpu 1                          another GPU, numbered as nvidia-smi does (asked when several can be used)
START-HERE.bat --gpus 0,2                       several GPUs sharing the model
START-HERE.bat --vram-reserve-mib 2048          leave 2 GB of VRAM free for other programs (remembered)
START-HERE.bat --browser                        open the Monitor page in the browser when the model is ready (remembered;
                                                --no-browser undoes it)
START-HERE.bat --setup --backend hip            the AMD engine on a PC that also has an NVIDIA card
START-HERE.bat --setup --host 0.0.0.0 --api-key <secret>     reachable from other devices, with a key
START-HERE.bat --calibrate                      tune the engine for this PC (about 15-30 minutes, longer on a slow card), then start
START-HERE.bat --check                          only check this PC
START-HERE.bat --update                         update without starting (what UPDATE.bat runs)
```

**Leaving VRAM for other programs (#493):** PolyStrata fills the graphics card's free VRAM with experts (the expert cache)
and leaves `--vram-reserve-mib` MiB free: 700 by default. For a game, a 3D program or another model beside it, leave
more: `START-HERE.bat --vram-reserve-mib 2048` (Linux: `./setup.sh --vram-reserve-mib 2048`) writes it into the
model's `strata-<model>.json` and starts it; at setup (`--setup --vram-reserve-mib 2048`) it goes into the new config.
By hand: add `"--vram-reserve-mib", "2048"` to the config's `"args"` list and restart. The expert cache is then that
much smaller, so answers can be a little slower. The engine sizes its cache from the VRAM free when it starts: what
another program already holds then is left alone anyway; the reserve is room for what it takes later.

**An AMD card that also drives a Linux desktop (#560 #516):** with the default reserve the expert cache fills the
card, and when the desktop (the compositor, a browser, a new app) needs more VRAM, the amdgpu driver moves GPU memory
to system RAM, which the model's experts already fill: the OOM killer then ends the desktop session (KWin, plasmashell)
or the compositor fails ("Failed to pin framebuffer"). `./setup.sh --vram-reserve-mib 3072` fixed it in both reports
(about 2.3 GB fewer experts in VRAM, a few % of speed). Setup says so on such a PC.

**A card under 8 GB (#496):** when the default reserve leaves the expert cache too little room, the engine lowers the
reserve until it fits, and warns if the card then ends nearly full. If the start stops with "no VRAM is left for the
expert cache", that log line says how much is short; an 8K context and `--draft-vocab en` at setup free the most.

**Model files downloaded by hand, or from a mirror (#495):** setup's step 5 prints the folder it expects them in
(`PolyStrata-data\models\<SIZE>\`, e.g. `PolyStrata-data\models\IQ3_XXS\`): put them there with their original names, or
point setup at them with `--gguf-dir`. To see what a file really holds before using or downloading it, `--inspect`
reads only its headers (a few MB, also over the network: a file, a folder, a URL, `ms:owner/repo` for ModelScope or
`hf:owner/repo`): every weight group's real bits per weight and storage types, and whether it is one of the files setup
installs (under any name), stored like one of them, or not one PolyStrata runs (another architecture, or experts in a
format no PolyStrata kernel reads). A name says little: Unsloth's `UD-IQ3_XXS` stores its routed experts at 3.22 bits per
weight, `UD-Q4_K_XL` at 5.10, the original `Q2_0` at 2.25 and `IQ3_S` at 3.33. To let setup download from a Hugging Face
mirror itself, set `HF_ENDPOINT` first (Windows: `set HF_ENDPOINT=https://hf-mirror.com`, Linux:
`export HF_ENDPOINT=https://hf-mirror.com`): the same pinned revisions and checks apply, and the MTP draft layer comes
from there too.

**From ModelScope (mainland China):** every repository setup downloads from (the GSQ-RCO models, Swift 1.5, the
Coder, Unsloth's files, and the original checkpoint the MTP draft layer is taken from) is on
[ModelScope](https://www.modelscope.cn) under the same name, with the same files. Setup never switches to it by
itself (a download that fails says how): use `--source modelscope` (`STRATA_SOURCE=modelscope`). ModelScope serves a
repository's current files (no pinned revision), so each file is checked against the SHA-256 ModelScope publishes for
it (self-attested: a changed upstream repository would carry its own hash), and the MTP tensors against the
pinned checkpoint's own hashes, as from Hugging Face; setup says so when you choose it. Measured on a PC in mainland
China on 2026-10-05: 11-14 MB/s; the Q2_0 and IQ3_S files matched their published SHA-256, and Q2_0's shard 1 matched
the copy from hf-mirror.com.

**Where the ready-made engines come from:** the Windows engine of PolyStrata's own models from
[PolyStrata's releases](https://github.com/simplexai-labs/PolyStrata/releases) (`STRATA_POLY_PREBUILT_URL` names another
folder). Qwen3.8-Flash-Next runs on the Strata project's engine, unchanged, so its ready-made engine comes from
[the Strata project's releases](https://github.com/Niko1221/Strata/releases) (`--prebuilt` names another folder).

On Linux the same options go to `./setup.sh`. `START-HERE.bat --help` lists them all. The server's own settings
(sharing the GPU with games, MCP tools, CORS, API keys, the API itself) are in the [details](DETAILS.md#using-it).
