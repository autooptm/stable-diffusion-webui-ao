import os
import sys


def enabled(name, default="1"):
    return os.environ.get(f"SD_WEBUI_{name}", default).strip().lower() not in ("0", "false", "no", "off")


def png_compress_level():
    try:
        return int(os.environ.get("SD_WEBUI_PNG_COMPRESS_LEVEL", "1"))
    except ValueError:
        return 1


def _low_vram():
    from modules import shared
    o = shared.cmd_opts
    return bool(getattr(o, "lowvram", False) or getattr(o, "medvram", False) or getattr(o, "medvram_sdxl", False))


def apply(m):
    import torch
    if not torch.cuda.is_available() or getattr(m, "model", None) is None:
        return
    unet = getattr(m.model, "diffusion_model", None)
    if unet is None:
        return

    if enabled("OPT_2") and not _low_vram():
        unet.to(memory_format=torch.channels_last)
        vae = getattr(m, "first_stage_model", None)
        if vae is not None:
            vae.to(memory_format=torch.channels_last)

    if enabled("OPT_5", default="0") and not _low_vram():
        _opt_6()

    if enabled("OPT_3") and not _low_vram():
        if not isinstance(unet.__dict__.get("forward"), UNetOpt7):
            UNetOpt7(unet).install()


def _opt_6():
    import torch
    import ldm.modules.attention as attention
    import ldm.modules.diffusionmodules.openaimodel as openaimodel

    for cls in (openaimodel.ResBlock, attention.BasicTransformerBlock):
        if getattr(cls, "_sd_opt_eager", None) is not None:
            continue
        eager = cls._forward
        compiled = torch.compile(eager)
        state = {"ok": True}

        def _forward(self, *args, _eager=eager, _opt_12=compiled, _state=state, _name=cls.__name__, **kwargs):
            if _state["ok"] and not torch.is_grad_enabled():
                try:
                    return _opt_12(self, *args, **kwargs)
                except Exception as e:  # noqa: BLE001
                    _state["ok"] = False
                    print(f"optimizing {_name} failed ({type(e).__name__}: {str(e)[:200]}); using the stock path")
            return _eager(self, *args, **kwargs)

        cls._sd_opt_eager = eager
        cls._forward = _forward


def undo(m):
    unet = getattr(getattr(m, "model", None), "diffusion_model", None)
    if unet is not None and isinstance(unet.__dict__.get("forward"), UNetOpt7):
        del unet.forward


def _networks_signature():
    from modules import shared
    names = []
    nets = sys.modules.get("networks")                     # extensions-builtin/Lora
    for n in getattr(nets, "loaded_networks", None) or []:
        names.append(("lora", getattr(n, "name", repr(n)), getattr(n, "te_multiplier", None), getattr(n, "unet_multiplier", None)))
    for h in getattr(shared, "loaded_hypernetworks", None) or []:
        names.append(("hypernet", getattr(h, "name", repr(h)), getattr(h, "multiplier", None)))
    return tuple(names)


class UNetOpt7:

    MAX_OPT_9 = 8

    def __init__(self, unet):
        self.unet = unet
        self.eager = unet.forward
        self.opt_10 = {}
        self.failed = set()
        self.last_networks = ()
        self.stats = {"hits_a": 0, "hits_b": 0, "eager": 0, "failed": 0}

    def install(self):
        self.unet.forward = self

    def _weights_key(self):
        ps = list(self.unet.parameters())
        return (ps[0].data_ptr(), ps[-1].data_ptr(), ps[0].dtype) if ps else ()

    def __call__(self, x, timesteps=None, context=None, *args, **kwargs):
        import torch
        from modules import sd_hijack, sd_unet, shared

        nets = _networks_signature()
        changed = nets != self.last_networks
        self.last_networks = nets
        if (changed or nets or args or kwargs or sd_unet.current_unet is not None
                or not torch.is_tensor(timesteps) or not torch.is_tensor(context)
                or not x.is_cuda or torch.is_grad_enabled()):
            self.stats["eager"] += 1
            return self.eager(x, timesteps, context, *args, **kwargs)

        key = (tuple(x.shape), x.dtype, tuple(timesteps.shape), timesteps.dtype,
               tuple(context.shape), context.dtype, x.device,
               torch.is_autocast_enabled(), torch.get_autocast_gpu_dtype(),
               getattr(sd_hijack.current_optimizer, "name", None), bool(shared.opts.upcast_attn),
               self._weights_key())
        if key in self.failed:
            return self.eager(x, timesteps, context)

        entry = self.opt_10.get(key)
        if entry is None:
            if len(self.opt_10) >= self.MAX_OPT_9:
                self.opt_10.clear()
            try:
                entry = self._opt_8(x, timesteps, context)
            except Exception as e:  # noqa: BLE001 -- never fail a generation over a graph
                print(f"UNet optimized path failed ({type(e).__name__}: {e}); using the stock path for this shape")
                self.failed.add(key)
                self.stats["failed"] += 1
                torch.cuda.synchronize()
                return self.eager(x, timesteps, context)
            self.opt_10[key] = entry
            self.stats["hits_a"] += 1

        graph, sx, st, sc, out = entry
        sx.copy_(x)
        st.copy_(timesteps)
        sc.copy_(context)
        graph.replay()
        self.stats["hits_b"] += 1
        return out.clone()

    def _opt_8(self, x, timesteps, context):
        import torch
        sx, st, sc = x.clone(), timesteps.clone(), context.clone()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                self.eager(sx, st, sc)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, capture_error_mode="thread_local"):
            out = self.eager(sx, st, sc)
        return graph, sx, st, sc, out
