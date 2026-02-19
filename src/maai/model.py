import torch
import torch.nn as nn
import time
import types
import sys
import numpy as np
import threading
import queue
import copy
import os

from .input import Base
from .util import load_vap_model
from .models.vap import VapGPT
from .models.vap_bc import VapGPT_bc
from .models.vap_bc_2type import VapGPT_bc_2type
from .models.vap_nod import VapGPT_nod
from .models.vap_nod_para import VapGPT_nod_para
from .models.config import VapConfig
# from .models.vap_prompt import VapGPT_prompt


# ---------------------------------------------------------------------------
# Compatibility helpers: register stub modules so that torch.load can
# unpickle objects originally saved under the 'vap' package namespace
# (e.g. vap.model_nod_para.VapConfig, vap.events.EventConfig, ...).
# ---------------------------------------------------------------------------

class _PermissiveModule(types.ModuleType):
    """Module stub that auto-generates permissive classes for any attribute access."""
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        cls = type(name, (), {
            '__init__': lambda self, *a, **kw: self.__dict__.update(kw),
            '__setstate__': lambda self, state: (
                self.__dict__.update(state) if isinstance(state, dict) else None
            ),
            '__repr__': lambda self: f"<Stub {name}({self.__dict__})>",
        })
        setattr(self, name, cls)
        return cls


def _make_permissive_class(name):
    """Create a class that accepts any constructor args for pickle compatibility."""
    return type(name, (), {
        '__init__': lambda self, *a, **kw: self.__dict__.update(kw),
        '__setstate__': lambda self, state: (
            self.__dict__.update(state) if isinstance(state, dict) else None
        ),
        '__repr__': lambda self: f"<Stub {name}({self.__dict__})>",
    })


def _setup_vap_compat():
    """Register 'vap' stub modules so torch.load can unpickle VAP_Nodding_para checkpoints."""
    registered_modules = []   # sys.modules keys to remove on cleanup
    injected_main = []        # __main__ attribute names to remove on cleanup

    # --- vap.* module stubs ---
    if 'vap' not in sys.modules or not hasattr(sys.modules['vap'], '__file__'):
        # Root package
        vap = _PermissiveModule('vap')
        sys.modules['vap'] = vap
        registered_modules.append('vap')

        # vap.model_nod_para: redirect VapConfig to our compatible version
        vap_model = _PermissiveModule('vap.model_nod_para')
        vap_model.VapConfig = VapConfig
        sys.modules['vap.model_nod_para'] = vap_model
        vap.model_nod_para = vap_model
        registered_modules.append('vap.model_nod_para')

        # Other submodules that may be referenced in pickled objects
        for submod in ['events', 'objective', 'modules', 'encoder',
                       'encoder_components', 'audio', 'utils', 'callbacks',
                       'zero_shot']:
            full_name = f'vap.{submod}'
            if full_name not in sys.modules:
                mod = _PermissiveModule(full_name)
                sys.modules[full_name] = mod
                setattr(vap, submod, mod)
                registered_modules.append(full_name)

    # --- __main__ stubs ---
    # When the training script was run directly (python train_nod_para.py),
    # classes like OptConfig and DataConfig are pickled under __main__.
    main_mod = sys.modules.get('__main__')
    if main_mod is not None:
        for cls_name in ['OptConfig', 'DataConfig', 'VAPModel']:
            if not hasattr(main_mod, cls_name):
                setattr(main_mod, cls_name, _make_permissive_class(cls_name))
                injected_main.append(cls_name)

    return registered_modules, injected_main


def _cleanup_vap_compat(registered):
    """Remove previously registered stub modules and __main__ injections."""
    registered_modules, injected_main = registered
    for name in reversed(registered_modules):
        sys.modules.pop(name, None)
    main_mod = sys.modules.get('__main__')
    if main_mod is not None:
        for cls_name in injected_main:
            try:
                delattr(main_mod, cls_name)
            except AttributeError:
                pass


def _build_nod_para_config(hyper_parameters: dict) -> VapConfig:
    """
    Build a VapConfig for nod_para mode from a training checkpoint's hyper_parameters.
    Extracts 'conf' dict (saved by save_hyperparameters) or falls back to vap_* prefixed keys.
    """
    conf_dict = {}

    # Try to get the 'conf' object (dataclass saved by Lightning save_hyperparameters)
    conf_obj = hyper_parameters.get("conf", None)
    if conf_obj is not None:
        if hasattr(conf_obj, "__dataclass_fields__"):
            # It's a dataclass instance
            from dataclasses import asdict
            conf_dict = asdict(conf_obj)
        elif isinstance(conf_obj, dict):
            conf_dict = conf_obj

    # Fallback: extract vap_* prefixed keys
    if not conf_dict:
        for k, v in hyper_parameters.items():
            if k.startswith("vap_"):
                conf_dict[k[4:]] = v

    # Build VapConfig with only known fields
    known_fields = set(VapConfig.__dataclass_fields__.keys())
    filtered = {k: v for k, v in conf_dict.items() if k in known_fields}

    return VapConfig(**filtered)

class Maai():
    
    BINS_P_NOW = [0, 1]
    BINS_PFUTURE = [2, 3]
    
    CALC_PROCESS_TIME_INTERVAL = 100

    def __init__(
        self,
        mode,
        lang: str,
        audio_ch1: Base,
        audio_ch2: Base,
        frame_rate: int = 10,
        context_len_sec: int = 20,
        device: str = "cpu",
        # num_channels: int = 2,
        cpc_model: str = os.path.expanduser("~/.cache/cpc/60k_epoch4-d0f474de.pt"),
        cache_dir: str = None,
        force_download: bool = False,
        use_kv_cache: bool = True,
        use_anchor_frames: bool = True,
        local_model = None,
        result_queue_maxsize: int = 10,
        print_process_time: bool = False,
    ):

        self.device = device
        if self.device not in ["cpu", "cuda"]:
            raise ValueError("Device must be 'cpu' or 'cuda'.")

        # -----------------------------------------------------------
        # nod_para mode: special loading flow (MaAI encoder, .ckpt)
        # -----------------------------------------------------------
        if mode == "nod_para":
            if local_model is None:
                raise ValueError("mode='nod_para' requires a local_model path to a .ckpt file.")

            print(f"[nod_para] Loading checkpoint from: {local_model}")
            # Register vap module stubs so pickle can resolve classes
            _vap_stubs = _setup_vap_compat()
            try:
                ckpt = torch.load(local_model, map_location="cpu", weights_only=False)
            finally:
                _cleanup_vap_compat(_vap_stubs)

            # Extract state_dict (strip 'model.' prefix if present from Lightning)
            sd = ckpt.get("state_dict", ckpt)
            if any(k.startswith("model.") for k in sd.keys()):
                sd = {k.replace("model.", "", 1) if k.startswith("model.") else k: v
                      for k, v in sd.items()}

            # Build VapConfig from checkpoint hyper_parameters
            hp = ckpt.get("hyper_parameters", {})
            conf = _build_nod_para_config(hp)

            # Override frame_hz to match requested frame_rate
            conf.frame_hz = frame_rate

            # Build model
            self.vap = VapGPT_nod_para(conf)

            # Build encoder from state_dict (infer architecture from tensor shapes)
            # n_heads is not stored in VapConfig; infer from encoder d_model
            # assuming head_dim=64 (WavLM standard).
            _enc_n_heads = None
            _conf_obj = hp.get("conf", None)
            if _conf_obj is not None and hasattr(_conf_obj, "encoder_n_heads"):
                _enc_n_heads = _conf_obj.encoder_n_heads
            if _enc_n_heads is None:
                _enc_proj_key = next(
                    (k for k in sd if k == "encoder.model.proj.weight"), None
                )
                if _enc_proj_key is not None:
                    _enc_d_model = sd[_enc_proj_key].shape[0]
                    _enc_n_heads = _enc_d_model // 64  # head_dim=64
                else:
                    _enc_n_heads = 8  # legacy fallback
            self.vap.load_encoder_from_state_dict(
                sd, frame_hz=frame_rate, lim_context_sec=context_len_sec,
                n_heads=_enc_n_heads,
            )

            # Load all weights
            load_result = self.vap.load_state_dict(sd, strict=False)
            print(f"[nod_para] Model weights loaded: {load_result}")

            # Extract nod_param_stats
            default_stats = {
                'range_mean': 0.0, 'range_std': 1.0,
                'speed_mean': 0.0, 'speed_std': 1.0,
                'swing_up_mean': 0.0, 'swing_up_std': 1.0,
            }
            nod_param_stats = ckpt.get("nod_param_stats", None)
            # Also check inside hyper_parameters
            if nod_param_stats is None:
                nod_param_stats = hp.get("nod_param_stats", default_stats)
            self.vap.nod_param_stats = nod_param_stats
            print(f"[nod_para] nod_param_stats: {nod_param_stats}")

            self.vap.to(self.device)
            self.vap = self.vap.eval()

        # -----------------------------------------------------------
        # All other modes: original CPC-based loading flow
        # -----------------------------------------------------------
        else:
            conf = VapConfig()

            if mode in ["vap", "vap_mc"]:
                self.vap = VapGPT(conf)
            elif mode == "bc":
                self.vap = VapGPT_bc(conf)
            elif mode == "bc_2type":
                self.vap = VapGPT_bc_2type(conf)
            elif mode == "nod":
                self.vap = VapGPT_nod(conf)
            elif mode == "vap_prompt":
                from .models.vap_prompt import VapGPT_prompt
                self.vap = VapGPT_prompt(conf)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            # Store the initial state of the model to check for unchanged parameters
            initial_state_dict = {name: param.clone() for name, param in self.vap.named_parameters()}

            if local_model is None:
                sd = load_vap_model(mode, frame_rate, context_len_sec, lang, device, cache_dir, force_download)
            else:
                print("Loading model from local file:", local_model)
                sd = torch.load(local_model, map_location="cpu")

            self.vap.load_encoder(cpc_model=cpc_model)
            self.vap.load_state_dict(sd, strict=False)

            # The downsampling parameters are not loaded by "load_state_dict"
            self.vap.encoder1.downsample[1].weight = nn.Parameter(sd['encoder.downsample.1.weight'])
            self.vap.encoder1.downsample[1].bias = nn.Parameter(sd['encoder.downsample.1.bias'])
            self.vap.encoder1.downsample[2].ln.weight = nn.Parameter(sd['encoder.downsample.2.ln.weight'])
            self.vap.encoder1.downsample[2].ln.bias = nn.Parameter(sd['encoder.downsample.2.ln.bias'])

            self.vap.encoder2.downsample[1].weight = nn.Parameter(sd['encoder.downsample.1.weight'])
            self.vap.encoder2.downsample[1].bias = nn.Parameter(sd['encoder.downsample.1.bias'])
            self.vap.encoder2.downsample[2].ln.weight = nn.Parameter(sd['encoder.downsample.2.ln.weight'])
            self.vap.encoder2.downsample[2].ln.bias = nn.Parameter(sd['encoder.downsample.2.ln.bias'])

            # Check for parameters that were not updated from their initial values
            for name, param in self.vap.named_parameters():
                if name in initial_state_dict:
                    if torch.equal(param.data, initial_state_dict[name].data):
                        if not name.startswith('encoder.'):
                            print(f"Warning: Parameter '{name}' was not updated from its initial value.")

            self.vap.to(self.device)
            self.vap = self.vap.eval()

        self.mode = mode
        self.mic1 = audio_ch1
        self.mic2 = audio_ch2

        # Always subscribe a dedicated queue for each mic if possible
        self._mic1_queue = self.mic1.subscribe()
        self._mic2_queue = self.mic2.subscribe()

        self.audio_contenxt_lim_sec = context_len_sec
        self.frame_rate = frame_rate
        
        # Context length of the audio embeddings (depends on frame rate)
        self.audio_context_len = int(self.audio_contenxt_lim_sec * self.frame_rate)
        
        self.sampling_rate = 16000

        if self.mode == "nod_para":
            # MaAI encoder: CNN receptive field = 400 samples (stride=320).
            # Overlap must be a multiple of 320 that covers the RF.
            # 640 = 2 × 320 → 2 CNN frames of overlap → n_skip=2
            self.frame_contxt_padding = 640
            self.encoder_n_skip = self.frame_contxt_padding // 320  # =2
        else:
            # CPC encoder: original overlap
            self.frame_contxt_padding = 320
            self.encoder_n_skip = 1
        
        # Frame size = chunk_size + overlap
        self.audio_frame_size = self.sampling_rate // self.frame_rate + self.frame_contxt_padding
        
        self.current_x1_audio = []
        self.current_x2_audio = []
        
        self.result_p_now = 0.
        self.result_p_future = 0.
        self.result_p_bc_react = 0.
        self.result_p_bc_emo = 0.
        self.result_p_bc = 0.
        self.result_p_nod_short = 0.
        self.result_p_nod_long = 0.
        self.result_p_nod_long_p = 0.
        self.result_last_time = -1
        
        self.result_vad = [0., 0.]

        self.process_time_abs = -1

        self.e1_full = []
        self.e2_full = []

        self.list_process_time_context = []
        self.last_interval_time = time.time()

        # Bounded queue to avoid unbounded memory growth in long-running sessions.
        # When full, we drop the oldest element and keep the latest (real-time friendly).
        if result_queue_maxsize is None or int(result_queue_maxsize) <= 0:
            self.result_dict_queue = queue.Queue()
        else:
            self.result_dict_queue = queue.Queue(maxsize=int(result_queue_maxsize))
        self.last_result = None

        self.use_kv_cache = use_kv_cache
        self.vap_cache = None
        self._encoder_frames_per_chunk = None  # detected at runtime from first encoder output

        # CNN feature buffer for nod_para encoder (no encoder KV cache).
        # Stores 50Hz CNN features so we only re-run Transformer, not CNN.
        self._cnn_feat_buf_ch1: list = []
        self._cnn_feat_buf_ch2: list = []
        # Sequence length that VAPモデル actually sees (10Hz frames)
        self._vap_seq_len = int(self.audio_contenxt_lim_sec * self.frame_rate) if self.mode == "nod_para" else 0
        # Anchor frames: the CNN causal zero-padding produces a distinctive
        # first frame (L2≈10.17) that the Transformer relies on as a
        # positional reference. F always has this because it re-runs CNN
        # from scratch each window. D/E lose it when the buffer slides.
        # We store these frames from the first chunk and always prepend
        # them before Transformer processing.
        self._cnn_anchor_frames = 5 if (self.mode == "nod_para" and use_anchor_frames) else 0
        self._cnn_anchor_ch1 = None  # shape: [1, N_anchor, 512]
        self._cnn_anchor_ch2 = None
        self._cnn_buf_trimmed = False  # True after first buffer trim (anchor frames evicted)
        # Max 50Hz frames to keep in the regular buffer.
        # Reduced by anchor size so that anchor + buffer = context length
        # (e.g. 995 + 5 = 1000), matching F's sequence length exactly.
        self._max_cnn_feat_frames = (int(self.audio_contenxt_lim_sec * 50) - self._cnn_anchor_frames) if self.mode == "nod_para" else 0

        # Thread control
        self._stop_event = threading.Event()
        self._worker_thread = None

        self.print_process_time = print_process_time
    
    def worker(self):
        
        # Clear the queues at the start
        # This is to ensure that the queues are empty before starting the processing loop
        self._mic1_queue.queue.clear()
        self._mic2_queue.queue.clear()
        
        while not self._stop_event.is_set():
            x1 = self.mic1.get_audio_data(self._mic1_queue)
            x2 = self.mic2.get_audio_data(self._mic2_queue)

            if self._stop_event.is_set() or x1 is None or x2 is None:
                break

            self.process(x1, x2)

            # Clear the queues if they are too large
            if self._mic1_queue.qsize() > 100:
                self._mic1_queue.queue.clear()
                print("[Warning] Audio queue (channel 1) overflow detected. Clearing audio queues.")
            if self._mic2_queue.qsize() > 100:
                self._mic2_queue.queue.clear()
                print("[Warning] Audio queue (channel 2) overflow detected. Clearing audio queues.")

            # print(self._mic1_queue.qsize(), self._mic2_queue.qsize())

            # self._mic1_queue.queue.clear()
            # self._mic2_queue.queue.clear()

    def start(self):

        self.mic1.start()
        self.mic2.start()
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self.worker, daemon=True)
        self._worker_thread.start()

        # Queueを空にする
        self._mic1_queue.queue.clear()
        self._mic2_queue.queue.clear()
    
    def stop(self, wait: bool = True, timeout: float = 2.0):
        """
        Safely stop the background processing thread.
        Args:
            wait (bool): If True, wait for the thread to finish.
            timeout (float): Max seconds to wait when joining.
        """
        self._stop_event.set()
        # Unblock blocking gets by pushing sentinels
        try:
            self._mic1_queue.put(None)
            self._mic2_queue.put(None)
        except Exception:
            pass
        if wait and self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)
        
        # Best-effort queue cleanup
        try:
            self._mic1_queue.queue.clear()
            self._mic2_queue.queue.clear()
        except Exception:
            pass
    
    def process(self, x1, x2):
        
        time_start = time.time()

        # Initialize buffer if empty
        # cache is None (first chunk): no frame_contxt_padding to match training
        # cache is not None: use overlap for CNN front-end
        chunk_size = self.sampling_rate // self.frame_rate
        _first_chunk = (self.mode == "nod_para" and len(self._cnn_feat_buf_ch1) == 0)
        if len(self.current_x1_audio) == 0:
            if _first_chunk:
                self.current_x1_audio = np.array([], dtype=np.float32)
                self.current_x2_audio = np.array([], dtype=np.float32)
            else:
                self.current_x1_audio = np.zeros(self.frame_contxt_padding, dtype=np.float32)
                self.current_x2_audio = np.zeros(self.frame_contxt_padding, dtype=np.float32)

        # Add to buffer
        self.current_x1_audio = np.concatenate([self.current_x1_audio, x1])
        self.current_x2_audio = np.concatenate([self.current_x2_audio, x2])
        # Return if not enough data to process a frame
        min_size = chunk_size if _first_chunk else self.audio_frame_size
        if len(self.current_x1_audio) < min_size:
            return

        # Extract data for inference
        x1_proc = self.current_x1_audio
        x2_proc = self.current_x2_audio
        if _first_chunk:
            x1_dist = x1_proc
            x2_dist = x2_proc
        else:
            x1_dist = x1_proc[self.frame_contxt_padding:]
            x2_dist = x2_proc[self.frame_contxt_padding:]

        with torch.no_grad():
            # Create tensors more efficiently with specified dtype and device
            x1_ = torch.from_numpy(x1_proc).float().unsqueeze(0).unsqueeze(0)
            x2_ = torch.from_numpy(x2_proc).float().unsqueeze(0).unsqueeze(0)
            
            # Move to device only once
            if self.device != 'cpu':
                x1_ = x1_.to(self.device, non_blocking=True)
                x2_ = x2_.to(self.device, non_blocking=True)

            # Encode audio
            if self.mode == "nod_para":
                # CNN on new chunk, accumulate 50Hz features,
                # then re-run Transformer on the full window each frame.
                _first_chunk = len(self._cnn_feat_buf_ch1) == 0
                _n_skip = 0 if _first_chunk else self.encoder_n_skip
                cnn1_new, cnn2_new = self.vap.encode_audio_cnn(
                    x1_, x2_, n_skip_frames=_n_skip,
                )

                # Store anchor frames from the first chunk (CNN causal
                # zero-padding produces distinctive frames that the
                # Transformer uses as positional reference).
                if _first_chunk and self._cnn_anchor_ch1 is None and self._cnn_anchor_frames > 0:
                    _na = min(self._cnn_anchor_frames, cnn1_new.shape[1])
                    self._cnn_anchor_ch1 = cnn1_new[:, :_na, :].detach().clone()
                    self._cnn_anchor_ch2 = cnn2_new[:, :_na, :].detach().clone()

                self._cnn_feat_buf_ch1.append(cnn1_new)
                self._cnn_feat_buf_ch2.append(cnn2_new)
                # Trim to max context (50Hz)
                cnn1_cat = torch.cat(self._cnn_feat_buf_ch1, dim=1)
                cnn2_cat = torch.cat(self._cnn_feat_buf_ch2, dim=1)
                if cnn1_cat.shape[1] > self._max_cnn_feat_frames:
                    cnn1_cat = cnn1_cat[:, -self._max_cnn_feat_frames:, :]
                    cnn2_cat = cnn2_cat[:, -self._max_cnn_feat_frames:, :]
                    self._cnn_feat_buf_ch1 = [cnn1_cat]
                    self._cnn_feat_buf_ch2 = [cnn2_cat]
                    self._cnn_buf_trimmed = True

                # Prepend anchor frames ONLY after the buffer has been trimmed
                # (i.e. the original zero-padded frames have been evicted).
                # Before trim, the buffer still contains the original frames.
                if self._cnn_buf_trimmed and self._cnn_anchor_ch1 is not None:
                    cnn1_for_tf = torch.cat([self._cnn_anchor_ch1, cnn1_cat], dim=1)
                    cnn2_for_tf = torch.cat([self._cnn_anchor_ch2, cnn2_cat], dim=1)
                else:
                    cnn1_for_tf = cnn1_cat
                    cnn2_for_tf = cnn2_cat

                # Run Transformer + downsample on full window with anchor
                e1, e2 = self.vap.encode_audio_transformer(cnn1_for_tf, cnn2_for_tf)

                # Trim to _vap_seq_len: discard leading frames produced
                # from the anchor prefix.
                if self._vap_seq_len > 0 and e1.shape[1] > self._vap_seq_len:
                    e1 = e1[:, -self._vap_seq_len:, :]
                    e2 = e2[:, -self._vap_seq_len:, :]

            else:
                e1, e2 = self.vap.encode_audio(x1_, x2_)

            # Detect frames-per-chunk from first encoder output
            if self._encoder_frames_per_chunk is None and e1.dim() >= 2:
                self._encoder_frames_per_chunk = e1.shape[1]

            # --- GPT forward ---
            if not self.use_kv_cache:
                # No KV cache: accumulate encoder outputs and forward full sequence.
                if self.mode == "nod_para":
                    # nod_para encoder already outputs the full sliding window.
                    out, _ = self.vap.forward(e1, e2, cache=None)
                else:
                    self.e1_full.append(e1)
                    self.e2_full.append(e2)
                    if len(self.e1_full) > self.audio_context_len:
                        self.e1_full.pop(0)
                    if len(self.e2_full) > self.audio_context_len:
                        self.e2_full.pop(0)

                    x1_full_ = torch.cat(self.e1_full, dim=1)
                    x2_full_ = torch.cat(self.e2_full, dim=1)
                    if self.device != 'cpu':
                        x1_full_ = x1_full_.to(self.device, non_blocking=True)
                        x2_full_ = x2_full_.to(self.device, non_blocking=True)
                    out, _ = self.vap.forward(x1_full_, x2_full_, cache=None)

            else:
                # KV cache path (shared by all modes).
                # nod_para: encoder re-runs Transformer on the full window
                # each frame, so feed only the last encoder frame to GPT.
                # Other modes: encoder outputs a new chunk each frame.
                if self.mode == "nod_para" and self.vap_cache is not None:
                    e1_gpt = e1[:, -1:, :]
                    e2_gpt = e2[:, -1:, :]
                else:
                    e1_gpt = e1
                    e2_gpt = e2

                out, self.vap_cache = self.vap.forward(e1_gpt, e2_gpt, cache=self.vap_cache)

                # Trim KV cache to the context limit.
                if self.vap_cache is not None:
                    if self.mode == "nod_para":
                        _max_cache = self._vap_seq_len - 1
                    else:
                        _fpc = self._encoder_frames_per_chunk or 1
                        _max_cache = (self.audio_context_len - 1) * _fpc
                    if _max_cache > 0:
                        new_cache = {}
                        for key, (k_list, v_list) in self.vap_cache.items():
                            new_k = [t[..., -_max_cache:, :] if isinstance(t, torch.Tensor) and t.dim() >= 3 else t for t in k_list]
                            new_v = [t[..., -_max_cache:, :] if isinstance(t, torch.Tensor) and t.dim() >= 3 else t for t in v_list]
                            new_cache[key] = (new_k, new_v)
                        self.vap_cache = new_cache

            # Pre-create result dict structure to avoid repeated key creation
            result_dict = {
                "t": time.time(),
                "x1": x1_dist.copy(),  # Only copy when necessary
                "x2": x2_dist.copy(),
            }
            
            # Use dictionary mapping for mode-specific outputs (faster than if-elif chain)
            mode_outputs = {
                "vap": lambda: {
                    "p_now": out['p_now'],
                    "p_future": out['p_future'],
                    "vad": out['vad'],
                    "p_bins": out['p_bins']
                },
                "vap_mc": lambda: {
                    "p_now": out['p_now'],
                    "p_future": out['p_future'],
                    "vad": out['vad'],
                    "p_bins": out['p_bins']
                },
                "vap_prompt": lambda: {
                    "p_now": out['p_now'],
                    "p_future": out['p_future'],
                    "vad": out['vad']
                },
                "bc": lambda: {
                    "p_bc": out['p_bc']
                },
                "bc_2type": lambda: {
                    "p_bc_react": out['p_bc_react'],
                    "p_bc_emo": out['p_bc_emo']
                },
                "nod": lambda: {
                    "p_bc": out['p_bc'],
                    "p_nod_short": out['p_nod_short'],
                    "p_nod_long": out['p_nod_long'],
                    "p_nod_long_p": out['p_nod_long_p']
                },
                "nod_para": lambda: {
                    "p_bc": out['p_bc'],
                    "p_nod": out['p_nod'],
                    "nod_count": out['nod_count'],
                    "nod_range": out['nod_range'],
                    "nod_speed": out['nod_speed'],
                    "nod_swing_up_binary": out['nod_swing_up_binary'],
                    "nod_swing_up_value": out['nod_swing_up_value'],
                    "nod_swing_up_continuous": out['nod_swing_up_continuous'],
                }
            }
            
            # Get mode-specific outputs
            if self.mode in mode_outputs:
                result_dict.update(mode_outputs[self.mode]())
            
            self.last_result = result_dict
            
            # Non-blocking put; if full, drop oldest and keep the newest result.
            try:
                self.result_dict_queue.put_nowait(result_dict)
            except queue.Full:
                try:
                    _ = self.result_dict_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.result_dict_queue.put_nowait(result_dict)
                except queue.Full:
                    # If still full due to races, drop this result.
                    pass
            
            if self.print_process_time:
                time_process = time.time() - time_start
                self.list_process_time_context.append(time_process)
                
                # Performance monitoring (unchanged for clarity)
                if len(self.list_process_time_context) > self.CALC_PROCESS_TIME_INTERVAL:
                    ave_proc_time = np.mean(self.list_process_time_context)  # np.mean is faster than np.average
                    num_process_frame = len(self.list_process_time_context) / (time.time() - self.last_interval_time)
                    self.last_interval_time = time.time()

                    print(f'[{self.mode}] Average processing time: {ave_proc_time:.5f} [sec], #process/sec: {num_process_frame:.3f}')
                    self.list_process_time_context.clear()  # clear() is faster than = []
                
                self.process_time_abs = time.time()

        # Keep only the overlap samples in the buffer (for CNN front-end)
        self.current_x1_audio = self.current_x1_audio[-self.frame_contxt_padding:].copy()
        self.current_x2_audio = self.current_x2_audio[-self.frame_contxt_padding:].copy()
    
    def get_result(self):
        return self.result_dict_queue.get()
    
    def set_prompt_ch1(self, prompt: str):
        """
        Set the prompt text for speaker 1. This method is only available for the 'vap_prompt' mode.
        
        Args:
            prompt (str): The prompt text for speaker 1.
        """
        
        if self.mode != "vap_prompt":
            raise ValueError("This method is only available for the 'vap_prompt' mode.")
        
        self.vap.set_prompt_ch1(prompt, self.device)

    def set_prompt_ch2(self, prompt: str):
        """
        Set the prompt text for speaker 2. This method is only available for the 'vap_prompt' mode.
        
        Args:
            prompt (str): The prompt text for speaker 2.
        """
        
        if self.mode != "vap_prompt":
            raise ValueError("This method is only available for the 'vap_prompt' mode.")
        
        self.vap.set_prompt_ch2(prompt, self.device)
