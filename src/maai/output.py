import sys
import math
import time
from typing import Any, Callable, Dict, List, Union
import numpy as np
import socket
import threading
import time
import pickle
import queue
from . import util
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

DEFAULT_VAP_BIN_TIMES_SEC = [0.2, 0.4, 0.6, 0.8]

# GuiPlot 専用: 先に QApplication が無いときだけデーモンスレッドで Qt を回し、
# メインが sleep だけでもワーカーから update() できるようにする。
_gui_plot_qt_lock = threading.Lock()
_gui_plot_qt_thread: threading.Thread | None = None
_gui_plot_qt_owner_ident: int | None = None
_gui_plot_qt_cmd_q: queue.Queue | None = None
_gui_plot_qt_ready = threading.Event()
_gui_plot_use_dedicated_qt_thread: bool = False


def _gui_plot_qt_owner_thread_main() -> None:
    global _gui_plot_qt_owner_ident, _gui_plot_qt_cmd_q
    from PyQt5.QtWidgets import QApplication

    _gui_plot_qt_owner_ident = threading.get_ident()
    _gui_plot_qt_cmd_q = queue.Queue()
    argv = sys.argv[:] if sys.argv else [""]
    if len(argv) == 0:
        argv = [""]
    app = QApplication(argv)
    _gui_plot_qt_ready.set()
    while True:
        try:
            while True:
                fn, retq = _gui_plot_qt_cmd_q.get_nowait()
                try:
                    fn()
                    if retq is not None:
                        retq.put((True, None))
                except BaseException as exc:  # noqa: BLE001
                    if retq is not None:
                        retq.put((False, exc))
        except queue.Empty:
            pass
        app.processEvents()
        time.sleep(1.0 / 120.0)


def _ensure_gui_plot_qt_owner() -> None:
    """先に QApplication が無い場合のみ専用スレッドを起動する。"""
    global _gui_plot_qt_thread, _gui_plot_use_dedicated_qt_thread
    from PyQt5.QtWidgets import QApplication

    with _gui_plot_qt_lock:
        if _gui_plot_qt_thread is not None and _gui_plot_qt_thread.is_alive():
            return
        if QApplication.instance() is not None:
            _gui_plot_use_dedicated_qt_thread = False
            return
        _gui_plot_use_dedicated_qt_thread = True
        _gui_plot_qt_ready.clear()
        _gui_plot_qt_thread = threading.Thread(
            target=_gui_plot_qt_owner_thread_main,
            daemon=True,
            name="MaAI-GuiPlot-Qt",
        )
        _gui_plot_qt_thread.start()
        if not _gui_plot_qt_ready.wait(30.0):
            raise RuntimeError("GuiPlot: Qt オーナースレッドの起動がタイムアウトしました。")


def _gui_plot_run_on_qt_owner(fn: Callable[[], Any]) -> None:
    """専用 Qt スレッド上で fn を実行。それ以外のスレッドからはキュー経由で同期実行。"""
    if not _gui_plot_use_dedicated_qt_thread:
        fn()
        return
    assert _gui_plot_qt_cmd_q is not None
    if threading.get_ident() == _gui_plot_qt_owner_ident:
        fn()
        return
    retq: queue.Queue = queue.Queue(maxsize=1)
    _gui_plot_qt_cmd_q.put((fn, retq))
    ok, payload = retq.get()
    if not ok:
        raise payload


def _draw_bar(value: float, length: int = 30) -> str:
    """基本的なバーグラフを描画"""
    bar_len = min(length, max(0, int(value * length)))
    return '█' * bar_len + '-' * (length - bar_len)


def _draw_symmetric_bar(value: float, length: int = 30) -> str:
    """対称的なバーグラフを描画（-1.0から1.0の範囲）"""
    max_len = length // 2
    value = max(-1.0, min(1.0, value))
    if value >= 0:
        pos = int(value * max_len)
        return ' ' * max_len + '│' + '█' * pos + ' ' * (max_len - pos)
    else:
        neg = int(-value * max_len)
        return ' ' * (max_len - neg) + '█' * neg + '│' + ' ' * max_len


def _draw_balance_bar(value: float, length: int = 30) -> str:
    """0.5を中心としたバランスバーを描画"""
    # 2チャネルの場合は1チャネル目のデータを渡すこと
    max_len = length // 2
    value = max(0.0, min(1.0, value))
    diff = value - 0.5
    if diff > 0:
        # value > 0.5: ch1が話す確率が高い → 左側にバー
        left = int(diff * 2 * max_len + 0.5)
        return ' ' * (max_len - left) + '█' * left + '│' + ' ' * max_len
    else:
        # value < 0.5: ch2が話す確率が高い → 右側にバー
        right = int(-diff * 2 * max_len + 0.5)
        return ' ' * max_len + '│' + '█' * right + ' ' * (max_len - right)


def _draw_rep_pred_incremental_bar(selected: int, num_classes: int, length: int = 30) -> str:
    """クラス 0..n-1 に対し、左から 1 区画ずつ増えるバー（0→1区画、1→2区画、2→3区画）。"""
    n = max(1, int(num_classes))
    sel = int(max(0, min(n - 1, selected)))
    n_fill = sel + 1  # 左から何区画を █ にするか（1..n）
    base, extra = divmod(length, n)
    parts: list[str] = []
    for i in range(n):
        seg = base + (1 if i < extra else 0)
        if i < n_fill:
            parts.append("█" * seg)
        else:
            parts.append("-" * seg)
    return "".join(parts)


def _rms(values: Union[List[float], tuple]) -> float:
    """RMS値を計算"""
    if not values:
        return 0.0
    return math.sqrt(sum(x * x for x in values) / len(values))


def _normalize_linear(value: float, vmin: float, vmax: float) -> float:
    """[vmin, vmax] を [0, 1] に線形正規化（範囲外はクリップ）。"""
    if vmax <= vmin:
        return 0.0
    v = max(vmin, min(vmax, float(value)))
    return (v - vmin) / (vmax - vmin)


def _is_nod_para_style_repetitions(result: Dict[str, Any]) -> bool:
    nc = result.get("nod_repetitions")
    return isinstance(nc, (list, tuple)) and len(nc) == 3


def _get_bar_for_value(key: str, value: Any, bar_length: int = 30, bar_type: str = "normal") -> tuple[str, float]:
    """キーと値に基づいて適切なバーを選択"""
    if isinstance(value, (list, tuple)):
        if len(value) > 2:
            if isinstance(value[0], (int, float)):
                # 数値のリストの場合、RMS値を計算
                rms_val = _rms(value) * 3  # type: ignore
                return _draw_bar(rms_val, bar_length), rms_val
            else:
                return "N/A", 0.0
        elif len(value) == 2:
            _value = value[0] / (value[0] + value[1])
            return _draw_balance_bar(float(_value), bar_length), value
        else:
            return "N/A", 0.0
    elif isinstance(value, (int, float)):
        return _draw_bar(float(value), bar_length), float(value)
    else:
        return "N/A", 0.0


class ConsoleBar:
    """
    maai.get_result()の内容をバーグラフで可視化するクラス
    """
    def __init__(self, bar_length: int = 30, bar_type: str = "normal"):
        self.bar_length = bar_length
        self.bar_type = bar_type
        self._first = True

    def update(self, result: Dict[str, Any]):
        if self._first:
            sys.stdout.write("\x1b[2J")  # 初期クリア
            self._first = False
        sys.stdout.write("\x1b[H")  # カーソルを左上に移動
        
        # 時刻の表示
        if 't' in result:
            dt = time.localtime(result['t'])
            ms = int((result['t'] - int(result['t'])) * 1000)
            print(f"Time: {dt.tm_year:04d}/{dt.tm_mon:02d}/{dt.tm_mday:02d} {dt.tm_hour:02d}:{dt.tm_min:02d}:{dt.tm_sec:02d}.{ms:03d}")
            print("-" * (self.bar_length + 30))
        
        is_nod_para = _is_nod_para_style_repetitions(result)

        # bar_typeがbalanceのとき、x1/x2を横並び表示（nod_para は後段で順序付き表示）
        if (
            self.bar_type == "balance"
            and "x1" in result
            and "x2" in result
            and not is_nod_para
        ):
            x1 = np.squeeze(np.array(result["x1"])).tolist()
            x2 = np.squeeze(np.array(result["x2"])).tolist()
            bar1, val1 = _get_bar_for_value("x1", x1, self.bar_length // 2 - 1, "normal")
            bar1 = bar1[::-1]
            bar2, val2 = _get_bar_for_value("x2", x2, self.bar_length // 2 - 1, "normal")
            print(f"x1 │ x2{' ' * 8}: {bar1} │ {bar2} ({val1:.4f}, {val2:.4f})")

        # vadがある場合はvad[0]をvad(x1)、vad[1]をvad(x2)として表示
        if "vad" in result:
            vad1 = result["vad"][0]
            vad2 = result["vad"][1]
            result["vad(x1)"] = vad1
            result["vad(x2)"] = vad2

        skip_nod_para: set = set()
        if is_nod_para:
            skip_nod_para = {
                "x1",
                "x2",
                "p_nod",
                "nod_range",
                "nod_speed",
                "nod_repetitions",
                "nod_repetitions_pred",
                "nod_swing_up",
                "nod_swing_up_pred",
            }
            # 順: x1, x2 → p_nod → range → speed → repetitions（3本）→ swing（確率のみ）
            if self.bar_type == "balance" and "x1" in result and "x2" in result:
                x1 = np.squeeze(np.array(result["x1"])).tolist()
                x2 = np.squeeze(np.array(result["x2"])).tolist()
                bar1, val1 = _get_bar_for_value(
                    "x1", x1, self.bar_length // 2 - 1, "normal"
                )
                bar1 = bar1[::-1]
                bar2, val2 = _get_bar_for_value(
                    "x2", x2, self.bar_length // 2 - 1, "normal"
                )
                print(
                    f"x1 │ x2{' ' * 8}: {bar1} │ {bar2} ({val1:.4f}, {val2:.4f})"
                )
            else:
                for k in ("x1", "x2"):
                    if k not in result:
                        continue
                    val = np.squeeze(np.array(result[k])).tolist()
                    bar, _value = _get_bar_for_value(
                        k, val, self.bar_length, self.bar_type
                    )
                    if type(_value) is float:
                        print(f"{k:15}: {bar} ({_value:.3f})")
                    elif type(_value) is list:
                        print(
                            f"{k:15}: {bar} ({', '.join(f'{v:.3f}' for v in _value)})"
                        )

            if "p_nod" in result:
                v = float(result["p_nod"])
                bar = _draw_bar(v, self.bar_length)
                print(f"{'p_nod':15}: {bar} ({v:.3f})")

            if "nod_range" in result:
                rv = float(result["nod_range"])
                nv = _normalize_linear(rv, 0.035, 0.15)
                bar = _draw_bar(nv, self.bar_length)
                print(f"{'nod_range':15}: {bar} ({rv:.4f})")

            if "nod_speed" in result:
                sv = float(result["nod_speed"])
                nv = _normalize_linear(sv, 0.08, 0.25)
                bar = _draw_bar(nv, self.bar_length)
                print(f"{'nod_speed':15}: {bar} ({sv:.4f})")

            nc = result["nod_repetitions"]
            labels = ("1", "2", "3+")
            for i, lab in enumerate(labels):
                v = float(nc[i])
                bar = _draw_bar(v, self.bar_length)
                label = f"nod_rep {lab}"
                print(f"{label:15}: {bar} ({v:.3f})")

            if "nod_repetitions_pred" in result:
                rp = int(result["nod_repetitions_pred"])
                bar = _draw_rep_pred_incremental_bar(rp, 3, self.bar_length)
                cls_labels = ("1", "2", "3+")
                lab = cls_labels[rp] if 0 <= rp < len(cls_labels) else "?"
                print(f"{'nod_rep_pred':15}: {bar} (class {lab})")

            if "nod_swing_up" in result:
                v = float(result["nod_swing_up"])
                bar = _draw_bar(v, self.bar_length)
                print(f"{'nod_swing_up':15}: {bar} ({v:.3f})")

            if "nod_swing_up_pred" in result:
                sp = int(max(0, min(1, int(result["nod_swing_up_pred"]))))
                bar = _draw_bar(float(sp), self.bar_length)
                print(f"{'swing_pred':15}: {bar} ({sp} = off/on)")
    
        # 各キーを動的に処理
        for key, value in result.items():
            if key in ("t", "vad", "bin_times"):
                continue
            if key in skip_nod_para:
                continue
            # x1/x2は既に横並びで表示したのでスキップ
            if self.bar_type == "balance" and key in ['x1', 'x2']:
                continue
            if not isinstance(value, (float, int)):
                value = np.squeeze(np.array(value)).tolist()
            if (
                key == "p_bins"
                and isinstance(value, list)
                and len(value) == 2
                and isinstance(value[0], list)
            ):
                for si, spk_bins in enumerate(value):
                    bins_str = ", ".join(f"{b:.3f}" for b in spk_bins)
                    bar = _draw_bar(float(np.mean(spk_bins)), self.bar_length)
                    print(f"{'p_bins(spk'+str(si+1)+')':15}: {bar} [{bins_str}]")
                continue
            if (
                key in ("p_bins_now", "p_bins_future")
                and isinstance(value, list)
                and len(value) == 2
            ):
                a, b = float(value[0]), float(value[1])
                print(f"{key:15}: spk1={a:.4f}  spk2={b:.4f}  (ビン合計÷2、各話者 [0,1])")
                continue
            bar, _value = _get_bar_for_value(key, value, self.bar_length, self.bar_type)
            if type(_value) is float:
                print(f"{key:15}: {bar} ({_value:.3f})")
            elif type(_value) is list:
                print(f"{key:15}: {bar} ({', '.join(f'{v:.3f}' for v in _value)})")
        print("-" * (self.bar_length + 30))

class TcpReceiver:
    def __init__(self, ip, port, mode):
        self.ip = ip
        self.port = port
        self.mode = mode
        self.sock = None
        self.result_queue = queue.Queue()
    
    def _bytearray_2_vapresult(self, data: bytes) -> Dict[str, Any]:
        if self.mode in ['vap', 'vap_mc', 'vap_prompt']:
            vap_result = util.conv_bytearray_2_vapresult(data)
        elif self.mode == 'bc_2type':
            vap_result = util.conv_bytearray_2_vapresult_bc_2type(data)
        elif self.mode == 'nod':
            vap_result = util.conv_bytearray_2_vapresult_nod(data)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        return vap_result
    
    def connect_server(self):    
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.ip, self.port))
        print('[CLIENT] Connected to the server')

    def _start_client(self):
        while True:
            try:
                self.connect_server()
                while True:
                    try:
                        size = int.from_bytes(self.sock.recv(4), 'little')
                        data = b''
                        while len(data) < size:
                            data += self.sock.recv(size - len(data))
                        vap_result = self._bytearray_2_vapresult(data)
                        self.result_queue.put(vap_result)

                    except Exception as e:
                        print('[CLIENT] Receive error:', e)
                        break  # 受信エラー時は再接続ループへ
            except Exception as e:
                print('[CLIENT] Connect error:', e)
                time.sleep(0.5)
                continue
            # 切断時はソケットを閉じて再接続ループへ
            try:
                if hasattr(self, 'sock') and self.sock is not None:
                    self.sock.close()
            except Exception:
                pass
            self.sock = None
            print('[CLIENT] Disconnected. Reconnecting...')
            time.sleep(0.5)

    def start(self):
        threading.Thread(target=self._start_client, daemon=True).start()
    
    def get_result(self):
        return self.result_queue.get()

class TcpTransmitter:
    def __init__(self, ip, port, mode):
        self.ip = ip
        self.port = port
        self.mode = mode
        self.result_queue = queue.Queue()
    
    def _vapresult_2_bytearray(self, result_dict: Dict[str, Any]) -> bytes:
        if self.mode in ['vap', 'vap_mc']:
            data_sent = util.conv_vapresult_2_bytearray(result_dict)
        elif self.mode == 'bc_2type':
            data_sent = util.conv_vapresult_2_bytearray_bc_2type(result_dict)
        elif self.mode == 'nod':
            data_sent = util.conv_vapresult_2_bytearray_nod(result_dict)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        return data_sent
        
    def _start_server(self):
        while True:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.bind((self.ip, self.port))
                s.listen(1)
                print('[OUT] Waiting for connection...')
                conn, addr = s.accept()
                print('[OUT] Connected by', addr)
                while True:
                    try:
                        result_dict = self.result_queue.get()
                        data_sent = self._vapresult_2_bytearray(result_dict)
                        data_sent_all = len(data_sent).to_bytes(4, 'little') + data_sent
                        conn.sendall(data_sent_all)
                    except Exception as e:
                        print('[OUT] Send error:', e)
                        break
            except Exception as e:
                print('[OUT] Disconnected by', addr)
                print(e)
                continue
            
    def start_server(self):
        threading.Thread(target=self._start_server, daemon=True).start()
        
    def update(self, result: Dict[str, Any]):
        self.result_queue.put(result)

# 新規追加: GUIでバーグラフを表示するクラス
class GuiBar:
    """matplotlibを用いて結果をバーグラフでGUI表示するクラス"""
    def __init__(self, bar_type: str = "normal"):
        self.bar_type = bar_type
        self.plt = plt
        self.fig, self.ax = plt.subplots()
        plt.ion()
        plt.show()
        # バーアーティストを保持してリアルタイム更新
        self.bars = None

        import seaborn as sns
        sns.set_theme(style="whitegrid")

    def update(self, result: Dict[str, Any]):
        """resultのキーと値をバーグラフで更新表示する"""
        labels = []
        values = []
        for key, value in result.items():
            if key == 't':
                continue
            # 配列やリストは適切にスカラー化
            if not isinstance(value, (int, float)):
                value = np.squeeze(np.array(value)).tolist()
            if isinstance(value, (list, tuple)):
                if len(value) > 2 and isinstance(value[0], (int, float)):
                    val = _rms(value) * 3
                elif len(value) == 2:
                    total = value[0] + value[1]
                    val = (value[1] / total) if total != 0 else 0.0
                else:
                    val = 0.0
            else:
                try:
                    val = float(value)
                except Exception:
                    continue
            labels.append(key)
            values.append(val)
        # 初回描画またはラベル数が変わった場合は新規描画
        if self.bars is None or len(self.bars) != len(values):
            self.ax.clear()
            self.bars = self.ax.bar(labels, values, color='skyblue')
            self.ax.set_ylim(0, 1)
            self.ax.set_xticks(range(len(labels)))
            self.ax.set_xticklabels(labels)
            self.ax.set_title('Result Bar Graph')
        else:
            # 既存のバーを更新
            for bar, v in zip(self.bars, values):
                bar.set_height(v)
        # 描画を反映
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        self.plt.pause(0.001)


class GuiPlot:
    
    # p_now / p_future / p_bins 以外の p_* スカラー系列の色（未登録は控えめな緑）
    _P_SCALAR_RGB: Dict[str, tuple[int, int, int]] = {
        "p_nod_short": (230, 70, 70),
        "p_nod_long": (255, 210, 70),
        "p_nod_long_p": (80, 200, 120),
        "p_bc_react": (230, 70, 70),
        "p_bc_emo": (255, 210, 70),
        "p_nod": (80, 200, 120),
        "p_bc": (230, 70, 70),
    }

    @staticmethod
    def _rgb_for_p_scalar(key: str) -> tuple[int, int, int]:
        return GuiPlot._P_SCALAR_RGB.get(key, (90, 170, 100))

    def __init__(
        self,
        shown_context_sec: int = 10,
        frame_rate: float = 10,
        sample_rate: int = 16000,
        figsize=(14, 10),
        use_fixed_draw_rate: bool = True,
        window_title: str = "MAAI Plot",
    ) -> None:
        _ensure_gui_plot_qt_owner()

        def _boot() -> None:
            self._bootstrap_ui(
                shown_context_sec=shown_context_sec,
                frame_rate=frame_rate,
                sample_rate=sample_rate,
                figsize=figsize,
                use_fixed_draw_rate=use_fixed_draw_rate,
                window_title=window_title,
                use_cross_thread_bridge=not _gui_plot_use_dedicated_qt_thread,
            )

        if _gui_plot_use_dedicated_qt_thread:
            _gui_plot_run_on_qt_owner(_boot)
        else:
            _boot()

    def _bootstrap_ui(
        self,
        *,
        shown_context_sec: int,
        frame_rate: float,
        sample_rate: int,
        figsize: Any,
        use_fixed_draw_rate: bool,
        window_title: str,
        use_cross_thread_bridge: bool,
    ) -> None:
        try:
            from PyQt5.QtCore import QObject, Qt, QThread, pyqtSignal
            from PyQt5.QtWidgets import QApplication, QVBoxLayout, QWidget

            import pyqtgraph as pg
        except ImportError as exc:
            raise ImportError(
                "GuiPlot には PyQt5 と pyqtgraph が必要です。"
                " 例: pip install PyQt5 pyqtgraph"
            ) from exc

        self._pg = pg
        if QApplication.instance() is None:
            self._app = QApplication([])
        else:
            self._app = QApplication.instance()

        self._root = QWidget()
        self._window_title = str(window_title)
        self._root.setWindowTitle(self._window_title)
        fw = max(800, int(figsize[0] * 100))
        fh = max(600, int(figsize[1] * 100))
        self._root.resize(fw, fh)
        pg.setConfigOptions(antialias=False)
        self.graph = pg.GraphicsLayoutWidget()
        lay = QVBoxLayout()
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.graph)
        self._root.setLayout(lay)

        self.shown_context_sec = float(shown_context_sec)
        self.frame_rate = float(frame_rate)
        self.sample_rate = int(sample_rate)
        self.MAX_CONTEXT_LEN = max(1, int(round(self.frame_rate * self.shown_context_sec)))
        self.MAX_CONTEXT_WAV_LEN = max(1, int(self.sample_rate * self.shown_context_sec))
        self.use_fixed_draw_rate = bool(use_fixed_draw_rate)
        self._last_draw_time = 0.0

        self.plots: Dict[str, Any] = {}
        self.curves: Dict[str, Any] = {}
        self.data_buffer: Dict[str, Any] = {}
        self.keys: List[str] = []
        self.initialized = False

        self._x_wav = np.linspace(-self.shown_context_sec, 0.0, self.MAX_CONTEXT_WAV_LEN)
        self._x_ctx = np.linspace(-self.shown_context_sec, 0.0, self.MAX_CONTEXT_LEN)

        self._update_bridge: Any = None
        if use_cross_thread_bridge:

            class _GuiPlotUpdateBridge(QObject):
                """threading.Thread 等からの update を Qt GUI スレッドへキューする。"""

                request = pyqtSignal(object)

                def __init__(self, owner: "GuiPlot") -> None:
                    super().__init__(owner._root)
                    self.request.connect(owner._update_impl, type=Qt.QueuedConnection)

            self._update_bridge = _GuiPlotUpdateBridge(self)

    def set_window_title(self, title: str) -> None:
        """ウィンドウのタイトルバーに表示する文字列を変更する。"""

        def _apply() -> None:
            self._window_title = str(title)
            self._root.setWindowTitle(self._window_title)

        _gui_plot_run_on_qt_owner(_apply)

    @staticmethod
    def _expand_threshold_crossings(x: np.ndarray, y: np.ndarray, th: float) -> tuple[np.ndarray, np.ndarray]:
        """隣接サンプル間で y が th を跨ぐとき、交点 (xc, th) を挿入した折れ線にする。"""
        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        if x.size == 0 or y.size == 0 or x.size != y.size:
            return x, y
        xe: list[float] = [float(x[0])]
        ye: list[float] = [float(y[0])]
        for i in range(1, len(y)):
            x0, x1 = float(x[i - 1]), float(x[i])
            y0, y1 = float(y[i - 1]), float(y[i])
            if y1 != y0 and (y0 - th) * (y1 - th) < 0:
                xc = x0 + (x1 - x0) * (th - y0) / (y1 - y0)
                xe.append(xc)
                ye.append(float(th))
            xe.append(float(x[i]))
            ye.append(float(y[i]))
        return np.asarray(xe, dtype=float), np.asarray(ye, dtype=float)

    @staticmethod
    def _split_hi_lo(x: np.ndarray, y: np.ndarray, th: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x, y = GuiPlot._expand_threshold_crossings(x, y, th)
        y = np.asarray(y, dtype=float).reshape(-1)
        hi = np.where(y >= th, y, np.nan)
        lo = np.where(y <= th, y, np.nan)
        return x, hi, lo

    @staticmethod
    def _nod_scalar(key: str, val: Any) -> float:
        if key == "nod_repetitions":
            if isinstance(val, (list, tuple, np.ndarray)):
                a = np.asarray(val, dtype=float).reshape(-1)
                return float(int(np.argmax(a))) if a.size else 0.0
            return float(max(0, min(2, int(round(float(val))))))
        if key == "nod_range":
            return float(max(0.035, min(0.15, float(val))))
        if key == "nod_speed":
            return float(max(0.08, min(0.25, float(val))))
        if key == "nod_swing_up":
            return float(max(0.0, min(1.0, float(val))))
        try:
            return float(val)
        except Exception:
            return 0.0

    def _draw_p_bins_pg(self, plot: Any, arr: np.ndarray, bin_times_value: Any) -> None:
        """2話者×ビン確率を、ビン幅に比例した横軸（将来時刻の累積秒）で描画する。"""
        import pyqtgraph as pg

        arr = np.asarray(arr, dtype=float)
        plot.clear()
        if arr.ndim != 2 or arr.shape[0] != 2:
            plot.setTitle(f"p_bins (invalid shape: {arr.shape})")
            return
        n_bins = int(arr.shape[1])
        try:
            bin_times = [float(x) for x in (bin_times_value or DEFAULT_VAP_BIN_TIMES_SEC)]
        except Exception:
            bin_times = list(DEFAULT_VAP_BIN_TIMES_SEC)
        if len(bin_times) != n_bins:
            bin_times = [1.0 for _ in range(n_bins)]
        edges = np.concatenate([[0.0], np.cumsum(bin_times)])

        plot.setTitle("p_bins (per-bin probability at current time)")
        plot.showGrid(x=True, y=True, alpha=0.15)
        plot.setMenuEnabled(False)
        plot.setMouseEnabled(x=False, y=False)
        plot.hideButtons()
        plot.setXRange(0.0, float(edges[-1]), padding=0.0)
        plot.setYRange(0.0, 2.0, padding=0.0)
        plot.setLabel("bottom", "Future time [s] (bin edges)")
        plot.setLabel("left", "Speaker")
        try:
            plot.getAxis("left").setTicks([[(0.5, "spk2"), (1.5, "spk1")]])
        except Exception:
            pass

        plot.addLine(y=1.0, pen=pg.mkPen((130, 130, 130), width=1))
        vpen = pg.mkPen((130, 130, 130), width=1, style=pg.QtCore.Qt.DashLine)
        for x in edges[1:-1]:
            plot.addLine(x=float(x), pen=vpen)

        for s in range(2):
            y0 = 1.0 if s == 0 else 0.0
            for b in range(n_bins):
                v = float(np.clip(arr[s, b], 0.0, 1.0))
                x0 = float(edges[b])
                w = float(bin_times[b])
                # 上段 spk1 = 黄、下段 spk2 = 青（p_now / x2 と揃える）
                hue_base = (245, 189, 0) if s == 0 else (80, 160, 240)
                color = (
                    int(hue_base[0] * (0.35 + 0.65 * v)),
                    int(hue_base[1] * (0.35 + 0.65 * v)),
                    int(hue_base[2] * (0.35 + 0.65 * v)),
                )
                bar = pg.BarGraphItem(
                    x=[x0 + (w / 2.0)],
                    width=[w],
                    y0=[y0],
                    height=[1.0],
                    brush=pg.mkBrush(color),
                    pen=pg.mkPen((30, 30, 30), width=0.8),
                )
                plot.addItem(bar)
                txt_color = (255, 255, 255) if v >= 0.55 else (20, 20, 20)
                txt = pg.TextItem(text=f"{v:.2f}", color=txt_color, anchor=(0.5, 0.5))
                txt.setPos(x0 + (w / 2.0), y0 + 0.5)
                plot.addItem(txt)

    def _cfg(self, plot: Any, title: str) -> None:
        plot.setTitle(title)
        plot.showGrid(x=True, y=True, alpha=0.15)
        plot.setMenuEnabled(False)
        plot.setMouseEnabled(x=False, y=False)
        plot.setXRange(-self.shown_context_sec, 0.0, padding=0.0)
        plot.hideButtons()

    def _init_fig(self, result: Dict[str, Any]) -> None:
        import pyqtgraph as pg

        special_keys = [
            "x1",
            "x2",
            "p_now",
            "p_future",
            "p_bins",
            "vad",
            "p_bins_now",
            "p_bins_future",
            "silero_vad_score",
        ]
        _skip_plot_keys = frozenset(
            {
                "nod_repetitions_pred",
                "nod_swing_up_pred",
                "bin_times",
            }
        )
        self.keys = [k for k in special_keys if k in result] + [
            k
            for k in result.keys()
            if k not in special_keys and k != "t" and k not in _skip_plot_keys
        ]
        self.graph.clear()
        self.plots.clear()
        self.curves.clear()
        self.data_buffer.clear()

        for row, key in enumerate(self.keys):
            p = self.graph.addPlot(row=row, col=0)
            self.plots[key] = p
            val = result[key]
            if not isinstance(val, (int, float)):
                val = np.squeeze(np.array(val))

            if key == "x1":
                self._cfg(p, "Input waveform 1")
                p.setYRange(-1.0, 1.0, padding=0.0)
                buf = np.zeros(self.MAX_CONTEXT_WAV_LEN, dtype=float)
                c = p.plot(self._x_wav, buf, pen=pg.mkPen((220, 200, 40), width=1.0))
                try:
                    c.setClipToView(True)
                except Exception:
                    pass
                self.curves[key] = c
                self.data_buffer[key] = list(buf)
            elif key == "x2":
                self._cfg(p, "Input waveform 2")
                p.setYRange(-1.0, 1.0, padding=0.0)
                buf = np.zeros(self.MAX_CONTEXT_WAV_LEN, dtype=float)
                c = p.plot(self._x_wav, buf, pen=pg.mkPen((80, 160, 240), width=1.0))
                try:
                    c.setClipToView(True)
                except Exception:
                    pass
                self.curves[key] = c
                self.data_buffer[key] = list(buf)
            elif key in ("p_now", "p_future"):
                t = "p_now (short-term)" if key == "p_now" else "p_future (long-term)"
                self._cfg(p, t)
                p.setYRange(0.0, 1.0, padding=0.0)
                buf = np.ones(self.MAX_CONTEXT_LEN, dtype=float) * 0.5
                arr = np.array(buf, dtype=float)
                x = self._x_ctx
                _, hi, lo = self._split_hi_lo(x, arr, 0.5)
                r, g, b = (245, 189, 0) if key == "p_now" else (245, 120, 0)
                hi_c = p.plot(
                    x,
                    hi,
                    pen=pg.mkPen(r, g, b, width=1.5),
                    fillLevel=0.5,
                    brush=self._pg.mkBrush(r, g, b, 90),
                )
                lo_c = p.plot(
                    x,
                    lo,
                    pen=pg.mkPen((80, 160, 240), width=1.5),
                    fillLevel=0.5,
                    brush=self._pg.mkBrush(80, 160, 240, 90),
                )
                try:
                    hi_c.setClipToView(True)
                    lo_c.setClipToView(True)
                except Exception:
                    pass
                self.curves[key] = {"hi": hi_c, "lo": lo_c}
                self.data_buffer[key] = list(buf)
            elif key == "vad":
                self._cfg(p, "Voice Activity Detection (VAD)")
                p.setYRange(-1.0, 1.0, padding=0.0)
                b1 = np.zeros(self.MAX_CONTEXT_LEN, dtype=float)
                b2 = np.zeros(self.MAX_CONTEXT_LEN, dtype=float)
                c1 = p.plot(
                    self._x_ctx,
                    b1,
                    pen=pg.mkPen((245, 189, 0), width=1.5),
                    fillLevel=0.0,
                    brush=self._pg.mkBrush(245, 189, 0, 90),
                )
                c2 = p.plot(
                    self._x_ctx,
                    -b2,
                    pen=pg.mkPen((80, 160, 240), width=1.5),
                    fillLevel=0.0,
                    brush=self._pg.mkBrush(80, 160, 240, 90),
                )
                self.curves[key] = (c1, c2)
                self.data_buffer[key] = [list(b1), list(b2)]
            elif key in ("p_bins_now", "p_bins_future"):
                title = (
                    "p_bins_now (bins 0–1 sum ÷2, spk ∈ [0,1])"
                    if key == "p_bins_now"
                    else "p_bins_future (bins 2–3 sum ÷2, spk ∈ [0,1])"
                )
                self._cfg(p, title)
                p.setYRange(-1.0, 1.0, padding=0.0)
                b1 = np.zeros(self.MAX_CONTEXT_LEN, dtype=float)
                b2 = np.zeros(self.MAX_CONTEXT_LEN, dtype=float)
                c1 = p.plot(
                    self._x_ctx,
                    b1,
                    pen=pg.mkPen((245, 189, 0), width=1.5),
                    fillLevel=0.0,
                    brush=self._pg.mkBrush(245, 189, 0, 90),
                )
                c2 = p.plot(
                    self._x_ctx,
                    -b2,
                    pen=pg.mkPen((80, 160, 240), width=1.5),
                    fillLevel=0.0,
                    brush=self._pg.mkBrush(80, 160, 240, 90),
                )
                self.curves[key] = (c1, c2)
                self.data_buffer[key] = [list(b1), list(b2)]
            elif key == "silero_vad_score":
                self._cfg(p, "Silero VAD score")
                p.setYRange(0.0, 1.0, padding=0.0)
                buf = np.zeros(self.MAX_CONTEXT_LEN, dtype=float)
                c = p.plot(
                    self._x_ctx,
                    buf,
                    pen=pg.mkPen((236, 112, 99), width=1.8),
                    fillLevel=0.0,
                    brush=self._pg.mkBrush(236, 112, 99, 70),
                )
                self.curves[key] = c
                self.data_buffer[key] = list(buf)
            elif key == "nod_range":
                self._cfg(p, "nod_range (0.035–0.15)")
                p.setYRange(0.035, 0.15, padding=0.0)
                buf = np.zeros(self.MAX_CONTEXT_LEN, dtype=float)
                c = p.plot(self._x_ctx, buf, pen=pg.mkPen((180, 140, 220), width=1.6))
                self.curves[key] = c
                self.data_buffer[key] = list(buf)
            elif key == "nod_speed":
                self._cfg(p, "nod_speed (0.08–0.25)")
                p.setYRange(0.08, 0.25, padding=0.0)
                buf = np.zeros(self.MAX_CONTEXT_LEN, dtype=float)
                c = p.plot(self._x_ctx, buf, pen=pg.mkPen((100, 200, 220), width=1.6))
                self.curves[key] = c
                self.data_buffer[key] = list(buf)
            elif key == "nod_repetitions":
                self._cfg(p, "nod_repetitions (1 / 2 / 3+)")
                p.setYRange(-0.15, 2.15, padding=0.0)
                buf = np.zeros(self.MAX_CONTEXT_LEN, dtype=float)
                c = p.plot(self._x_ctx, buf, pen=pg.mkPen((255, 170, 50), width=2.0))
                self.curves[key] = c
                self.data_buffer[key] = list(buf)
            elif key == "nod_swing_up":
                self._cfg(p, "nod_swing_up")
                p.setYRange(0.0, 1.0, padding=0.0)
                buf = np.zeros(self.MAX_CONTEXT_LEN, dtype=float)
                c = p.plot(
                    self._x_ctx,
                    buf,
                    pen=pg.mkPen((240, 100, 160), width=1.5),
                    fillLevel=0.0,
                    brush=self._pg.mkBrush(240, 100, 160, 80),
                )
                self.curves[key] = c
                self.data_buffer[key] = list(buf)
            elif key == "p_bins":
                arr = np.asarray(val, dtype=float)
                self.data_buffer[key] = arr
                self.curves[key] = None
                self._draw_p_bins_pg(p, arr, result.get("bin_times"))
            elif key.startswith("p_"):
                self._cfg(p, key)
                p.setYRange(0.0, 1.0, padding=0.0)
                buf = np.zeros(self.MAX_CONTEXT_LEN, dtype=float)
                r, g, b = self._rgb_for_p_scalar(key)
                c = p.plot(
                    self._x_ctx,
                    buf,
                    pen=pg.mkPen((r, g, b), width=1.5),
                    fillLevel=0.0,
                    brush=self._pg.mkBrush(r, g, b, 80),
                )
                self.curves[key] = c
                self.data_buffer[key] = list(buf)
            elif isinstance(val, (np.ndarray, list, tuple)) and np.array(val).ndim == 1 and len(val) > 1:
                p.setTitle(key)
                p.showGrid(x=True, y=True, alpha=0.15)
                p.setMenuEnabled(False)
                buf = list(np.asarray(val, dtype=float).reshape(-1))
                x = np.arange(len(buf), dtype=float)
                c = p.plot(x, buf, pen=pg.mkPen((120, 220, 120), width=1.0))
                self.curves[key] = c
                self.data_buffer[key] = buf
                p.setXRange(0, max(1, len(buf) - 1), padding=0.0)
            elif isinstance(val, (int, float, np.floating, np.integer)):
                self._cfg(p, f"{key} (scalar)")
                p.setYRange(0.0, 1.0, padding=0.0)
                c = p.plot([0.0], [float(val)], pen=pg.mkPen((255, 165, 0), width=2))
                self.curves[key] = c
                self.data_buffer[key] = float(val)
            else:
                p.setTitle(key)
                p.hideAxis("left")

        self._root.show()
        self.initialized = True

    def update(self, result: Dict[str, Any]) -> None:
        """どのスレッドから呼んでも安全。"""
        if _gui_plot_use_dedicated_qt_thread:
            _gui_plot_run_on_qt_owner(lambda: self._update_impl(result))
            return
        from PyQt5.QtCore import QThread

        if self._update_bridge is not None and QThread.currentThread() is not self._root.thread():
            self._update_bridge.request.emit(result)
            return
        self._update_impl(result)

    def _update_impl(self, result: Dict[str, Any]) -> None:
        from PyQt5.QtWidgets import QApplication

        draw = True
        if self.use_fixed_draw_rate:
            now = time.time()
            if now - self._last_draw_time < 0.2:
                draw = False
            else:
                self._last_draw_time = now

        if not self.initialized:
            self._init_fig(result)

        for key in self.keys:
            if key not in result:
                continue
            val = result[key]
            if key in ("x1", "x2") and key in self.curves:
                buf = self.data_buffer[key]
                frame_samples = max(1, int(round(self.sample_rate / self.frame_rate)))
                v = np.squeeze(np.array(val)).reshape(-1)
                tail = v[-frame_samples:] if v.size else np.array([], dtype=float)
                buf = buf + list(tail)
                if len(buf) > self.MAX_CONTEXT_WAV_LEN:
                    buf = buf[-self.MAX_CONTEXT_WAV_LEN :]
                self.data_buffer[key] = buf
                if draw:
                    x = np.linspace(-self.shown_context_sec, 0.0, len(buf))
                    self.curves[key].setData(x, np.asarray(buf, dtype=float))
            elif key in ("p_now", "p_future") and key in self.curves:
                buf = self.data_buffer[key]
                try:
                    fv = float(np.asarray(val).reshape(-1)[0])
                except Exception:
                    fv = 0.0
                fv = float(max(0.0, min(1.0, fv)))
                buf = buf + [fv]
                if len(buf) > self.MAX_CONTEXT_LEN:
                    buf = buf[-self.MAX_CONTEXT_LEN :]
                self.data_buffer[key] = buf
                if draw:
                    x = np.linspace(-self.shown_context_sec, 0.0, len(buf))
                    arr = np.asarray(buf, dtype=float)
                    _, hi, lo = self._split_hi_lo(x, arr, 0.5)
                    self.curves[key]["hi"].setData(x, hi)
                    self.curves[key]["lo"].setData(x, lo)
            elif key == "vad" and key in self.curves:
                b1, b2 = self.data_buffer[key]
                vad1, vad2 = result[key]
                b1 = list(b1) + [float(vad1)]
                b2 = list(b2) + [float(vad2)]
                if len(b1) > self.MAX_CONTEXT_LEN:
                    b1 = b1[-self.MAX_CONTEXT_LEN :]
                if len(b2) > self.MAX_CONTEXT_LEN:
                    b2 = b2[-self.MAX_CONTEXT_LEN :]
                self.data_buffer[key] = [b1, b2]
                if draw:
                    x = np.linspace(-self.shown_context_sec, 0.0, len(b1))
                    a1 = np.asarray(b1, dtype=float)
                    a2 = np.asarray(b2, dtype=float)
                    c1, c2 = self.curves[key]
                    c1.setData(x, np.maximum(a1, 0.0))
                    c2.setData(x, -np.maximum(a2, 0.0))
            elif key in ("p_bins_now", "p_bins_future") and key in self.curves:
                b1, b2 = self.data_buffer[key]
                arr = np.asarray(val, dtype=float).reshape(-1)
                v1 = float(arr[0]) if arr.size > 0 else 0.0
                v2 = float(arr[1]) if arr.size > 1 else 0.0
                v1 = max(0.0, v1)
                v2 = max(0.0, v2)
                b1 = list(b1) + [v1]
                b2 = list(b2) + [v2]
                if len(b1) > self.MAX_CONTEXT_LEN:
                    b1 = b1[-self.MAX_CONTEXT_LEN :]
                if len(b2) > self.MAX_CONTEXT_LEN:
                    b2 = b2[-self.MAX_CONTEXT_LEN :]
                self.data_buffer[key] = [b1, b2]
                if draw:
                    x = np.linspace(-self.shown_context_sec, 0.0, len(b1))
                    a1 = np.asarray(b1, dtype=float)
                    a2 = np.asarray(b2, dtype=float)
                    c1, c2 = self.curves[key]
                    c1.setData(x, np.maximum(a1, 0.0))
                    c2.setData(x, -np.maximum(a2, 0.0))
            elif key == "silero_vad_score" and key in self.curves:
                buf = self.data_buffer[key]
                try:
                    fv = float(val)
                except Exception:
                    fv = 0.0
                buf = buf + [fv]
                if len(buf) > self.MAX_CONTEXT_LEN:
                    buf = buf[-self.MAX_CONTEXT_LEN :]
                self.data_buffer[key] = buf
                if draw:
                    x = np.linspace(-self.shown_context_sec, 0.0, len(buf))
                    self.curves[key].setData(x, np.clip(np.asarray(buf, dtype=float), 0.0, 1.0))
            elif key in ("nod_range", "nod_speed", "nod_repetitions", "nod_swing_up") and key in self.curves:
                buf = self.data_buffer[key]
                ns = self._nod_scalar(key, val)
                buf = buf + [ns]
                if len(buf) > self.MAX_CONTEXT_LEN:
                    buf = buf[-self.MAX_CONTEXT_LEN :]
                self.data_buffer[key] = buf
                if draw:
                    x = np.linspace(-self.shown_context_sec, 0.0, len(buf))
                    self.curves[key].setData(x, np.asarray(buf, dtype=float))
            elif key.startswith("p_") and key not in (
                "p_now",
                "p_future",
                "p_bins",
                "p_bins_now",
                "p_bins_future",
            ) and key in self.curves:
                buf = self.data_buffer[key]
                try:
                    fv = float(val)
                except Exception:
                    fv = 0.0
                buf = buf + [fv]
                if len(buf) > self.MAX_CONTEXT_LEN:
                    buf = buf[-self.MAX_CONTEXT_LEN :]
                self.data_buffer[key] = buf
                if draw:
                    x = np.linspace(-self.shown_context_sec, 0.0, len(buf))
                    self.curves[key].setData(x, np.clip(np.asarray(buf, dtype=float), 0.0, 1.0))
            elif key == "p_bins" and key in self.plots:
                arr = np.asarray(val, dtype=float)
                self.data_buffer[key] = arr
                if draw:
                    self._draw_p_bins_pg(self.plots[key], arr, result.get("bin_times"))
            elif key in self.curves and self.curves[key] is not None:
                buf = self.data_buffer[key]
                if isinstance(val, (np.ndarray, list, tuple)) and len(np.asarray(val).reshape(-1)) > 1:
                    buf = list(buf) + list(np.asarray(val, dtype=float).reshape(-1))
                    if len(buf) > self.MAX_CONTEXT_WAV_LEN:
                        buf = buf[-self.MAX_CONTEXT_WAV_LEN :]
                    self.data_buffer[key] = buf
                    if draw:
                        x = np.arange(len(buf), dtype=float)
                        self.curves[key].setData(x, np.asarray(buf, dtype=float))
                else:
                    try:
                        v = float(val)
                    except Exception:
                        v = 0.0
                    self.data_buffer[key] = v
                    if draw:
                        self.curves[key].setData([0.0], [v])

        if draw:
            QApplication.processEvents()