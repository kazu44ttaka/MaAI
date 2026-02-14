import sys
import math
import time
from typing import Dict, Any, List, Union
import numpy as np
import socket
import threading
import time
import pickle
import queue
from . import util
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import seaborn as sns

DEFAULT_VAP_BIN_TIMES_SEC = [0.2, 0.4, 0.6, 0.8]

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


def _rms(values: Union[List[float], tuple]) -> float:
    """RMS値を計算"""
    if not values:
        return 0.0
    return math.sqrt(sum(x * x for x in values) / len(values))


def _format_value(value: Any, max_length: int = 50) -> str:
    """値を適切な形式でフォーマット"""
    if isinstance(value, (list, tuple)):
        if len(value) > 0:
            if isinstance(value[0], (int, float)):
                # 数値のリストの場合、最初の数個の値を表示
                preview = value[:5]
                if len(value) > 5:
                    return f"[{', '.join(f'{v:.3f}' for v in preview)}...] ({len(value)} items)"
                else:
                    return f"[{', '.join(f'{v:.3f}' for v in preview)}]"
            else:
                return f"[{', '.join(str(v) for v in value[:3])}...]" if len(value) > 3 else str(value)
        else:
            return "[]"
    elif isinstance(value, float):
        return f"{value:.4f}"
    elif isinstance(value, int):
        return str(value)
    else:
        str_val = str(value)
        if len(str_val) > max_length:
            return str_val[:max_length-3] + "..."
        return str_val


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
        
        # bar_typeがbalanceのとき、x1/x2の値とバーを2行で横並び表示
        if self.bar_type == "balance" and 'x1' in result and 'x2' in result:
            x1 = np.squeeze(np.array(result['x1'])).tolist()
            x2 = np.squeeze(np.array(result['x2'])).tolist()
            bar1, val1 = _get_bar_for_value('x1', x1, self.bar_length // 2 - 1, "normal")
            bar1 = bar1[::-1]
            bar2, val2 = _get_bar_for_value('x2', x2, self.bar_length // 2 - 1, "normal")
            print(f"x1 │ x2{' ' * 8}: {bar1} │ {bar2} ({val1:.4f}, {val2:.4f})")
        
        # vadがある場合はvad[0]をvad(x1)、vad[1]をvad(x2)として表示
        if 'vad' in result:
            vad1 = result['vad'][0]
            vad2 = result['vad'][1]
            result['vad(x1)'] = vad1
            result['vad(x2)'] = vad2
    
        # 各キーを動的に処理
        for key, value in result.items():
            if key == 't' or key == 'vad':
                continue
            # x1/x2は既に横並びで表示したのでスキップ
            if self.bar_type == "balance" and key in ['x1', 'x2']:
                continue
            if not isinstance(value, (float, int)):
                value = np.squeeze(np.array(value)).tolist()
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
        elif self.mode == 'nod_para':
            vap_result = util.conv_bytearray_2_vapresult_nod_para(data)
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
        elif self.mode == 'nod_para':
            data_sent = util.conv_vapresult_2_bytearray_nod_para(result_dict)
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
    def __init__(self, shown_context_sec: int = 10, frame_rate: int = 10, sample_rate: int = 16000, figsize=(14, 10), use_fixed_draw_rate: bool = True):
        self.figsize = figsize
        self.shown_context_sec = shown_context_sec
        self.frame_rate = frame_rate
        self.sample_rate = sample_rate
        self.MAX_CONTEXT_LEN = frame_rate * shown_context_sec
        self.MAX_CONTEXT_WAV_LEN = sample_rate * shown_context_sec
        self.plt = plt
        self.fig = None
        self.axes = dict()
        self.lines = dict()
        self.images = dict()
        self.patches = dict()
        self.fills = dict()
        self.keys = []
        self.initialized = False
        self.data_buffer = dict()  # key: list or float
        self.use_fixed_draw_rate = use_fixed_draw_rate
        self._last_draw_time = 0.0
        sns.set_theme()
        sns.set_context(font_scale=1.3)

    def _init_fig(self, result: Dict[str, any]):
        special_keys = ['x1', 'x2', 'p_now', 'p_future', 'p_bins', 'vad',
                        'p_nod', 'p_bc', 'nod_count_probs', 'nod_range', 'nod_speed', 'p_swing_up', 'nod_swing_up_value']
        self.keys = [k for k in special_keys if k in result] + [k for k in result.keys() if k not in special_keys and k != 't']
        
        # p_binsが存在する場合、各話者のビン1-2とビン3-4の平均を計算して追加
        if 'p_bins' in result:
            arr = np.array(result['p_bins'], dtype=float)
            if arr.ndim == 2 and arr.shape[0] == 2 and arr.shape[1] >= 4:
                # ビン1-2の平均とビン3-4の平均を計算
                p_bins_spk1_bin12 = float(np.mean(arr[0, 0:2]))
                p_bins_spk1_bin34 = float(np.mean(arr[0, 2:4]))
                p_bins_spk2_bin12 = float(np.mean(arr[1, 0:2]))
                p_bins_spk2_bin34 = float(np.mean(arr[1, 2:4]))
                
                # p_binsの後に追加
                p_bins_idx = self.keys.index('p_bins')
                self.keys.insert(p_bins_idx + 1, 'p_bins_spk1_bin12')
                self.keys.insert(p_bins_idx + 2, 'p_bins_spk1_bin34')
                self.keys.insert(p_bins_idx + 3, 'p_bins_spk2_bin12')
                self.keys.insert(p_bins_idx + 4, 'p_bins_spk2_bin34')
                
                # resultにも追加（初期値として）
                result['p_bins_spk1_bin12'] = p_bins_spk1_bin12
                result['p_bins_spk1_bin34'] = p_bins_spk1_bin34
                result['p_bins_spk2_bin12'] = p_bins_spk2_bin12
                result['p_bins_spk2_bin34'] = p_bins_spk2_bin34
        
        n = len(self.keys)
        self.fig, axs = self.plt.subplots(n, 1, figsize=self.figsize, squeeze=False, tight_layout=True)
        axs = axs.flatten()
        self.axes = {}
        self.lines = {}
        self.images = {}
        self.patches = {}
        self.fills = {}
        self.data_buffer = {}
        tab_colors = list(mcolors.TABLEAU_COLORS.values())
        p_keys = [k for k in self.keys if k.startswith('p_') and k not in ('p_now', 'p_future', 'p_bins')]
        color_map = {k: tab_colors[i % len(tab_colors)] for i, k in enumerate(p_keys)}
        th_map = {k: 0.5 for k in p_keys}
        for i, key in enumerate(self.keys):
            ax = axs[i]
            self.axes[key] = ax
            val = result[key]
            if not isinstance(val, (int, float)):
                val = np.squeeze(np.array(val))
            if key == 'x1':
                time_x1 = np.linspace(-self.shown_context_sec, 0, self.MAX_CONTEXT_WAV_LEN)
                buf = np.zeros(len(time_x1))
                line, = ax.plot(time_x1, buf, c='y')
                self.lines[key] = line
                self.data_buffer[key] = list(buf)
                ax.set_title('Input waveform 1')
                ax.set_ylim(-1, 1)
                ax.set_xlim(-self.shown_context_sec, 0)
            elif key == 'x2':
                time_x2 = np.linspace(-self.shown_context_sec, 0, self.MAX_CONTEXT_WAV_LEN)
                buf = np.zeros(len(time_x2))
                line, = ax.plot(time_x2, buf, c='b')
                self.lines[key] = line
                self.data_buffer[key] = list(buf)
                ax.set_title('Input waveform 2')
                ax.set_ylim(-1, 1)
                ax.set_xlim(-self.shown_context_sec, 0)
            elif key == 'p_now':
                time_x3 = np.linspace(-self.shown_context_sec, 0, self.MAX_CONTEXT_LEN)
                buf = np.ones(len(time_x3)) * 0.5
                fill1 = ax.fill_between(time_x3, y1=0.5, y2=buf, where=buf > 0.5, color='y', interpolate=True)
                fill2 = ax.fill_between(time_x3, y1=buf, y2=0.5, where=buf < 0.5, color='b', interpolate=True)
                self.fills[key] = (fill1, fill2)
                self.data_buffer[key] = list(buf)
                ax.set_title('Output p_now (short-term turn-taking prediction)')
                ax.set_xlim(-self.shown_context_sec, 0)
                ax.set_ylim(0, 1)
            elif key == 'p_future':
                time_x4 = np.linspace(-self.shown_context_sec, 0, self.MAX_CONTEXT_LEN)
                buf = np.ones(len(time_x4)) * 0.5
                fill1 = ax.fill_between(time_x4, y1=0.5, y2=buf, where=buf > 0.5, color='y', interpolate=True)
                fill2 = ax.fill_between(time_x4, y1=buf, y2=0.5, where=buf < 0.5, color='b', interpolate=True)
                self.fills[key] = (fill1, fill2)
                self.data_buffer[key] = list(buf)
                ax.set_title('Output p_future (long-term turn-taking prediction)')
                ax.set_xlim(-self.shown_context_sec, 0)
                ax.set_ylim(0, 1)
            elif key == 'p_bins':
                # p_bins: per-speaker, per-bin probability (0..1), e.g. shape (2, n_bins)
                arr = np.array(val, dtype=float)
                if arr.ndim != 2 or arr.shape[0] != 2:
                    ax.text(0.5, 0.5, f"Invalid p_bins shape: {arr.shape}", ha='center', va='center')
                    ax.set_title('p_bins (invalid)')
                else:
                    n_bins = int(arr.shape[1])
                    # Bin widths are proportional to bin duration (seconds).
                    # Use provided bin_times if present; otherwise default to VAP bins.
                    bin_times = result.get('bin_times', DEFAULT_VAP_BIN_TIMES_SEC)
                    try:
                        bin_times = [float(x) for x in bin_times]
                    except Exception:
                        bin_times = DEFAULT_VAP_BIN_TIMES_SEC
                    if len(bin_times) != n_bins:
                        # Fallback: uniform widths if mismatch
                        bin_times = [1.0 for _ in range(n_bins)]
                    edges = np.concatenate([[0.0], np.cumsum(bin_times)])
                    total_w = float(edges[-1])

                    # 2 rows (speakers) × n_bins segments, colored by probability (0..1 intensity)
                    # Speaker 1: Blues, Speaker 2: Oranges
                    cmap1 = plt.get_cmap('Blues')
                    cmap2 = plt.get_cmap('Oranges')
                    rects = [[None for _ in range(n_bins)] for _ in range(2)]

                    ax.set_xlim(0.0, total_w)
                    ax.set_ylim(0, 2)
                    ax.set_title('p_bins (per-bin probability at current time)')
                    # Show cumulative bin edges on x-axis (e.g., [0.2, 0.6, 1.2, 2.0])
                    xticks = edges[1:]
                    ax.set_xticks(xticks)
                    ax.set_xticklabels([f"{float(t):.1f}" for t in xticks])
                    ax.set_yticks([0.5, 1.5])
                    ax.set_yticklabels(['spk2', 'spk1'])  # top=spk1, bottom=spk2 visually
                    ax.set_xlabel('Future time [s] (bin edges)')
                    ax.set_ylabel('Speaker')
                    ax.tick_params(axis='both', which='both', length=0)

                    # Bin boundary lines (dashed)
                    for x in edges[1:-1]:
                        ax.axvline(float(x), color='k', linestyle='--', linewidth=0.8, alpha=0.4)
                    ax.axhline(1, color='k', linewidth=0.8, alpha=0.4)

                    texts = [[None for _ in range(n_bins)] for _ in range(2)]
                    for s in range(2):
                        for b in range(n_bins):
                            v = float(np.clip(arr[s, b], 0.0, 1.0))
                            cmap = cmap1 if s == 0 else cmap2
                            # y: speaker 0 -> top row (1..2), speaker 1 -> bottom row (0..1)
                            y0 = 1 if s == 0 else 0
                            x0 = float(edges[b])
                            w = float(bin_times[b])
                            r = mpatches.Rectangle(
                                (x0, y0),
                                w,
                                1.0,
                                facecolor=cmap(v),
                                edgecolor=(0, 0, 0, 0.2),
                                linewidth=0.8,
                            )
                            ax.add_patch(r)
                            rects[s][b] = r
                            
                            # 各ビンの中心に確率値を表示
                            text_x = x0 + w / 2
                            text_y = y0 + 0.5
                            # 背景色に応じてテキスト色を調整（暗い色の場合は白、明るい色の場合は黒）
                            text_color = 'white' if v > 0.5 else 'black'
                            t = ax.text(text_x, text_y, f'{v:.2f}', 
                                       ha='center', va='center', 
                                       fontsize=9, fontweight='bold',
                                       color=text_color)
                            texts[s][b] = t

                    # store for updates
                    self.patches[key] = {
                        "rects": rects,
                        "texts": texts,
                        "n_bins": n_bins,
                        "bin_times": bin_times,
                        "cmap1": cmap1,
                        "cmap2": cmap2,
                    }
                    self.data_buffer[key] = arr  # keep last value
            elif key in ['p_bins_spk1_bin12', 'p_bins_spk1_bin34', 'p_bins_spk2_bin12', 'p_bins_spk2_bin34']:
                # p_nowやp_futureと同じ表示形式（0.5を基準にした塗りつぶし）
                time_x = np.linspace(-self.shown_context_sec, 0, self.MAX_CONTEXT_LEN)
                buf = np.ones(len(time_x)) * 0.5
                # 話者1は黄色、話者2は青色
                if 'spk1' in key:
                    fill1 = ax.fill_between(time_x, y1=0.5, y2=buf, where=buf > 0.5, color='y', interpolate=True)
                    fill2 = ax.fill_between(time_x, y1=buf, y2=0.5, where=buf < 0.5, color='b', interpolate=True)
                else:
                    fill1 = ax.fill_between(time_x, y1=0.5, y2=buf, where=buf > 0.5, color='b', interpolate=True)
                    fill2 = ax.fill_between(time_x, y1=buf, y2=0.5, where=buf < 0.5, color='y', interpolate=True)
                self.fills[key] = (fill1, fill2)
                self.data_buffer[key] = list(buf)
                # タイトルを設定
                if key == 'p_bins_spk1_bin12':
                    ax.set_title('p_bins Speaker1 Bin1-2 Average')
                elif key == 'p_bins_spk1_bin34':
                    ax.set_title('p_bins Speaker1 Bin3-4 Average')
                elif key == 'p_bins_spk2_bin12':
                    ax.set_title('p_bins Speaker2 Bin1-2 Average')
                elif key == 'p_bins_spk2_bin34':
                    ax.set_title('p_bins Speaker2 Bin3-4 Average')
                ax.set_xlim(-self.shown_context_sec, 0)
                ax.set_ylim(0, 1)
            elif key == 'nod_count_probs':
                # 頷き回数クラスの確率分布（棒グラフ）
                arr = np.array(val, dtype=float).flatten()
                n_classes = len(arr)
                class_labels = [str(i) for i in range(n_classes)]
                bar_colors = ['#4CAF50', '#2196F3', '#FF9800', '#E91E63', '#9C27B0'][:n_classes]
                bars = ax.bar(class_labels, arr, color=bar_colors)
                self.lines[key] = bars
                self.data_buffer[key] = arr
                ax.set_title('Nod Count Probabilities')
                ax.set_ylim(0, 1)
                ax.set_ylabel('Probability')
                ax.set_xlabel('Count class')
            elif key in ['nod_range', 'nod_speed', 'nod_swing_up_value']:
                # 回帰値の時系列プロット
                time_x = np.linspace(-self.shown_context_sec, 0, self.MAX_CONTEXT_LEN)
                buf = np.zeros(len(time_x))
                color = {'nod_range': '#E91E63', 'nod_speed': '#9C27B0', 'nod_swing_up_value': '#00BCD4'}[key]
                line, = ax.plot(time_x, buf, c=color)
                self.lines[key] = line
                self.data_buffer[key] = list(buf)
                title_map = {'nod_range': 'Nod Range', 'nod_speed': 'Nod Speed', 'nod_swing_up_value': 'Nod Swing Up Value'}
                ax.set_title(title_map[key])
                ax.set_xlim(-self.shown_context_sec, 0)
                # y軸は自動スケーリング（updateで調整）
            elif key.startswith('p_'):
                color = color_map.get(key, 'green')
                th = th_map.get(key, 0.5)
                time_x = np.linspace(-self.shown_context_sec, 0, self.MAX_CONTEXT_LEN)
                buf = np.zeros(len(time_x))
                fill = ax.fill_between(time_x, y1=0.0, y2=buf, color=color, interpolate=True)
                self.fills[key] = fill
                self.data_buffer[key] = list(buf)
                ax.set_title(key)
                ax.set_xlim(-self.shown_context_sec, 0)
                ax.set_ylim(0, th*2)
                ax.axhline(y=th, color='black', linestyle='--')
            elif key == 'vad':
                time_x = np.linspace(-self.shown_context_sec, 0, self.MAX_CONTEXT_LEN)
                buf1 = np.zeros(self.MAX_CONTEXT_LEN)
                buf2 = np.zeros(self.MAX_CONTEXT_LEN)
                fill1 = ax.fill_between(time_x, y1=0, y2=buf1, where=buf1>0, color='y', alpha=0.7, label='VAD1', interpolate=True)
                fill2 = ax.fill_between(time_x, y1=0, y2=-buf2, where=buf2>0, color='b', alpha=0.7, label='VAD2', interpolate=True)
                self.fills[key] = (fill1, fill2)
                self.data_buffer[key] = [list(buf1), list(buf2)]
                ax.set_title('Voice Activity Detection (VAD)')
                ax.set_ylabel('VAD2  VAD1')
                ax.set_ylim(-1, 1)
                ax.set_xlim(-self.shown_context_sec, 0)
                ax.axhline(0, color='black', linestyle='--', linewidth=1)
                ax.legend(loc='upper right')
            elif isinstance(val, (np.ndarray, list, tuple)) and np.array(val).ndim == 1 and len(val) > 1:
                x = np.arange(len(val))
                line, = ax.plot(x, val, c='g')
                self.lines[key] = line
                self.data_buffer[key] = list(val)
                ax.set_title(key)
                ax.set_xlim(0, len(val))
            elif isinstance(val, (int, float, np.floating, np.integer)):
                bar = ax.bar([key], [val], color='orange')
                self.lines[key] = bar
                self.data_buffer[key] = float(val)
                ax.set_title(key)
                ax.set_ylim(0, 1)
            else:
                ax.text(0.5, 0.5, str(val), ha='center', va='center')
                ax.set_title(key)
            # By default, only show x tick labels on the last subplot to reduce clutter.
            # Exception: p_bins uses x-axis as future-time bin edges, so it should always show tick labels.
            if key == 'p_bins':
                ax.tick_params(labelbottom=True)
            elif i == len(self.keys) - 1:
                ax.set_xlabel('Time [s]')
                ax.tick_params(labelbottom=True)
            else:
                ax.tick_params(labelbottom=False)
        self.fig.tight_layout()
        self.plt.ion()
        self.plt.show()
        self.initialized = True

    def update(self, result: Dict[str, any]):
        import time
        draw = True
        if self.use_fixed_draw_rate:
            now = time.time()
            if now - self._last_draw_time < 0.2:
                draw = False
            else:
                self._last_draw_time = now
        if not self.initialized:
            self._init_fig(result)
        tab_colors = list(mcolors.TABLEAU_COLORS.values())
        p_keys = [k for k in self.keys if k.startswith('p_') and k not in ('p_now', 'p_future')]
        color_map = {k: tab_colors[i % len(tab_colors)] for i, k in enumerate(p_keys)}
        for key in self.keys:
            if key not in result:
                continue
            val = result[key]
            if key in ['x1', 'x2'] and key in self.lines:
                buf = self.data_buffer[key]
                val = val[-self.sample_rate // self.frame_rate:]
                buf = buf + list(val)
                if len(buf) > self.MAX_CONTEXT_WAV_LEN:
                    buf = buf[-self.MAX_CONTEXT_WAV_LEN:]
                self.data_buffer[key] = buf
                if draw:
                    time_x = np.linspace(-self.shown_context_sec, 0, self.MAX_CONTEXT_WAV_LEN)
                    self.lines[key].set_data(time_x, buf)
            elif (key == 'p_now' or key == 'p_future') and key in self.fills:
                buf = self.data_buffer[key]
                buf = buf + [float(val[0])]
                if len(buf) > self.MAX_CONTEXT_LEN:
                    buf = buf[-self.MAX_CONTEXT_LEN:]
                self.data_buffer[key] = buf
                if draw:
                    time_x = np.linspace(-self.shown_context_sec, 0, self.MAX_CONTEXT_LEN)
                    ax = self.axes[key]
                    arr = np.array(buf)
                    fills = self.fills[key]
                    for f in fills:
                        if f is not None:
                            f.remove()
                    fill1 = ax.fill_between(time_x, y1=0.5, y2=arr, where=arr > 0.5, color='y', interpolate=True)
                    fill2 = ax.fill_between(time_x, y1=arr, y2=0.5, where=arr < 0.5, color='b', interpolate=True)
                    self.fills[key] = [fill1, fill2]
            elif key in ['p_bins_spk1_bin12', 'p_bins_spk1_bin34', 'p_bins_spk2_bin12', 'p_bins_spk2_bin34'] and key in self.fills:
                buf = self.data_buffer[key]
                # valはスカラー値なので、直接floatに変換
                val_float = float(val) if isinstance(val, (int, float, np.floating, np.integer)) else float(val[0])
                buf = buf + [val_float]
                if len(buf) > self.MAX_CONTEXT_LEN:
                    buf = buf[-self.MAX_CONTEXT_LEN:]
                self.data_buffer[key] = buf
                if draw:
                    time_x = np.linspace(-self.shown_context_sec, 0, self.MAX_CONTEXT_LEN)
                    ax = self.axes[key]
                    arr = np.array(buf)
                    fills = self.fills[key]
                    for f in fills:
                        if f is not None:
                            f.remove()
                    # 話者1は黄色が上、話者2は青色が上
                    if 'spk1' in key:
                        fill1 = ax.fill_between(time_x, y1=0.5, y2=arr, where=arr > 0.5, color='y', interpolate=True)
                        fill2 = ax.fill_between(time_x, y1=arr, y2=0.5, where=arr < 0.5, color='b', interpolate=True)
                    else:
                        fill1 = ax.fill_between(time_x, y1=0.5, y2=arr, where=arr > 0.5, color='b', interpolate=True)
                        fill2 = ax.fill_between(time_x, y1=arr, y2=0.5, where=arr < 0.5, color='y', interpolate=True)
                    self.fills[key] = [fill1, fill2]
            elif key == 'nod_count_probs' and key in self.lines:
                # 頷き回数クラスの確率分布を更新
                arr = np.array(val, dtype=float).flatten()
                self.data_buffer[key] = arr
                if draw:
                    bars = self.lines[key]
                    for bar, v in zip(bars, arr):
                        bar.set_height(v)
            elif key in ['nod_range', 'nod_speed', 'nod_swing_up_value'] and key in self.lines:
                # 回帰値の時系列を更新
                buf = self.data_buffer[key]
                buf = buf + [float(val)]
                if len(buf) > self.MAX_CONTEXT_LEN:
                    buf = buf[-self.MAX_CONTEXT_LEN:]
                self.data_buffer[key] = buf
                if draw:
                    time_x = np.linspace(-self.shown_context_sec, 0, self.MAX_CONTEXT_LEN)
                    self.lines[key].set_data(time_x, buf)
                    ax = self.axes[key]
                    ax.relim()
                    ax.autoscale_view()
            elif key == 'p_bins' and key in self.patches:
                arr = np.array(val, dtype=float)
                if arr.ndim == 2 and arr.shape[0] == 2:
                    meta = self.patches[key]
                    n_bins_expected = int(meta.get("n_bins", arr.shape[1]))
                    if arr.shape[1] != n_bins_expected:
                        # If bin count changes at runtime, easiest is to re-init the figure
                        # (rare; avoids complex patch reconstruction in-place)
                        self.initialized = False
                        break
                    # If bin_times are provided and changed, re-init to rebuild geometry
                    if 'bin_times' in result:
                        try:
                            bt = [float(x) for x in result.get('bin_times')]
                        except Exception:
                            bt = None
                        if bt is not None and bt != meta.get("bin_times"):
                            self.initialized = False
                            break
                    rects = meta["rects"]
                    texts = meta.get("texts", [[None for _ in range(n_bins_expected)] for _ in range(2)])
                    cmap1 = meta["cmap1"]
                    cmap2 = meta["cmap2"]
                    for s in range(2):
                        cmap = cmap1 if s == 0 else cmap2
                        for b in range(n_bins_expected):
                            v = float(np.clip(arr[s, b], 0.0, 1.0))
                            rects[s][b].set_facecolor(cmap(v))
                            # テキストも更新
                            if texts[s][b] is not None:
                                texts[s][b].set_text(f'{v:.2f}')
                                # 背景色に応じてテキスト色を調整
                                text_color = 'white' if v > 0.5 else 'black'
                                texts[s][b].set_color(text_color)
                    self.data_buffer[key] = arr
                    
                    # 各話者のビン1-2とビン3-4の平均を計算して更新
                    if arr.shape[1] >= 4:
                        p_bins_spk1_bin12 = float(np.mean(arr[0, 0:2]))
                        p_bins_spk1_bin34 = float(np.mean(arr[0, 2:4]))
                        p_bins_spk2_bin12 = float(np.mean(arr[1, 0:2]))
                        p_bins_spk2_bin34 = float(np.mean(arr[1, 2:4]))
                        
                        # 平均値をresultに追加（次のループで処理される）
                        result['p_bins_spk1_bin12'] = p_bins_spk1_bin12
                        result['p_bins_spk1_bin34'] = p_bins_spk1_bin34
                        result['p_bins_spk2_bin12'] = p_bins_spk2_bin12
                        result['p_bins_spk2_bin34'] = p_bins_spk2_bin34
            elif key.startswith('p_') and key in self.fills:
                buf = self.data_buffer[key]
                color = color_map.get(key, 'green')
                buf = buf + [float(val)]
                if len(buf) > self.MAX_CONTEXT_LEN:
                    buf = buf[-self.MAX_CONTEXT_LEN:]
                self.data_buffer[key] = buf
                if draw:
                    time_x = np.linspace(-self.shown_context_sec, 0, self.MAX_CONTEXT_LEN)
                    ax = self.axes[key]
                    arr = np.array(buf)
                    f = self.fills[key]
                    if f is not None:
                        f.remove()
                    fill = ax.fill_between(time_x, y1=0.0, y2=arr, color=color, interpolate=True)
                    self.fills[key] = fill
            elif key == 'vad' and key in self.fills:
                buf1, buf2 = self.data_buffer[key]
                vad1, vad2 = result[key]
                buf1 = list(buf1) + [float(vad1)]
                buf2 = list(buf2) + [float(vad2)]
                if len(buf1) > self.MAX_CONTEXT_LEN:
                    buf1 = buf1[-self.MAX_CONTEXT_LEN:]
                if len(buf2) > self.MAX_CONTEXT_LEN:
                    buf2 = buf2[-self.MAX_CONTEXT_LEN:]
                self.data_buffer[key] = [buf1, buf2]
                if draw:
                    time_x = np.linspace(-self.shown_context_sec, 0, self.MAX_CONTEXT_LEN)
                    ax = self.axes[key]
                    arr1 = np.array(buf1)
                    arr2 = np.array(buf2)
                    fills = self.fills[key]
                    for f in fills:
                        if f is not None:
                            f.remove()
                    fill1 = ax.fill_between(time_x, y1=0, y2=arr1, where=arr1>0, color='y', alpha=0.7, label='VAD1', interpolate=True)
                    fill2 = ax.fill_between(time_x, y1=0, y2=-arr2, where=arr2>0, color='b', alpha=0.7, label='VAD2', interpolate=True)
                    self.fills[key] = [fill1, fill2]
            elif key in self.lines:
                buf = self.data_buffer[key]
                if isinstance(val, (np.ndarray, list, tuple)) and len(val) > 1:
                    buf = list(buf) + list(val)
                    if len(buf) > self.MAX_CONTEXT_WAV_LEN: # Changed from self.maxlen to self.MAX_CONTEXT_WAV_LEN
                        buf = buf[-self.MAX_CONTEXT_WAV_LEN:]
                    self.data_buffer[key] = buf
                    if draw:
                        x = np.arange(len(buf))
                        self.lines[key].set_data(x, buf)
                        self.axes[key].set_xlim(0, len(buf))
                        self.axes[key].relim()
                        self.axes[key].autoscale_view()
                else:
                    v = float(val)
                    self.data_buffer[key] = v
                    if draw:
                        self.lines[key][0].set_height(v)
        if draw:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()