#!/bin/bash
# ==============================================================================
# nod_para モデルのテスト用バッチスクリプト
#
# チェックポイント:
#   VapGPT-NOD-PARA_10.0Hz_ad20s_134_maai_medium_lr0.001_bcpw3.0_bclw0.2_
#   nodpw3.0_nodlw1.0_cntlw1.0_rnglw1.0_spdlw1.0_swblw1.0_swvlw0.0_
#   cnt-ce_reg-mse_swu-bv_hMLP11110h128_sEnc128_RAdamF_wd0.05_do0.2_hdo0.3_
#   swpw4.5_swth0.02 / epoch10-val_nod_all_4.33023.ckpt
#
# MAAI エンコーダ:
#   maai_enc/medium_student-epoch=32-val_loss=0.014346.ckpt
# ==============================================================================

set -e

# --- パス設定 ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

LOCAL_MODEL="/n/work5/katou/VAP_Nodding_para/runs/VapGPT-NOD-PARA_10.0Hz_ad20s_134_maai_medium_lr0.001_bcpw3.0_bclw0.2_nodpw3.0_nodlw1.0_cntlw1.0_rnglw1.0_spdlw1.0_swblw1.0_swvlw0.0_cnt-ce_reg-mse_swu-bv_hMLP11110h128_sEnc128_RAdamF_wd0.05_do0.2_hdo0.3_swpw4.5_swth0.02/epoch10-val_nod_all_4.33023.ckpt"
MAAI_CHECKPOINT="/n/work5/katou/VAP_Nodding_para/runs/maai_enc/medium_student-epoch=32-val_loss=0.014346.ckpt"

# --- モデル設定（チェックポイント名から読み取り） ---
#   10.0Hz → frame_rate=10
#   ad20s  → context_len_sec=20
#   134    → channel_layers=1, cross_layers=3, num_heads=4 (VapConfig デフォルト)
#   hMLP11110h128 → count=1, range=1, speed=1, swing_binary=1, swing_value=0, hidden=128
#   sEnc128 → shared_encoder=1, shared_encoder_dim=128
#   cnt-ce  → nod_count_binary=0 (3クラス分類)
#   swu-bv  → swing_up_mode="binary_and_value"
#   do0.2   → dropout=0.2
#   hdo0.3  → nod_head_dropout=0.3
FRAME_RATE=10
CONTEXT_LEN_SEC=20
DEVICE="cuda"   # GPU を使う場合は "cuda" に変更

# --- WAV ファイル ---
WAV1="${SCRIPT_DIR}/../wav_sample/jpn_inoue_16k.wav"
WAV2="${SCRIPT_DIR}/../wav_sample/jpn_sumida_16k.wav"

# --- チェックポイント存在確認 ---
if [ ! -f "$LOCAL_MODEL" ]; then
    echo "[ERROR] PLチェックポイントが見つかりません: $LOCAL_MODEL"
    exit 1
fi
if [ ! -f "$MAAI_CHECKPOINT" ]; then
    echo "[ERROR] MAAIエンコーダチェックポイントが見つかりません: $MAAI_CHECKPOINT"
    exit 1
fi

echo "=============================================="
echo " nod_para テスト"
echo "=============================================="
echo "PLチェックポイント : $LOCAL_MODEL"
echo "MAAIエンコーダ     : $MAAI_CHECKPOINT"
echo "フレームレート     : ${FRAME_RATE} Hz"
echo "コンテキスト長     : ${CONTEXT_LEN_SEC} 秒"
echo "デバイス           : ${DEVICE}"
echo "=============================================="

# テストモード選択
echo ""
echo "テストモードを選択してください:"
echo "  1) ConsoleBar  - 2WAVファイル + コンソールバー表示"
echo "  2) GuiPlot     - 2WAVファイル + GUIプロット表示"
echo "  3) TCP         - 2WAVファイル + TCP送受信テスト"
echo "  4) Mic         - マイク入力 + コンソールバー表示"
echo ""
read -p "選択 [1-4] (デフォルト: 1): " MODE_CHOICE
MODE_CHOICE=${MODE_CHOICE:-1}

case $MODE_CHOICE in
    1)
        echo ""
        echo ">>> ConsoleBar テストを開始します..."
        /n/work5/katou/MaAI/.venv/bin/python "${SCRIPT_DIR}/nod_para_2wav.py" \
            --local_model "$LOCAL_MODEL" \
            --maai_checkpoint "$MAAI_CHECKPOINT" \
            --wav1 "$WAV1" \
            --wav2 "$WAV2" \
            --device "$DEVICE" \
            --frame_rate "$FRAME_RATE" \
            --context_len_sec "$CONTEXT_LEN_SEC"
        ;;
    2)
        echo ""
        echo ">>> GuiPlot テストを開始します..."
        /n/work5/katou/MaAI/.venv/bin/python "${SCRIPT_DIR}/nod_para_2wav_GuiPlot.py" \
            --local_model "$LOCAL_MODEL" \
            --maai_checkpoint "$MAAI_CHECKPOINT" \
            --wav1 "$WAV1" \
            --wav2 "$WAV2" \
            --device "$DEVICE" \
            --frame_rate "$FRAME_RATE" \
            --context_len_sec "$CONTEXT_LEN_SEC"
        ;;
    3)
        echo ""
        echo ">>> TCP テストを開始します..."
        /n/work5/katou/MaAI/.venv/bin/python "${SCRIPT_DIR}/nod_para_2wav_TCP.py" \
            --local_model "$LOCAL_MODEL" \
            --maai_checkpoint "$MAAI_CHECKPOINT" \
            --wav1 "$WAV1" \
            --wav2 "$WAV2" \
            --device "$DEVICE" \
            --frame_rate "$FRAME_RATE" \
            --context_len_sec "$CONTEXT_LEN_SEC"
        ;;
    4)
        echo ""
        echo ">>> マイクテストを開始します..."
        /n/work5/katou/MaAI/.venv/bin/python "${SCRIPT_DIR}/nod_para_mic.py" \
            --local_model "$LOCAL_MODEL" \
            --maai_checkpoint "$MAAI_CHECKPOINT" \
            --device "$DEVICE" \
            --frame_rate "$FRAME_RATE" \
            --context_len_sec "$CONTEXT_LEN_SEC"
        ;;
    *)
        echo "[ERROR] 無効な選択です: $MODE_CHOICE"
        exit 1
        ;;
esac
