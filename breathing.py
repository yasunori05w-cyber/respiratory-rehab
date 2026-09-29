import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
import soundfile as sf
import io
from audio_recorder_streamlit import audio_recorder

# --- 共通設定 ---
FRAME_LENGTH = 2048
HOP_LENGTH = 512

st.set_page_config(page_title="呼吸・嚥下アセスメント", layout="centered")

st.title("🗣️ 呼吸・嚥下アセスメント")
st.markdown("スマホのマイクを使用して、呼吸機能や嚥下防御力を簡易評価します。")

# タブの作成
tab1, tab2, tab3 = st.tabs(["🌬️ スパイロ", "🤧 咳(PCF)", "🎤 発声(MPT)"])

# 録音データをNumPy配列に変換する共通関数
def process_audio_bytes(audio_bytes):
    # バイトデータをsoundfileで読み込み
    audio, sr = sf.read(io.BytesIO(audio_bytes))
    # ステレオの場合はモノラルに変換
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr

# ==========================================
# 1. スパイロメトリーモード
# ==========================================
with tab1:
    st.subheader("スパイロメトリー（肺機能）")
    st.info("大きく息を吸ってから、マイクアイコンを押して **一気に長く（約6秒間）** 息を吐ききってください。終わったらもう一度アイコンを押して停止します。")
    
    # 【変更箇所】無音で勝手に切れないようにpause_thresholdを追加
    audio_bytes_1 = audio_recorder(
        text="タップで開始 🔴 / もう一度タップで停止 ⬛", 
        pause_threshold=30.0, 
        key="spiro"
    )
    
    if audio_bytes_1:
        st.audio(audio_bytes_1, format="audio/wav")
        audio, sr = process_audio_bytes(audio_bytes_1)
        duration = len(audio) / sr
        
        with st.spinner("解析中..."):
            signal = np.abs(audio)
            window_size = int(sr * 0.1)
            smoothed = np.convolve(signal, np.ones(window_size)/window_size, mode='valid')
            pseudo_flow = smoothed ** 2 
            pseudo_volume = np.cumsum(pseudo_flow) * (1.0 / sr)
            
            fvc = pseudo_volume[-1]
            one_sec_idx = min(sr, len(pseudo_volume) - 1)
            fev1 = pseudo_volume[one_sec_idx]
            ratio = (fev1 / fvc * 100) if fvc > 0 else 0

            st.success("測定完了")
            col1, col2, col3 = st.columns(3)
            col1.metric("推定FVC", f"{fvc:.4f}")
            col2.metric("推定FEV1", f"{fev1:.4f}")
            col3.metric("1秒率", f"{ratio:.1f} %")
            
            if ratio < 70.0:
                st.warning("⚠️ 閉塞性パターンの疑い (息が素早く吐けていません)")
            else:
                st.info("✅ 1秒率は正常範囲です")
                
            fig, ax = plt.subplots(figsize=(6, 3))
            time_axis = np.linspace(0, duration, len(pseudo_volume))
            ax.plot(time_axis, pseudo_volume, color='teal')
            ax.set_title("Pseudo Volume-Time Curve")
            ax.set_xlabel("Time (s)")
            st.pyplot(fig)

# ==========================================
# 2. 咳の強さ（疑似PCF）モード
# ==========================================
with tab2:
    st.subheader("咳の強さ（気道防御力）")
    st.info("お腹に力を入れて、録音中に **「ゴホン！」と1回だけ** 強い咳をしてください。終わったら停止を押します。")
    
    # 【変更箇所】無音で勝手に切れないようにpause_thresholdを追加
    audio_bytes_2 = audio_recorder(
        text="タップで開始 🔴 / もう一度タップで停止 ⬛", 
        pause_threshold=30.0, 
        key="cough"
    )
    
    if audio_bytes_2:
        st.audio(audio_bytes_2, format="audio/wav")
        audio, sr = process_audio_bytes(audio_bytes_2)
        duration = len(audio) / sr
        
        with st.spinner("解析中..."):
            rms = librosa.feature.rms(y=audio, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
            peak_energy = np.max(rms)

            st.success("測定完了")
            st.metric("最大音量スコア", f"{peak_energy:.4f}")
            
            if peak_energy > 0.15:
                st.info("✅ 強く有効な咳です。気道防御力が期待できます。")
            elif peak_energy > 0.05:
                st.warning("⚠️ やや弱い咳です。呼吸筋トレーニングを推奨します。")
            else:
                st.error("❌ 咳が弱い、または無声化の可能性があります。")

            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(np.linspace(0, duration, len(audio)), audio, color='darkorange')
            ax.set_title("Cough Power (Waveform)")
            ax.set_xlabel("Time (s)")
            st.pyplot(fig)

# ==========================================
# 3. 発声持続時間（疑似MPT）モード
# ==========================================
with tab3:
    st.subheader("発声持続時間（声帯閉鎖力）")
    st.info("大きく息を吸ってから、録音中に **「アーーー」とできるだけ長く** 声を出してください。限界が来たら停止を押します。")
    
    # 【変更箇所】無音で勝手に切れないようにpause_thresholdを追加
    audio_bytes_3 = audio_recorder(
        text="タップで開始 🔴 / もう一度タップで停止 ⬛", 
        pause_threshold=30.0, 
        key="mpt"
    )
    
    if audio_bytes_3:
        st.audio(audio_bytes_3, format="audio/wav")
        audio, sr = process_audio_bytes(audio_bytes_3)
        duration = len(audio) / sr
        
        with st.spinner("解析中..."):
            rms = librosa.feature.rms(y=audio, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
            threshold = 0.02
            voice_frames = np.where(rms > threshold)[0]
            
            mpt_seconds = len(voice_frames) * HOP_LENGTH / sr if len(voice_frames) > 0 else 0.0

            st.success("測定完了")
            st.metric("推定発声持続時間", f"{mpt_seconds:.1f} 秒")
            
            if mpt_seconds >= 8.0:
                st.info("✅ 良好です。声帯閉鎖と呼気コントロールが保たれています。")
            else:
                st.warning("⚠️ 息の漏れ、または呼気サポート低下の疑いがあります。")

            fig, ax = plt.subplots(figsize=(6, 3))
            times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=HOP_LENGTH)
            ax.fill_between(times, rms, color='skyblue', alpha=0.7)
            ax.axhline(y=threshold, color='red', linestyle='--', label='Threshold')
            ax.set_title("Phonation Duration (Voice Energy)")
            ax.set_xlabel("Time (s)")
            st.pyplot(fig)
