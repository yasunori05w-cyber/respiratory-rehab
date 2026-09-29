import flet as ft
import numpy as np
import sounddevice as sd
import librosa
import matplotlib
import matplotlib.pyplot as plt
import time
import io
import base64

# バックエンドを非GUIに設定（Flet描画用）
matplotlib.use("agg")

# --- 共通設定値 ---
FS = 44100
FRAME_LENGTH = 2048
HOP_LENGTH = 512

# 【最強の解決策】Matplotlibのグラフを画像(Base64)に変換する関数
def fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig) # メモリ解放
    # 画像としてFletに返す（バージョンに依存せず絶対に表示される）
    return ft.Image(src_base64=img_str, expand=True, fit=ft.ImageFit.CONTAIN)

def main(page: ft.Page):
    page.title = "総合 呼吸・嚥下リハ支援アプリ"
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.theme_mode = ft.ThemeMode.LIGHT
    page.padding = 10

    page.appbar = ft.AppBar(
        title=ft.Text("呼吸・嚥下アセスメント", weight=ft.FontWeight.BOLD),
        bgcolor=ft.colors.TEAL_700,
        color=ft.colors.WHITE,
        center_title=True,
    )

    # ==========================================
    # 1. スパイロメトリー（肺機能）モード
    # ==========================================
    spiro_status = ft.Text("待機中", weight=ft.FontWeight.BOLD)
    spiro_result = ft.Text("")
    spiro_chart = ft.Container()

    def run_spiro(e):
        duration = 6
        spiro_btn.disabled = True
        spiro_status.value = "🔴 測定中... (6秒間、全力で息を吐ききってください！)"
        spiro_status.color = ft.colors.RED
        spiro_result.value = ""
        spiro_chart.content = None
        page.update()

        audio = sd.rec(int(duration * FS), samplerate=FS, channels=1, dtype='float32')
        sd.wait()
        
        spiro_status.value = "⏳ 解析中..."
        page.update()

        signal = np.abs(audio.flatten())
        window_size = int(FS * 0.1)
        smoothed = np.convolve(signal, np.ones(window_size)/window_size, mode='valid')
        pseudo_flow = smoothed ** 2 
        pseudo_volume = np.cumsum(pseudo_flow) * (1.0 / FS)
        
        fvc = pseudo_volume[-1]
        one_sec_idx = min(FS, len(pseudo_volume) - 1)
        fev1 = pseudo_volume[one_sec_idx]
        ratio = (fev1 / fvc * 100) if fvc > 0 else 0

        res_text = f"推定FVC: {fvc:.4f}\n推定FEV1: {fev1:.4f}\n1秒率 (FEV1/FVC): {ratio:.1f} %\n\n"
        if ratio < 70.0:
            res_text += "⚠️ 閉塞性パターンの疑い (息が素早く吐けていません)"
        else:
            res_text += "✅ 1秒率は正常範囲です"
        spiro_result.value = res_text
        
        fig, ax = plt.subplots(figsize=(6, 3))
        time_axis = np.linspace(0, duration, len(pseudo_volume))
        ax.plot(time_axis, pseudo_volume, color='teal')
        ax.set_title("Pseudo Volume-Time Curve")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Volume (Relative)")
        fig.tight_layout()
        
        spiro_chart.content = fig_to_image(fig) # 画像としてセット
        spiro_status.value = "✅ 測定完了"
        spiro_status.color = ft.colors.BLACK
        spiro_btn.disabled = False
        page.update()

    spiro_btn = ft.ElevatedButton("スパイロ測定開始 (6秒)", icon=ft.icons.AIR, on_click=run_spiro, width=280, height=50, style=ft.ButtonStyle(bgcolor=ft.colors.TEAL_600, color=ft.colors.WHITE))
    
    tab_spiro = ft.Tab(text="スパイロ", icon=ft.icons.AIR, content=ft.Column([ft.Container(height=10), spiro_btn, spiro_status, spiro_result, spiro_chart], horizontal_alignment=ft.CrossAxisAlignment.CENTER))

    # ==========================================
    # 2. 咳の強さ（疑似PCF）モード
    # ==========================================
    cough_status = ft.Text("待機中", weight=ft.FontWeight.BOLD)
    cough_result = ft.Text("")
    cough_chart = ft.Container()

    def run_cough(e):
        duration = 3
        cough_btn.disabled = True
        cough_status.value = "🔴 測定中... (思い切り「ゴホン！」と咳をしてください)"
        cough_status.color = ft.colors.RED
        cough_result.value = ""
        cough_chart.content = None
        page.update()

        audio = sd.rec(int(duration * FS), samplerate=FS, channels=1, dtype='float32')
        sd.wait()
        
        cough_status.value = "⏳ 解析中..."
        page.update()

        audio = audio.flatten()
        rms = librosa.feature.rms(y=audio, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
        peak_energy = np.max(rms)

        res_text = f"最大音量スコア: {peak_energy:.4f}\n\n"
        if peak_energy > 0.15:
            res_text += "✅ 強く有効な咳です。気道防御力が期待できます。"
        elif peak_energy > 0.05:
            res_text += "⚠️ やや弱い咳です。呼吸筋トレーニングを推奨します。"
        else:
            res_text += "❌ 咳が弱い、または無声化の可能性があります。"
        cough_result.value = res_text

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(np.linspace(0, duration, len(audio)), audio, color='darkorange')
        ax.set_title("Cough Power (Waveform)")
        ax.set_xlabel("Time (s)")
        fig.tight_layout()
        
        cough_chart.content = fig_to_image(fig) # 画像としてセット
        cough_status.value = "✅ 測定完了"
        cough_status.color = ft.colors.BLACK
        cough_btn.disabled = False
        page.update()

    cough_btn = ft.ElevatedButton("咳の測定開始 (3秒)", icon=ft.icons.SICK, on_click=run_cough, width=280, height=50, style=ft.ButtonStyle(bgcolor=ft.colors.ORANGE_600, color=ft.colors.WHITE))
    
    tab_cough = ft.Tab(text="咳(PCF)", icon=ft.icons.SICK, content=ft.Column([ft.Container(height=10), cough_btn, cough_status, cough_result, cough_chart], horizontal_alignment=ft.CrossAxisAlignment.CENTER))

    # ==========================================
    # 3. 発声持続時間（疑似MPT）モード
    # ==========================================
    mpt_status = ft.Text("待機中", weight=ft.FontWeight.BOLD)
    mpt_result = ft.Text("")
    mpt_chart = ft.Container()

    def run_mpt(e):
        duration = 10
        mpt_btn.disabled = True
        mpt_status.value = "🔴 測定中... (「アーーー」とできるだけ長く！)"
        mpt_status.color = ft.colors.RED
        mpt_result.value = ""
        mpt_chart.content = None
        page.update()

        audio = sd.rec(int(duration * FS), samplerate=FS, channels=1, dtype='float32')
        sd.wait()
        
        mpt_status.value = "⏳ 解析中..."
        page.update()

        audio = audio.flatten()
        rms = librosa.feature.rms(y=audio, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
        threshold = 0.02
        voice_frames = np.where(rms > threshold)[0]
        
        mpt_seconds = len(voice_frames) * HOP_LENGTH / FS if len(voice_frames) > 0 else 0.0

        res_text = f"推定発声持続時間: 約 {mpt_seconds:.1f} 秒\n\n"
        if mpt_seconds >= 8.0:
            res_text += "✅ 良好です。声帯閉鎖と呼気コントロールが保たれています。"
        else:
            res_text += "⚠️ 息の漏れ、または呼気サポート低下の疑いがあります。"
        mpt_result.value = res_text

        fig, ax = plt.subplots(figsize=(6, 3))
        times = librosa.frames_to_time(np.arange(len(rms)), sr=FS, hop_length=HOP_LENGTH)
        ax.fill_between(times, rms, color='skyblue', alpha=0.7)
        ax.axhline(y=threshold, color='red', linestyle='--', label='Threshold')
        ax.set_title("Phonation Duration (Voice Energy)")
        ax.set_xlabel("Time (s)")
        ax.legend()
        fig.tight_layout()
        
        mpt_chart.content = fig_to_image(fig) # 画像としてセット
        mpt_status.value = "✅ 測定完了"
        mpt_status.color = ft.colors.BLACK
        mpt_btn.disabled = False
        page.update()

    mpt_btn = ft.ElevatedButton("発声測定開始 (10秒)", icon=ft.icons.MIC, on_click=run_mpt, width=280, height=50, style=ft.ButtonStyle(bgcolor=ft.colors.BLUE_600, color=ft.colors.WHITE))

    tab_mpt = ft.Tab(text="発声(MPT)", icon=ft.icons.MIC, content=ft.Column([ft.Container(height=10), mpt_btn, mpt_status, mpt_result, mpt_chart], horizontal_alignment=ft.CrossAxisAlignment.CENTER))

    # ==========================================
    # タブの配置と起動
    # ==========================================
    tabs = ft.Tabs(selected_index=0, animation_duration=300, tabs=[tab_spiro, tab_cough, tab_mpt], expand=1)
    page.add(tabs)

if __name__ == "__main__":
    # 専用画面ではなく、使い慣れたブラウザ（ChromeやEdge）で起動する
    ft.app(target=main, view=ft.AppView.WEB_BROWSER)