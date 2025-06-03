import pyaudio
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# 设备枚举（首次运行需验证）
p = pyaudio.PyAudio()
print("Available audio devices:")
for i in range(p.get_device_count()):
    dev = p.get_device_info_by_index(i)
    if dev['maxInputChannels'] > 0:
        print(f"Index {i}: {dev['name']} (Input channels: {dev['maxInputChannels']})")
p.terminate()

# === 新增参数 ===
VOLUME_THRESHOLD = 200  # 音量阈值（根据实测调整）
# ================

# 核心参数（需与设备匹配）
RESPEAKER_RATE = 16000
RESPEAKER_CHANNELS = 6
USED_CHANNELS = 4
RESPEAKER_INDEX = 1
CHUNK = 512

# 算法参数
REMAIN_LENGTH = 10
SEG_NUM = 3
SEG_LEN = 500
OVERLAP = 50
SOUND_V = 343

# 坐标系统
RADIUS = 3
WIDTH = 10
relative_dis = np.zeros(3)  # 确保存在

# 数据结构
cor_res_lists = [[] for _ in range(USED_CHANNELS - 1)]
audio_data = [[] for _ in range(USED_CHANNELS)]
cor_data = [[] for _ in range(USED_CHANNELS)]
M = np.array([[0.045,0],[0.045,0.045],[0,0.045]])
keep_theta = [0]

def tdoa():
    global relative_dis
    for i in range(USED_CHANNELS-1):
        for j in range(SEG_NUM):
            tmp_cor_res = np.correlate(cor_data[0][j*SEG_LEN:(j+1)*SEG_LEN], 
                                     cor_data[i][j*SEG_LEN:(j+1)*SEG_LEN], 
                                     mode='full')
            discrete_res = (np.argmax(tmp_cor_res) - (SEG_LEN - 1)) / RESPEAKER_RATE
            cor_res_lists[i].append(discrete_res)
        cor_res_lists[i] = cor_res_lists[i][-REMAIN_LENGTH:]
        relative_dis[i] = Counter(cor_res_lists[i]).most_common(1)[0][0] * SOUND_V
    S = np.linalg.inv(M.T @ M) @ M.T @ relative_dis
    keep_theta[0] = np.arctan2(S[1], S[0])
    return keep_theta[0]

# 音频初始化
audio = pyaudio.PyAudio()
try:
    stream = audio.open(
        rate=RESPEAKER_RATE,
        format=pyaudio.paInt16,
        channels=RESPEAKER_CHANNELS,
        input=True,
        input_device_index=RESPEAKER_INDEX,
        frames_per_buffer=CHUNK*4,
        start=False
    )
    stream.start_stream()
except Exception as e:
    print(f"无法打开音频流：{e}")
    audio.terminate()
    exit()

# 可视化设置
plt.ion()
fig, ax = plt.subplots(figsize=(5,5))
point, = ax.plot([], [], 'o', color='gold')
circle = plt.Circle((WIDTH/2,WIDTH/2), RADIUS, color='lightgreen', fill=False)
ax.add_artist(circle)
ax.set_xlim(0, WIDTH)
ax.set_ylim(0, WIDTH)

# 主循环
try:
    print("开始录音...")
    read_count, silent_count = 0, 0
    while True:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            read_count = 0
        except OSError as e:
            read_count += 1
            if read_count > 5: break
            print(f"读取错误: {e}")
            continue

        audio_chunk = np.frombuffer(data, dtype=np.int16)
        
        # === 新增音量阈值检测 ===
        # 提取主通道（通道1）并计算平均幅度
        main_channel = audio_chunk[1::RESPEAKER_CHANNELS]  # 调整索引以适应实际使用的通道
        current_volume = np.mean(np.abs(main_channel))
        
        # 静音控制逻辑
        if current_volume < VOLUME_THRESHOLD:
            silent_count += 1
            if silent_count > 3:  # 连续3次静音才清除定位点
                point.set_data([], [])
                plt.draw()
                plt.pause(0.01)
            continue
        else:
            silent_count = 0  # 有效声音时重置计数器
        # ======================

        if len(audio_chunk) < CHUNK * RESPEAKER_CHANNELS:
            print(f"数据不完整: {len(audio_chunk)}")
            continue
            
        for i in range(USED_CHANNELS):
            audio_data[i].extend(audio_chunk[i+1::RESPEAKER_CHANNELS].tolist())

        if len(audio_data[0]) > SEG_LEN * SEG_NUM:
            for i in range(USED_CHANNELS):
                cor_data[i] = audio_data[i][0:SEG_LEN*SEG_NUM]
                audio_data[i] = audio_data[i][SEG_LEN*SEG_NUM-OVERLAP:]
            try:
                theta = tdoa()
                x = [WIDTH/2 - RADIUS*np.cos(theta-np.pi/2)]
                y = [WIDTH/2 - RADIUS*np.sin(theta-np.pi/2)]
                point.set_data(x, y)
                plt.draw()
                plt.pause(0.01)
            except Exception as e:
                print(f"处理错误: {e}")

        if not plt.fignum_exists(fig.number): break

except KeyboardInterrupt:
    print("手动停止")
finally:
    if stream.is_active: stream.stop_stream()
    stream.close()
    audio.terminate()
    plt.ioff()
    plt.close()