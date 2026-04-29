import cv2
import mediapipe as mp
import time
import numpy as np
import os
from collections import deque
from datetime import datetime

import mediapipe as mp
print(mp)
print(mp.__file__)

# ================= INIT =================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Optimasi untuk performa lebih baik
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,  # Diturunkan sedikit untuk deteksi lebih stabil
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

# Set resolusi optimal
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

# Cek apakah kamera terbuka
if not cap.isOpened():
    print("Error: Kamera tidak dapat diakses!")
    exit()

# ================= SETTINGS =================
COLORS = {
    'Right': (0, 255, 0),  # Hijau untuk tangan kanan
    'Left': (255, 80, 0)   # Orange untuk tangan kiri
}
FINGER_NAMES = ["Ibu Jari", "Telunjuk", "Tengah", "Manis", "Kelingking"]
FINGER_NAMES_SHORT = ["J1", "J2", "J3", "J4", "J5"]

# Smoothing parameters
history_len = 8  # Ditambah untuk smoothing lebih baik
finger_history = [deque(maxlen=history_len) for _ in range(5)]
hand_position_history = deque(maxlen=5)  # Untuk stabilisasi posisi

# FPS calculation
pTime = time.time()
fps_history = deque(maxlen=30)

# Frame skip (optimasi)
frame_skip = 1  # Ubah ke 2 untuk performa lebih tinggi
frame_count = 0

# Screenshot
screenshot_dir = "screenshots"
if not os.path.exists(screenshot_dir):
    os.makedirs(screenshot_dir)

# UI Messages
screenshot_msg = ""
screenshot_timer = 0
message_queue = deque(maxlen=3)  # Untuk multiple messages

# Gesture history untuk stabilisasi
gesture_history = deque(maxlen=5)

# Mode display
show_landmarks = True
show_bounding_box = True
show_finger_bars = True

# Kalibrasi (opsional)
calibration_mode = False
calibration_data = {}

# ================= FUNCTIONS =================
def angle(a, b, c):
    """Menghitung sudut antara tiga titik"""
    ba = np.array([a.x - b.x, a.y - b.y])
    bc = np.array([c.x - b.x, c.y - b.y])
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Mencegah nilai di luar range
    return np.degrees(np.arccos(cos_angle))

def distance(p1, p2):
    """Menghitung jarak Euclidean antara dua landmark"""
    return np.hypot(p1.x - p2.x, p1.y - p2.y)

def smooth_fingers(current_fingers):
    """Smoothing finger states dengan voting"""
    for i in range(5):
        finger_history[i].append(current_fingers[i])
    
    smoothed = []
    for i in range(5):
        # Voting: 1 jika lebih dari 50% frame terakhir mendeteksi terlipat
        avg = sum(finger_history[i]) / len(finger_history[i])
        smoothed.append(1 if avg >= 0.5 else 0)
    
    return smoothed

def smooth_gesture(current_gesture):
    """Stabilisasi gesture dengan mode terbanyak"""
    gesture_history.append(current_gesture)
    if len(gesture_history) < 3:
        return current_gesture
    
    # Ambil gesture paling sering muncul
    from collections import Counter
    most_common = Counter(gesture_history).most_common(1)[0][0]
    return most_common

def detect_thumb_improved(lm, hand_label):
    """Deteksi ibu jari dengan metode yang lebih akurat"""
    tip = lm[4]   # Ujung ibu jari
    ip = lm[3]    # Sendi pertama ibu jari
    mcp = lm[2]   # Sendi pangkal ibu jari
    index_tip = lm[8]  # Ujung telunjuk
    wrist = lm[0]  # Pergelangan tangan
    
    corrected_label = "Right" if hand_label == "Left" else "Left"
    
    # Method 1: Perbandingan jarak ke telunjuk
    dist_tip_to_index = distance(tip, index_tip)
    dist_ip_to_index = distance(ip, index_tip)
    m1 = 1 if dist_tip_to_index > dist_ip_to_index * 1.15 else 0
    
    # Method 2: Perbandingan jarak ke pergelangan
    dist_tip_to_wrist = distance(tip, wrist)
    dist_ip_to_wrist = distance(ip, wrist)
    m2 = 1 if dist_tip_to_wrist > dist_ip_to_wrist * 1.1 else 0
    
    # Method 3: Cross product untuk deteksi arah
    vec_mcp_to_ip = np.array([ip.x - mcp.x, ip.y - mcp.y])
    vec_ip_to_tip = np.array([tip.x - ip.x, tip.y - ip.y])
    cross = np.cross(vec_mcp_to_ip, vec_ip_to_tip)
    
    if corrected_label == "Right":
        m3 = 1 if cross < -0.015 else 0
    else:
        m3 = 1 if cross > 0.015 else 0
    
    # Method 4: Posisi horizontal relatif (untuk tambahan akurasi)
    m4 = 1 if (tip.x > mcp.x if corrected_label == "Right" else tip.x < mcp.x) else 0
    
    # Gabungkan dengan pembobotan
    score = m1 + m2 + m3 + m4
    return 1 if score >= 2 else 0

def count_fingers(hand_landmarks, hand_label):
    """Menghitung jumlah jari yang terangkat"""
    lm = hand_landmarks.landmark
    fingers = []
    
    # Ibu jari (deteksi khusus)
    fingers.append(detect_thumb_improved(lm, hand_label))
    
    # 4 jari lainnya menggunakan angle
    finger_tips = [8, 12, 16, 20]  # Ujung jari
    finger_pips = [6, 10, 14, 18]  # Sendi tengah
    finger_mcps = [5, 9, 13, 17]   # Sendi pangkal
    
    for i in range(4):
        ang = angle(lm[finger_mcps[i]], lm[finger_pips[i]], lm[finger_tips[i]])
        # Threshold dinamis berdasarkan posisi
        threshold = 150 if i == 0 else 160  # Telunjuk lebih fleksibel
        fingers.append(1 if ang > threshold else 0)
    
    return fingers, sum(fingers)

def detect_advanced_gesture(fingers):
    """Deteksi gesture yang lebih lengkap dalam Bahasa Indonesia"""
    f = fingers
    
    # Gesture dasar
    if f == [1, 1, 1, 1, 1]:
        return "🖐️ BUKA"
    elif f == [0, 0, 0, 0, 0]:
        return "✊ TINJU"
    elif f == [0, 1, 0, 0, 0]:
        return "☝️ TUNJUK"
    elif f == [1, 0, 0, 0, 1]:
        return "🤘 ROCK"
    elif f == [0, 1, 1, 0, 0]:
        return "✌️ VICTORY"
    elif f == [0, 0, 1, 0, 0]:
        return "🖕 TENGAH"  # Hati-hati dengan gesture ini
    elif f == [1, 1, 0, 0, 0]:
        return "👍 SUKA"
    elif f == [0, 0, 0, 1, 1]:
        return "🤙 TELEPON"
    elif f == [1, 0, 0, 0, 0]:
        return "👍 JEMPOL"
    elif f == [0, 0, 0, 0, 1]:
        return "🤙 KELINGKING"
    
    # Custom gesture untuk angka
    total = sum(f)
    if total == 1:
        idx = f.index(1)
        if idx == 1: return "1️⃣ SATU"
        if idx == 2: return "2️⃣ DUA"
        if idx == 3: return "3️⃣ TIGA"
        if idx == 4: return "4️⃣ EMPAT"
    elif total == 2 and f[1] == 1 and f[2] == 1:
        return "2️⃣ DUA"
    elif total == 3 and f[1] == 1 and f[2] == 1 and f[3] == 1:
        return "3️⃣ TIGA"
    elif total == 4 and f[1] == 1 and f[2] == 1 and f[3] == 1 and f[4] == 1:
        return "4️⃣ EMPAT"
    
    return "❓ TIDAK DIKENAL"

def draw_bounding_box(frame, landmarks, w, h, color, padding=30):
    """Draw bounding box dengan padding yang lebih baik"""
    xs = [int(lm.x * w) for lm in landmarks.landmark]
    ys = [int(lm.y * h) for lm in landmarks.landmark]
    
    x1 = max(0, min(xs) - padding)
    y1 = max(0, min(ys) - padding)
    x2 = min(w, max(xs) + padding)
    y2 = min(h, max(ys) + padding)
    
    # Box dengan sudut melengkung (simulasi)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Tambahkan efek glow
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    
    return x1, y1, x2, y2

def draw_finger_status(frame, x, y, fingers, color):
    """Draw finger status bar dengan visual lebih baik"""
    bar_width = 20
    bar_height = 80
    spacing = 5
    
    for i in range(5):
        bx = x + i * (bar_width + spacing)
        by = y
        
        # Background
        cv2.rectangle(frame, (bx, by), (bx + bar_width, by + bar_height), (50, 50, 50), -1)
        
        # Fill berdasarkan status
        if fingers[i]:
            fill_height = bar_height
            fill_color = color
        else:
            fill_height = int(bar_height * 0.2)
            fill_color = (100, 100, 100)
        
        cv2.rectangle(frame, (bx, by + bar_height - fill_height), 
                     (bx + bar_width, by + bar_height), fill_color, -1)
        
        # Border
        cv2.rectangle(frame, (bx, by), (bx + bar_width, by + bar_height), (255, 255, 255), 1)
        
        # Label
        cv2.putText(frame, FINGER_NAMES_SHORT[i], (bx + 2, by + bar_height + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

def add_message(msg, duration=2):
    """Tambahkan pesan ke antrian"""
    message_queue.append((msg, time.time(), duration))

def calculate_fps():
    """Menghitung FPS dengan smoothing"""
    global pTime
    cTime = time.time()
    fps = 1 / (cTime - pTime + 1e-6)
    pTime = cTime
    fps_history.append(fps)
    return int(sum(fps_history) / len(fps_history))

def draw_ui(frame, w, h, fps, hand_count, total_fingers):
    """Draw UI yang lebih informatif dalam Bahasa Indonesia"""
    # Background panel semi-transparan
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (250, 130), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Informasi
    y_offset = 30
    cv2.putText(frame, f"FPS: {fps}", (15, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    cv2.putText(frame, f"Tangan: {hand_count}", (15, y_offset + 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    cv2.putText(frame, f"Jari: {total_fingers}", (15, y_offset + 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # Kontrol info
    cv2.putText(frame, "Tekan: S(simpan) | L(titik) | B(kotak) | R(reset) | ESC(keluar)",
               (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

def draw_messages(frame, w, h):
    """Draw pesan-pesan sementara"""
    current_time = time.time()
    y_offset = h - 80
    
    for msg, timestamp, duration in message_queue:
        if current_time - timestamp < duration:
            # Hitung ukuran teks untuk background
            (text_w, text_h), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            
            # Background
            cv2.rectangle(frame, (w//2 - text_w//2 - 10, y_offset - text_h - 5),
                         (w//2 + text_w//2 + 10, y_offset + 5), (0, 0, 0), -1)
            cv2.rectangle(frame, (w//2 - text_w//2 - 10, y_offset - text_h - 5),
                         (w//2 + text_w//2 + 10, y_offset + 5), (0, 255, 255), 1)
            
            # Text
            cv2.putText(frame, msg, (w//2 - text_w//2, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y_offset -= 40

# ================= MAIN LOOP =================
print("Sistem Dimulai! Tekan ESC untuk keluar, S untuk screenshot")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Gagal membaca frame dari kamera")
        break
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    frame_count += 1
    if frame_count % frame_skip != 0:
        cv2.imshow("Sistem Pengenalan Gestur Tangan", frame)
        if cv2.waitKey(1) == 27:
            break
        continue
    
    # Konversi ke RGB untuk MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False  # Optimasi
    results = hands.process(rgb)
    rgb.flags.writeable = True
    
    total_fingers_all = 0
    hand_count = 0
    
    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label
            # Ubah label ke Bahasa Indonesia untuk tampilan
            label_display = "Kanan" if label == "Right" else "Kiri"
            color = COLORS[label]
            
            # Draw landmarks dengan style yang lebih baik
            if show_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
            
            # Deteksi jari
            fingers, total = count_fingers(hand_landmarks, label)
            fingers = smooth_fingers(fingers)
            
            # Deteksi gesture
            gesture = detect_advanced_gesture(fingers)
            gesture = smooth_gesture(gesture)
            
            total_fingers_all += total
            hand_count += 1
            
            # Draw bounding box
            if show_bounding_box:
                x1, y1, x2, y2 = draw_bounding_box(frame, hand_landmarks, w, h, color)
            else:
                # Hitung tetap untuk posisi teks
                xs = [int(lm.x * w) for lm in hand_landmarks.landmark]
                ys = [int(lm.y * h) for lm in hand_landmarks.landmark]
                x1, y1 = min(xs) - 30, min(ys) - 30
                x2, y2 = max(xs) + 30, max(ys) + 30
            
            # Label tangan dan jumlah jari
            cv2.putText(frame, f"{label_display} ({total})", (x1, y1 - 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Gesture name dengan background
            (text_w, text_h), _ = cv2.getTextSize(gesture, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(frame, (x1, y1 - text_h - 15), (x1 + text_w + 10, y1 - 10), (0, 0, 0), -1)
            cv2.putText(frame, gesture, (x1 + 5, y1 - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Finger status bars
            if show_finger_bars:
                draw_finger_status(frame, x1, y2 + 10, fingers, color)
    
    # FPS dan UI
    fps = calculate_fps()
    draw_ui(frame, w, h, fps, hand_count, total_fingers_all)
    draw_messages(frame, w, h)
    
    # Tampilkan frame
    cv2.imshow("Sistem Pengenalan Gestur Tangan", frame)
    
    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    
    if key == 27:  # ESC
        add_message("Keluar...", 1)
        break
    
    elif key == ord('s') or key == ord('S'):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{screenshot_dir}/gesture_{timestamp}.png"
        cv2.imwrite(filename, frame)
        add_message(f"Tersimpan: {filename}", 2)
        print(f"Screenshot tersimpan: {filename}")
    
    elif key == ord('l') or key == ord('L'):
        show_landmarks = not show_landmarks
        add_message(f"Titik referensi: {'NYALA' if show_landmarks else 'MATI'}", 1)
    
    elif key == ord('b') or key == ord('B'):
        show_bounding_box = not show_bounding_box
        add_message(f"Kotak batas: {'NYALA' if show_bounding_box else 'MATI'}", 1)
    
    elif key == ord('f') or key == ord('F'):
        show_finger_bars = not show_finger_bars
        add_message(f"Bilah jari: {'NYALA' if show_finger_bars else 'MATI'}", 1)
    
    elif key == ord('r') or key == ord('R'):
        # Reset semua history
        finger_history = [deque(maxlen=history_len) for _ in range(5)]
        gesture_history.clear()
        add_message("Sistem Di-reset!", 1)
        print("System reset")

# Cleanup
print("Membersihkan...")
cap.release()
cv2.destroyAllWindows()
hands.close()
print("Program selesai dengan sukses")