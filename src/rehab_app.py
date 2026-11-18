import sys
import time
import threading
import serial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pynput import keyboard

from PyQt5.QtWidgets import (QApplication, QMainWindow, QTextEdit, QVBoxLayout, 
                             QHBoxLayout, QWidget, QPushButton, QLabel, QProgressBar, 
                             QGroupBox, QMessageBox)
from PyQt5.QtGui import QPixmap, QFont, QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# =====================================================================
# [0] 아두이노 시리얼 통신 설정
# =====================================================================
PORT = 'COM3'  
BAUD = 115200

try:
    py_serial = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)
    arduino_connected = True
except Exception as e:
    arduino_connected = False
    print(f"❌ 아두이노 연결 실패: {e}")

arduino_data = {'flex': 512, 'fsr': 0}

# =====================================================================
# [1] 시스템 전역 변수 및 제어 플래그
# =====================================================================
is_running = True
current_mode = "SETUP"  
space_event = threading.Event() 
pain_event = 0  

M1_INIT_POS = 0    
M2_INIT_POS = 180  
m1_limit = 180  
m2_limit = 0    
flex_max_open = 0 
flex_max_close = 1023 
dyn_flex_open = 0
dyn_flex_close = 1023

avg_fsr = 0
initial_k = 20  
bayesian_safety_score = 0.8 
cycle_count = 0 

rehab_data_log = []
patient_pain_log = {'safe': 0, 'pain1': 0, 'pain2': 0}
pain_scatter_points = [] 
rom_growth_log = [] 

# =====================================================================
# [2] LinUCB 알고리즘 & 아두이노 제어
# =====================================================================
class LinUCBAgent:
    def __init__(self, n_arms, n_features, alpha=0.2):
        self.n_arms = n_arms
        self.n_features = n_features
        self.alpha = alpha
        self.A = [np.identity(n_features) for _ in range(n_arms)]
        self.b = [np.zeros((n_features, 1)) for _ in range(n_arms)]
        
    def select_action(self, context):
        x = np.array(context).reshape(-1, 1)
        p = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[a])
            theta_hat = A_inv.dot(self.b[a])
            p[a] = theta_hat.T.dot(x)[0][0] + self.alpha * np.sqrt(x.T.dot(A_inv).dot(x))[0][0]
        return np.argmax(p)

    def update(self, action, context, reward):
        x = np.array(context).reshape(-1, 1)
        self.A[action] += x.dot(x.T)
        self.b[action] += reward * x

linucb_agent = None
SPEED_ARMS = []

def update_bayesian_safety(pain_level):
    global bayesian_safety_score, patient_pain_log
    if pain_level == 2:   
        bayesian_safety_score = max(0.1, bayesian_safety_score * 0.4)
        patient_pain_log['pain2'] += 1
    elif pain_level == 1: 
        bayesian_safety_score = max(0.1, bayesian_safety_score * 0.8)
        patient_pain_log['pain1'] += 1
    elif pain_level == 0: 
        bayesian_safety_score = min(1.0, bayesian_safety_score + 0.05)
        patient_pain_log['safe'] += 1

def read_arduino_data():
    global arduino_data, is_running
    while is_running:
        if arduino_connected and py_serial.in_waiting > 0:
            try:
                line = py_serial.readline().decode('utf-8').strip()
                if line.startswith("DATA,"):
                    parts = line.split(',')
                    arduino_data['flex'] = int(parts[1])
                    arduino_data['fsr'] = int(parts[2])
            except Exception:
                pass
        time.sleep(0.01)

def on_press(key):
    global is_running, pain_event
    try:
        if key == keyboard.Key.space:
            if current_mode == "SETUP" or current_mode == "WAIT_RL":
                space_event.set()
            elif current_mode == "RL":
                pain_event = 3
                is_running = False
                space_event.set()
        elif hasattr(key, 'char'):
            if current_mode == "RL":
                if key.char == '1': pain_event = 1
                elif key.char == '2': pain_event = 2
            elif key.char == 'q':
                is_running = False
                space_event.set()
    except Exception: pass

def send_motor(motor_num, angle, speed=100):
    if arduino_connected:
        cmd = f"M{motor_num},{int(angle)},{int(speed)}\n"
        py_serial.write(cmd.encode('utf-8'))

# =====================================================================
# [3] PyQt5 GUI 백그라운드 Worker Thread
# =====================================================================
class RehabWorker(QThread):
    sig_log = pyqtSignal(str)
    sig_state = pyqtSignal(str, int) 
    sig_stats = pyqtSignal(int, int, int) 
    sig_phase = pyqtSignal(str) # 버튼 텍스트 변경용 시그널
    sig_finish = pyqtSignal()

    def run(self):
        if not arduino_connected:
            self.sig_log.emit("⚠️ 아두이노 미연결 (테스트 모드로 작동 중이거나 연결을 확인하세요)")
        
        self.sig_phase.emit("SETUP")
        self.run_setup_scenario()
        if is_running:
            self.rl_rehab_loop()
        
        self.sig_finish.emit()

    def log(self, text):
        self.sig_log.emit(text)

    def run_setup_scenario(self):
        global m1_limit, m2_limit
        global flex_max_open, flex_max_close, initial_k, avg_fsr
        
        self.log("\n========================================")
        self.log(" 🤖 AI 스마트 상지 재활 시스템 초기화")
        self.log("========================================")
        self.log("▶ 시작하려면 [진행 (Space)] 버튼을 누르세요.")
        space_event.clear(); space_event.wait()
        if not is_running: return
        
        self.log("⚙️ 모터 0점 세팅 중...")
        send_motor(1, M1_INIT_POS); send_motor(2, M2_INIT_POS); time.sleep(1)

        self.log("\n▶ [1단계] 모터 최대 가동범위 세팅")
        space_event.clear(); space_event.wait()
        self.log("\n[안내] 환자가 손바닥을 최대한 펴게 해주세요.\n▶ 2번 모터 당김 시작 (다 펴지면 클릭/스페이스)")
        space_event.clear(); space_event.wait()
        curr_m2 = M2_INIT_POS
        space_event.clear()
        self.sig_state.emit("EXTEND", 50)
        while is_running and not space_event.is_set() and curr_m2 >= 0:
            send_motor(2, curr_m2, 20); curr_m2 -= 1; time.sleep(0.05)
        m2_limit = curr_m2
        send_motor(2, M2_INIT_POS)
        
        self.log("\n[안내] 환자가 주먹을 쥐게 해주세요.\n▶ 1번 모터 당김 시작 (다 쥐면 클릭/스페이스)")
        space_event.clear(); space_event.wait()
        curr_m1 = M1_INIT_POS
        space_event.clear()
        self.sig_state.emit("BEND", 50)
        while is_running and not space_event.is_set() and curr_m1 <= 180:
            send_motor(1, curr_m1, 20); curr_m1 += 1; time.sleep(0.05)
        m1_limit = curr_m1
        send_motor(1, M1_INIT_POS)

        self.log("\n▶ [2단계] 센서 기준값 수집 (클릭/스페이스 진행)")
        space_event.clear(); space_event.wait()
        self.log("▶ 손가락 펴기, 센서 측정 중 (다 펴지면 클릭)")
        space_event.clear(); space_event.wait()
        curr_m2 = M2_INIT_POS
        space_event.clear()
        while is_running and not space_event.is_set() and curr_m2 >= m2_limit:
            send_motor(2, curr_m2, 20); curr_m2 -= 1; time.sleep(0.05)
        flex_max_open = arduino_data['flex']
        send_motor(2, M2_INIT_POS)
        
        self.log("▶ 주먹 쥐기, 센서 측정 중 (다 쥐면 클릭)")
        space_event.clear(); space_event.wait()
        curr_m1 = M1_INIT_POS
        space_event.clear()
        while is_running and not space_event.is_set() and curr_m1 <= m1_limit:
            send_motor(1, curr_m1, 20); curr_m1 += 1; time.sleep(0.05)
        flex_max_close = arduino_data['flex']
        send_motor(1, M1_INIT_POS)

        self.log("\n▶ [3단계] 근력 측정 시작 (클릭/스페이스 진행)")
        space_event.clear(); space_event.wait()
        self.log("📢 5초간 근력 측정 모듈을 있는 힘껏 쥐어주세요!")
        fsr_sum = 0
        for i in range(5, 0, -1):
            self.log(f" ⏱️ {i}초... (현재 근력 인식값: {arduino_data['fsr']})")
            fsr_sum += arduino_data['fsr']
            time.sleep(1)
        avg_fsr = fsr_sum / 5.0
        initial_k = max(15, min(80, int((avg_fsr / 1023.0) * 45) + 15)) 
        self.log("\n  캘리브레이션 완료! 재활 준비가 끝났습니다.")
        self.sig_state.emit("IDLE", 0)

    def execute_rl_phase_gui(self, motor_num, init_angle, limit_angle, target_flex, is_bend):
        global pain_event, bayesian_safety_score
        curr_angle = init_angle
        step = 1 if limit_angle > init_angle else -1
        phase_history = []
        flex_range = max(1, abs(flex_max_close - flex_max_open))
        
        while is_running:
            curr_flex = arduino_data['flex']
            
            percent = int((abs(curr_flex - flex_max_open) / flex_range) * 100)
            percent = max(0, min(100, percent))
            state_text = "BEND" if is_bend else "EXTEND"
            self.sig_state.emit(state_text, percent)

            self.sig_stats.emit(cycle_count, patient_pain_log['pain1'], patient_pain_log['pain2'])
            
            if is_bend:
                if (flex_max_close > flex_max_open and curr_flex >= target_flex) or (flex_max_close < flex_max_open and curr_flex <= target_flex): break
            else:
                if (flex_max_open > flex_max_close and curr_flex >= target_flex) or (flex_max_open < flex_max_close and curr_flex <= target_flex): break

            if (step > 0 and curr_angle >= limit_angle) or (step < 0 and curr_angle <= limit_angle): break

            if pain_event == 3: return False, phase_history
            elif pain_event == 2: 
                self.log("\n[🚨 통증 감지] 즉시 안전 위치로 복귀 및 범위 축소!")
                pain_scatter_points.append((percent, 2))
                update_bayesian_safety(2)
                if phase_history: linucb_agent.update(phase_history[-1][0], phase_history[-1][1], reward=-100)
                pain_event = 0
                return False, phase_history
            elif pain_event == 1:
                self.log("\n[⚠️ 뻐근함 감지] 해당 구간의 속도를 낮춥니다.")
                pain_scatter_points.append((percent, 1))
                update_bayesian_safety(1)
                if phase_history: linucb_agent.update(phase_history[-1][0], phase_history[-1][1], reward=-10)
                pain_event = 0

            norm_flex = abs(curr_flex - flex_max_open) / flex_range
            norm_grip = avg_fsr / 1023.0
            context = [norm_flex, norm_grip, bayesian_safety_score, 1.0]
            
            action_idx = linucb_agent.select_action(context)
            ai_speed = SPEED_ARMS[action_idx]
            phase_history.append((action_idx, context))
            
            rehab_data_log.append({
                'Cycle': cycle_count, 'Flex_Value': curr_flex, 
                'Motor_Speed': ai_speed, 'Safety_Score': round(bayesian_safety_score, 3)
            })

            send_motor(motor_num, curr_angle, ai_speed)
            curr_angle += step
            time.sleep(0.05) 

        return True, phase_history

    def rl_rehab_loop(self):
        global current_mode, is_running, cycle_count
        global dyn_flex_open, dyn_flex_close, linucb_agent, SPEED_ARMS, rom_growth_log
        
        self.log("\n▶ [진행 (Space)] 버튼을 누르면 본격적인 재활이 시작됩니다.")
        current_mode = "WAIT_RL"
        space_event.clear(); space_event.wait()
        if not is_running: return
        
        current_mode = "RL"
        self.sig_phase.emit("RL") # UI 버튼 긴급중지로 변경
        
        dyn_flex_open = flex_max_open
        dyn_flex_close = flex_max_close
        base_rom = abs(flex_max_close - flex_max_open)
        SPEED_ARMS = [initial_k, min(100, initial_k + 10), min(100, initial_k + 20), min(100, initial_k + 30)]
        linucb_agent = LinUCBAgent(n_arms=len(SPEED_ARMS), n_features=4)

        while is_running:
            cycle_count += 1
            curr_rom = abs(dyn_flex_close - dyn_flex_open)
            rom_percent = (curr_rom / max(1, base_rom)) * 100
            rom_growth_log.append({'Cycle': cycle_count, 'ROM_Percent': rom_percent})
            
            self.sig_stats.emit(cycle_count, patient_pain_log['pain1'], patient_pain_log['pain2'])
            self.log(f"\n--- [Cycle {cycle_count}] 재활 진행 중 ---")
            
            success_bend, hist_bend = self.execute_rl_phase_gui(1, M1_INIT_POS, m1_limit, dyn_flex_close, is_bend=True)
            if not is_running: break
            if not success_bend: 
                self.log("🔄 안전 롤백 실행 중..."); send_motor(1, M1_INIT_POS, 60); send_motor(2, M2_INIT_POS, 60); dyn_flex_open = flex_max_open; dyn_flex_close = flex_max_close; time.sleep(2)
                continue

            send_motor(1, M1_INIT_POS, 60); time.sleep(1)
            if not is_running: break

            success_ext, hist_ext = self.execute_rl_phase_gui(2, M2_INIT_POS, m2_limit, dyn_flex_open, is_bend=False)
            if not is_running: break
            if not success_ext: 
                self.log("🔄 안전 롤백 실행 중..."); send_motor(1, M1_INIT_POS, 60); send_motor(2, M2_INIT_POS, 60); dyn_flex_open = flex_max_open; dyn_flex_close = flex_max_close; time.sleep(2)
                continue

            send_motor(2, M2_INIT_POS, 60)
            self.sig_state.emit("IDLE", 0)
            
            self.log("✅ 사이클 완료! (목표 가동범위 2% 확장)")
            update_bayesian_safety(0)
            for act, ctx in hist_bend + hist_ext: linucb_agent.update(act, ctx, reward=+10)
                
            gap = abs(flex_max_close - flex_max_open)
            expand_val = gap * 0.02 
            if flex_max_close > flex_max_open:
                dyn_flex_close += expand_val; dyn_flex_open -= expand_val
            else:
                dyn_flex_close -= expand_val; dyn_flex_open += expand_val

            for _ in range(30):
                if not is_running: break
                time.sleep(0.1)


# =====================================================================
# [4] GUI 메인 윈도우 클래스 (의료기기 디자인 전면 개편)
# =====================================================================
class RehabApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Clinical Rehab System - 상지 재활 로봇 제어 센터")
        self.resize(1300, 750)
        self.setFont(QFont("Malgun Gothic", 10))
        
        # 전체 윈도우 배경색 (깔끔한 연회색)
        self.setStyleSheet("QMainWindow { background-color: #f4f6f9; }")
        
        self.initUI()
        
        self.worker = RehabWorker()
        self.worker.sig_log.connect(self.append_log)
        self.worker.sig_state.connect(self.update_visuals)
        self.worker.sig_stats.connect(self.update_stats)
        self.worker.sig_phase.connect(self.update_phase_ui)
        self.worker.sig_finish.connect(self.show_patient_dashboard)
        self.worker.start()

    def initUI(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        h_layout = QHBoxLayout(main_widget)
        h_layout.setContentsMargins(20, 20, 20, 20)
        h_layout.setSpacing(20)

        # ---------------- 좌측 영역 (AI 콘솔 + 피드백 제어부) ----------------
        v_left = QVBoxLayout()
        
        # 콘솔 그룹박스 (전문적인 테두리 적용)
        console_group = QGroupBox("AI 실시간 제어 로그")
        console_group.setStyleSheet("""
            QGroupBox { font-size: 14pt; font-weight: bold; color: #2c3e50; border: 2px solid #bdc3c7; border-radius: 8px; margin-top: 1ex; padding: 10px; background-color: white;}
            QGroupBox::title { subcontrol-origin: margin; left: 15px; padding: 0 5px 0 5px; }
        """)
        v_console = QVBoxLayout(console_group)
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        # 콘솔 내부는 어두운 테마로 가독성 및 전문성 확보
        self.console.setStyleSheet("background-color: #1e272e; color: #00d8d6; font-family: 'Malgun Gothic', Consolas; font-size: 13pt; border: none;")
        v_console.addWidget(self.console)
        v_left.addWidget(console_group, stretch=7)

        # 컨트롤 버튼 그룹
        btn_layout = QHBoxLayout()
        btn_style_base = "font-family: 'Malgun Gothic'; font-size: 22px; font-weight: bold; border-radius: 10px; min-height: 80px;"
        
        self.btn_space = QPushButton("▶ 진행 (Space)")
        # 초기 세팅 모드일 때는 파란색
        self.btn_space.setStyleSheet(f"background-color: #0984e3; color: white; {btn_style_base}")
        self.btn_space.clicked.connect(self.simulate_space)
        
        self.btn_pain1 = QPushButton("⚠️ 뻐근함 (1)")
        self.btn_pain1.setStyleSheet(f"background-color: #fdcb6e; color: #2d3436; {btn_style_base}")
        self.btn_pain1.clicked.connect(lambda: self.simulate_pain(1))
        
        self.btn_pain2 = QPushButton("🚨 고통 (2)")
        self.btn_pain2.setStyleSheet(f"background-color: #d63031; color: white; {btn_style_base}")
        self.btn_pain2.clicked.connect(lambda: self.simulate_pain(2))

        btn_layout.addWidget(self.btn_space, stretch=2)
        btn_layout.addWidget(self.btn_pain1, stretch=1)
        btn_layout.addWidget(self.btn_pain2, stretch=1)
        v_left.addLayout(btn_layout, stretch=3)

        # ---------------- 우측 영역 (시각화 + 현황판) ----------------
        v_right = QVBoxLayout()
        
        # 실시간 통계 현황판 (의료 모니터 스타일)
        h_stats = QHBoxLayout()
        stat_style = "font-family: 'Malgun Gothic'; font-size: 18px; font-weight: bold; color: #2c3e50; background-color: white; border: 1px solid #dcdde1; border-radius: 8px; padding: 15px;"
        
        self.lbl_cycle = QLabel("🔄 현재 사이클: 0")
        self.lbl_cycle.setAlignment(Qt.AlignCenter)
        self.lbl_cycle.setStyleSheet(stat_style)
        
        self.lbl_pain1 = QLabel("⚠️ 뻐근함 누적: 0")
        self.lbl_pain1.setAlignment(Qt.AlignCenter)
        self.lbl_pain1.setStyleSheet(stat_style)
        
        self.lbl_pain2 = QLabel("🚨 고통 누적: 0")
        self.lbl_pain2.setAlignment(Qt.AlignCenter)
        self.lbl_pain2.setStyleSheet(stat_style)

        h_stats.addWidget(self.lbl_cycle)
        h_stats.addWidget(self.lbl_pain1)
        h_stats.addWidget(self.lbl_pain2)
        v_right.addLayout(h_stats)

        # 모션 시각화 그룹박스
        visual_group = QGroupBox("실시간 신체 반응 및 목표 ROM 달성률")
        visual_group.setStyleSheet("""
            QGroupBox { font-size: 14pt; font-weight: bold; color: #2c3e50; border: 2px solid #bdc3c7; border-radius: 8px; margin-top: 1ex; background-color: white;}
            QGroupBox::title { subcontrol-origin: margin; left: 15px; padding: 0 5px 0 5px; }
        """)
        v_visual = QVBoxLayout(visual_group)
        v_visual.setContentsMargins(20, 30, 20, 20)
        v_visual.setSpacing(20)

        # 이미지 (작게)
        h_img = QHBoxLayout()
        self.image_label = QLabel("대기 중")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(140, 140) 
        self.image_label.setStyleSheet("border: 2px solid #ecf0f1; background-color: #f8f9fa; font-family: 'Malgun Gothic'; font-size: 16px; border-radius: 70px;") # 원형 디자인 느낌
        h_img.addStretch()
        h_img.addWidget(self.image_label)
        h_img.addStretch()
        v_visual.addLayout(h_img)
        
        # 메인 진행률 바 (강조)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setMinimumHeight(140) 
        self.progress_bar.setStyleSheet("""
            QProgressBar { 
                border: 3px solid #b2bec3; 
                border-radius: 20px; 
                text-align: center; 
                font-family: 'Malgun Gothic';
                font-size: 55px; 
                font-weight: bold; 
                color: #2d3436;
                background-color: #dfe6e9;
            }
            QProgressBar::chunk { 
                background-color: #00cec9; 
                border-radius: 16px; 
            }
        """)
        v_visual.addWidget(self.progress_bar)
        
        v_right.addWidget(visual_group)
        
        h_layout.addLayout(v_left, 5) 
        h_layout.addLayout(v_right, 5)

    def append_log(self, text):
        self.console.append(text)
        self.console.verticalScrollBar().setValue(self.console.verticalScrollBar().maximum())

    def update_stats(self, cycle, p1_cnt, p2_cnt):
        self.lbl_cycle.setText(f"🔄 현재 사이클: {cycle}")
        self.lbl_pain1.setText(f"⚠️ 뻐근함 누적: {p1_cnt}")
        self.lbl_pain2.setText(f"🚨 고통 누적: {p2_cnt}")

    def update_phase_ui(self, phase):
        """치료 모드 진입 시 버튼 디자인 변경 (긴급 중지)"""
        if phase == "RL":
            self.btn_space.setText("🛑 긴급 중지 (Space)")
            self.btn_space.setStyleSheet("font-family: 'Malgun Gothic'; font-size: 24px; font-weight: bold; border-radius: 10px; min-height: 80px; background-color: #d63031; color: white; border: 3px solid #ff7675;")

    def simulate_space(self): space_event.set()
    def simulate_pain(self, level): 
        global pain_event
        pain_event = level

    def update_visuals(self, state, percent):
        self.progress_bar.setValue(percent)
        
        if state == "BEND":
            pixmap = QPixmap("fist.png")
            if not pixmap.isNull(): self.image_label.setPixmap(pixmap.scaled(140, 140, Qt.KeepAspectRatio))
            else: self.image_label.setText("✊\n주먹 쥐기")
            self.progress_bar.setStyleSheet("QProgressBar { border: 3px solid #b2bec3; border-radius: 20px; text-align: center; font-family: 'Malgun Gothic'; font-size: 55px; font-weight: bold; color: #2d3436; background-color: #dfe6e9; } QProgressBar::chunk { background-color: #fdcb6e; border-radius: 16px; }")
            
        elif state == "EXTEND":
            pixmap = QPixmap("open.png")
            if not pixmap.isNull(): self.image_label.setPixmap(pixmap.scaled(140, 140, Qt.KeepAspectRatio))
            else: self.image_label.setText("🖐\n손 펴기")
            self.progress_bar.setStyleSheet("QProgressBar { border: 3px solid #b2bec3; border-radius: 20px; text-align: center; font-family: 'Malgun Gothic'; font-size: 55px; font-weight: bold; color: #2d3436; background-color: #dfe6e9; } QProgressBar::chunk { background-color: #0984e3; border-radius: 16px; }")
            
        else:
            self.image_label.setText("대기 중")
            self.progress_bar.setValue(0)

    def show_patient_dashboard(self):
        if not rom_growth_log:
            print("저장된 재활 데이터가 없습니다.")
            return 
            
        # 엑셀 저장 여부 팝업 (Yes/No)
        reply = QMessageBox.question(self, '치료 결과 저장', 
                                     '재활 세션이 종료되었습니다.\n치료 결과 및 센서 세부 내역을 엑셀 파일로 저장하시겠습니까?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        
        if reply == QMessageBox.Yes:
            df = pd.DataFrame(rehab_data_log)
            df.to_excel("Rehab_AI_Report.xlsx", index=False)
            self.append_log("\n[알림] 데이터가 'Rehab_AI_Report.xlsx' 에 성공적으로 저장되었습니다.")
            QMessageBox.information(self, "저장 완료", "Rehab_AI_Report.xlsx 파일로 저장이 완료되었습니다.")
        
        # 환자 차트 그리기
        plt.rc('font', family='Malgun Gothic') 
        fig = plt.figure(figsize=(16, 5.5))
        fig.canvas.manager.set_window_title('환자 맞춤형 재활 성과 리포트')
        fig.patch.set_facecolor('#f4f6f9')
        gs = GridSpec(1, 3, figure=fig)
        
        ax1 = fig.add_subplot(gs[0, 0])
        cycles = [log['Cycle'] for log in rom_growth_log]
        percents = [log['ROM_Percent'] for log in rom_growth_log]
        ax1.fill_between(cycles, percents, color="#74b9ff", alpha=0.5)
        ax1.plot(cycles, percents, color="#0984e3", alpha=0.8, linewidth=2.5)
        ax1.set_title("1. 가동범위 성장 곡선", fontsize=12, fontweight='bold')
        ax1.set_xlabel("재활 사이클")
        ax1.set_ylabel("범위 성장 달성률 (%)")
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        ax2 = fig.add_subplot(gs[0, 1])
        labels = ['안전 구동', '뻐근함(1)', '강한 통증(2)']
        sizes = [patient_pain_log['safe'], patient_pain_log['pain1'], patient_pain_log['pain2']]
        colors = ['#00b894', '#fdcb6e', '#d63031']
        sizes_filt = [s for s in sizes if s > 0]
        labels_filt = [l for s, l in zip(sizes, labels) if s > 0]
        colors_filt = [c for s, c in zip(sizes, colors) if s > 0]
        
        if sizes_filt:
            wedges, texts, autotexts = ax2.pie(sizes_filt, labels=labels_filt, colors=colors_filt, autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=0.4, edgecolor='w'))
            plt.setp(texts, fontsize=10, fontweight='bold')
            plt.setp(autotexts, fontsize=10, color='black', fontweight='bold')
            ax2.set_title("2. 재활치료 기록 현황", fontsize=12, fontweight='bold')
        
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot([0, 100], [0, 0], color="#ffffff", linewidth=12, zorder=1)
        for pt in pain_scatter_points:
            percent, p_type = pt
            color = '#fdcb6e' if p_type == 1 else '#d63031'
            size = 120 if p_type == 1 else 250
            ax3.scatter(percent, 0, color=color, s=size, zorder=2, edgecolors='black', alpha=0.8)
            
        ax3.set_yticks([])
        ax3.set_xlim(-5, 105)
        ax3.set_title("3. 통증 발생 지점 맵", fontsize=12, fontweight='bold')
        ax3.set_xlabel("손가락 위치 (0%: 다 펴짐 ↔ 100%: 꽉 쥠)")

        plt.tight_layout(pad=3.0)
        plt.show(block=False)

if __name__ == "__main__":
    arduino_thread = threading.Thread(target=read_arduino_data, daemon=True)
    arduino_thread.start()
    
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    
    app = QApplication(sys.argv)
    app.setFont(QFont("Malgun Gothic", 10))
    
    ex = RehabApp()
    ex.show()
    sys.exit(app.exec_())