# H.O.P.E Hand — Wire-Driven Soft Rehabilitation Glove for Finger Paresis

**H.O.P.E Hand** (팀 희망회로, 2026 소원 H.O.P.E Project, Kwangwoon University) is a low-cost,
wire-driven soft wearable robot for hand/finger rehabilitation after stroke or other
neurological injury, paired with a **contextual-bandit (LinUCB) reinforcement-learning
controller** that adapts motor speed to each patient's pain feedback in real time.

> This repository packages the project's control/GUI software for a public portfolio. The
> hardware (glove, scissor-linkage joints, bellows, wiring) and the Arduino-side firmware are
> not included; this repo covers the PC-side Python application that runs the rehabilitation
> logic, drives the motors over serial, and renders the clinician-facing GUI.

## Problem

- Stroke causes hemiparesis in the majority of survivors, and finger/hand paresis is
  particularly disruptive to daily activities, yet existing rehabilitation robots are either
  too expensive (~KRW 77,000,000 for EMG-sensor-based devices) or too crude (pneumatic
  SYREBO-style gloves with no force feedback) for wide clinical or home deployment.
- Existing devices also cannot modulate assistance to each patient's pain tolerance in real
  time, risking tissue/muscle damage when a stiff joint is forced beyond a safe range.

## Approach

**Hardware concept.** A fabric glove carries a Velcro-mounted, silicone wire guide per finger
that routes the tendon wire along the dorsal side, driven by a servo motor per finger
(tendon-driven, under the glove — the project's early proposal called for a plastic
scissor-mechanism joint and a rubber bellows joint, but the built/demonstrated prototype
replaced both with this simpler Velcro + silicone-wire-guide mount). A flex sensor on the back
of the finger measures joint angle; an FSR (force) sensor in the palm measures grip force. An
Arduino Nano reads the sensors and drives the servo(s); the PC side (this repo) hosts the
control logic and the clinician UI.

**Control software — three layers:**

1. **Calibration.** On first wear, the device sweeps each finger through its full passive
   range, logging the mechanical min/max servo angle (`M1_INIT_POS`/`m1_limit`,
   `M2_INIT_POS`/`m2_limit`) and the corresponding flex-sensor readings
   (`flex_max_open`/`flex_max_close`). A 5-second grip-strength probe sets an initial motor
   speed `initial_k` scaled to the patient's measured FSR force.
2. **LinUCB-based adaptive assistance.** Rather than a deep-RL agent (which needs thousands
   of episodes a rehab session cannot provide), a **Linear Upper Confidence Bound (LinUCB)
   contextual bandit** picks a motor speed from a small discrete arm set
   (`SPEED_ARMS = [k, k+10, k+20, k+30]`) at each control step, using a 4-D context
   `[normalized flex position, normalized grip strength, Bayesian safety score, bias]`. The
   agent updates online from two feedback channels:
   - **Keyboard/button pain signal** — `0` = no pain (reward +10 on a completed cycle,
     safety score multiplicatively recovers), `1` = ache (reward −10, safety score ×0.8),
     `2` = sharp pain (reward −100, immediate rollback to the home position, safety score
     ×0.4), `3`/space during the active phase = emergency stop.
   - **Dynamic ROM expansion** — every successfully completed bend/extend cycle with no pain
     event expands the next cycle's target ROM by 2% (a physical-therapy-style progressive
     overload step); a pain event instead resets the target ROM back to the calibrated safe
     baseline.
3. **GUI and session reporting.** A PyQt5 application (`RehabApp`) runs the control loop in a
   background `QThread` (`RehabWorker`) so the UI stays responsive, shows live flex-progress
   and cycle/pain counters, and — on session end — asks whether to export the full
   cycle-by-cycle log to Excel and renders a 3-panel Matplotlib summary (ROM growth curve,
   pain-event pie chart, pain-location scatter along the finger's flex range).

## Hardware (as built)

The final, demonstrated prototype is a **two-finger** (index + middle) tendon-driven unit with
a **detachable** strap/tendon assembly — it can be re-mounted onto whichever finger needs
treatment rather than building out all five fingers at once. The build consists of a fabric
glove, a 3D-printed servo mounting bracket with a Velcro + silicone wire-guide tendon mount
(no scissor-linkage or bellows joint — those were in the project's early proposal but dropped
in the built prototype in favor of this simpler mount), a battery/driver circuit box, and a
separate finger-grip-strength measurement switch used during calibration. No lab/school budget
was used; parts were self-funded and the mechanical components were 3D-printed.

## Open-source components

| Library | Purpose in this project | License |
|---|---|---|
| NumPy | LinUCB linear algebra (matrix inverse, dot products) | BSD-3-Clause |
| pandas | Structuring and exporting per-cycle rehabilitation logs | BSD-3-Clause |
| Matplotlib | 3-panel session report (ROM curve, pain pie chart, pain-location scatter) | PSF / Matplotlib |
| pySerial | Arduino ↔ PC serial communication | PSF |
| pynput | OS-level keyboard hook for real-time pain/emergency-stop input, independent of GUI focus | LGPLv3 |
| openpyxl | Excel export of the session log | MIT |

Planned open-source contributions: publishing the hardware-independent LinUCB control package
separately for reuse, reporting/patching issues found in the libraries above, and — pending
IRB-style clinical testing — releasing an anonymized rehabilitation dataset and simulator.

## Repository structure

```
hope-rehab-robot/
├── src/
│   └── rehab_app.py      # Full control + GUI application (serial I/O, LinUCB agent,
│                          # setup/calibration state machine, PyQt5 GUI, session report)
├── requirements.txt
└── README.md / README.ko.md
```

## Running it

```bash
pip install -r requirements.txt
python src/rehab_app.py
```

The script expects an Arduino on a serial port (`PORT = 'COM3'`, `BAUD = 115200` — edit these
constants for your setup) that speaks a simple `DATA,<flex>,<fsr>\n` protocol and accepts
`M<motor_num>,<angle>,<speed>\n` commands. Without a connected Arduino the app still launches
in a degraded "no device" mode (motor commands are silently dropped, sensor values stay at
their defaults), which is useful for reviewing the control flow and GUI without hardware.

## Results

- A working two-finger (index + middle), detachable, tendon-driven prototype was built and demonstrated, with the
  flex/FSR sensing loop, Arduino Nano control, and LinUCB-driven assistance integrated
  end-to-end.
- The report's "Model Convergence: Uncertainty Decay" plot shows the LinUCB agent's
  action-value uncertainty dropping toward zero within the first rehabilitation cycles,
  i.e. the bandit converges on a stable speed policy with far fewer samples than a deep-RL
  approach would need — consistent with the motivation for choosing LinUCB over DQN/PPO for a
  setting where each session only yields a few dozen control steps.
- The "Patient Compliance vs. ROM Expansion" plot shows the Bayesian safety score and the
  dynamic ROM target tracking each other as designed: ROM expands steadily while the safety
  score stays high (no pain reported), and both collapse together back to the calibrated
  baseline the moment a pain event (button 2) is triggered — i.e., the progressive-overload
  and safety-rollback logic behaved as intended in the recorded session.

## Future improvement plan

1. **Hardware** — improve wire-tension transfer for more stable motion, add more rehabilitation
   modes, and miniaturize the device toward a practical wearable form.
2. **AI algorithm** — move beyond the current linear bandit toward a model that can capture
   more complex, irregular pain responses, and let the agent learn the ROM-expansion ratio
   itself instead of using a fixed 2% step.
3. **Sensor/hardware reliability** — combine the single flex sensor with an additional angle
   sensor to reduce measurement error, and add EMG-based muscle-signal sensing so unintentional
   muscle tension can be detected even when the patient cannot press the pain button.
4. **Clinical data-driven treatment** — move from single-session logs to cumulative,
   multi-session treatment history so the system can find the right treatment intensity faster
   in later sessions, and validate/tune the model against real clinical data.
5. **Software structure and safety** — refactor the current script into modules/classes for
   maintainability, and strengthen the automatic safety-stop behavior for serial communication
   failures or unexpected errors.

## Team

**희망회로 (Team HOPE Circuit)** — 2026 소원 H.O.P.E Project, Kwangwoon University.

| Member | Role |
|---|---|
| Byeongkyu Ahn (안병규) — Robotics, Kwangwoon Univ. | Arduino programming, sensor control, rehabilitation-mode implementation, system control-algorithm (LinUCB) development, report writing |
| Changbin Park (박창빈) — Robotics, Kwangwoon Univ. | 3D modeling, rehab-device structural design, servo/wire actuation assembly, sensor & circuit integration, video production |
| Seongjae Koo (구성재) — Electronic Eng., Kookmin Univ. | Hardware/software integration, poster & presentation materials, report writing, education materials |

Advisor: Prof. Yesob Lee, Dept. of Robotics, Kwangwoon University.

## References

- Korea Disease Control and Prevention Agency (KDCA), *Cerebro-cardiovascular Disease
  Statistics*; KDCA National Health Information Portal.
- SYREBO hand rehabilitation robot product series.
- Hand of Hope (HOH) robotic hand rehabilitation system.
- J. Iqbal et al., "Exoskeletons for upper limb rehabilitation: A systematic review,"
  *Frontiers in Robotics and AI*, 2020.
- C. L. Lynch & M. R. Popovic, "Functional electrical stimulation," *IEEE Control Systems
  Magazine*, 2008.
