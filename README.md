# Mitsubishi Electric RV-7FM Predictive Maintenance & Progressive Degradation Analytics

## Project Overview

This project develops an operating-condition-aware predictive maintenance framework for a **Mitsubishi Electric RV-7FM-D1-S15 6-axis industrial robot** using high-frequency operational telemetry from the **Mitsubishi Electric Anomaly Detection (ME-AD) dataset**.

The dataset represents cyclic industrial pick-and-place operations performed while a small defect was introduced into the **Joint 3 actuator**, allowing lubricant to escape progressively and producing an evolving degradation process.

Rather than treating every unusual signal as evidence of failure, the project distinguishes genuine degradation from normal changes caused by robot speed, trajectory and operating condition.

The analysis progresses from healthy behavioural baselining through degradation detection, multivariate anomaly detection and persistent early-warning logic to maintenance decision support.

---

## Business Question

> **How can high-frequency operational telemetry from a Mitsubishi Electric RV-7FM-D1-S15 6-axis industrial robot be used to detect and track progressive Joint 3 actuator degradation caused by lubricant loss, distinguish genuine mechanical deterioration from normal variations in robot speed and pick-and-place trajectory, and provide actionable early warning for predictive maintenance before degradation reaches an evidently faulty state?**

---

## Analytical Objective

The project builds an operating-condition-aware framework capable of:

- Establishing normal robot behaviour under healthy operation.
- Quantifying the effect of speed and trajectory on robot dynamics.
- Detecting progressive changes associated with Joint 3 degradation.
- Separating local Joint 3 behaviour from coupled multi-joint response.
- Constructing an interpretable degradation-based health indicator.
- Independently detecting statistically unusual behaviour using Isolation Forest.
- Requiring persistent evidence before issuing maintenance warnings.
- Identifying robot trajectories that act as effective diagnostic probes.
- Quantifying the operating-cycle opportunity available for maintenance planning.

The analytical progression is:

**Healthy Behaviour → Operating-Condition Effects → Degradation Signatures → Health Indicator → Anomaly Detection → Persistent Warning → Maintenance Decision Support**

---

## Physical System

The machine under investigation is a:

**Mitsubishi Electric RV-7FM-D1-S15**

- 6-axis industrial robot
- Cyclic pick-and-place operation
- Multiple trajectory and speed configurations
- Progressive Joint 3 actuator degradation
- High-frequency multivariate telemetry

The degradation mechanism is associated with progressive lubricant loss from the Joint 3 actuator.

An important analytical challenge is that robot telemetry naturally changes when the robot performs different movements. Higher velocity, acceleration or torque therefore cannot automatically be interpreted as degradation.

The analysis must first understand what is normal for each operating condition.

---

## Dataset

The project uses the **Mitsubishi Electric Anomaly Detection (ME-AD)** dataset.

### Dataset Scale

- **28,824 robot cycles**
- **16 operating conditions**
- High-frequency telemetry sampled at approximately **281 Hz**
- Joint position, velocity, acceleration and effort measurements
- Six robot joints
- Healthy, intermediate and evidently faulty operating regions

The source dataset contains both raw CSV telemetry and cleaned/filtered PKL representations.

Because the complete dataset is very large, raw and intermediate data are intentionally excluded from this Git repository.

---

## Operating Conditions

The experiment contains three major trajectory families.

### Family 1

Two complete transfers between point A and two fixed B locations, executed at different commanded speeds and layouts.

Operation codes:

`1050, 1051, 1052, 1053, 1100, 1101, 1102, 1103`

### Family 2

Single pick-and-place movements between point A and a randomly positioned B location with six trajectory configurations.

Operation codes:

`2000, 2100, 2200, 2300, 2400, 2500`

### Family 3

Smooth blended trajectory variants.

Operation codes:

`3000, 3100`

This operating-condition structure is central to the analysis because degradation observability varies substantially between trajectories.

---

## Telemetry

The telemetry contains kinematic and effort-related signals for all six joints.

Core physical variables include:

- `q` — joint position
- `dq` — joint velocity
- `ddq` — joint acceleration
- `tau` — joint effort / torque-related signal

The analysis focuses particularly on effort-related behaviour because progressive mechanical degradation may increase or alter the effort required to preserve commanded robot motion.

However, the project does **not** assume that high torque automatically indicates degradation.

Torque must be interpreted relative to the robot's movement and operating condition.

---

# Analytical Workflow

## Phase 1 — Data Inventory & Structural Validation

The first phase established the structure, scale and integrity of the ME-AD dataset.

The complete analytical population contained:

**28,824 robot cycles across 16 operating conditions.**

The experiment was separated into benchmark regions representing:

- Training healthy
- Test healthy
- Intermediate / unlabelled progression
- Evidently faulty

This preserved the large intermediate region instead of artificially assigning a precise degradation onset where none was provided by the dataset.

---

## Phase 2 — Signal Quality & Physical Interpretation

Representative cycles were inspected across operating conditions and degradation regions.

The analysis examined:

- Signal completeness
- Cycle length
- Joint movement
- Effort behaviour
- Abnormally long or short cycles
- Near-stationary or structurally unusual sequences

Automated structural quality controls were later applied to the complete dataset.

Of the 28,824 cycles:

**28,806 were retained as structurally usable cycles.**

The small number of structurally abnormal cycles were excluded from downstream modelling where appropriate.

---

## Phase 3 — Healthy Behaviour Baseline

Healthy operation was analysed first to establish how robot dynamics change under normal operating conditions.

This demonstrated that robot behaviour varies materially with:

- Commanded speed
- Trajectory
- Layout
- Joint
- Dynamic loading

Joint 3 was found to carry substantial effort during healthy operation, reinforcing the need for condition-aware comparison rather than global thresholds.

A key engineering conclusion emerged:

> **A robot cycle should be compared with the healthy behaviour of the same operating condition, not with an undifferentiated global baseline.**

---

## Phase 4 — Progressive Degradation Analysis

All usable cycles were analysed sequentially to examine how robot behaviour changed as the experiment progressed toward the evidently faulty region.

Joint 3 displayed substantial late-stage degradation signatures, particularly in effort-related features.

However, degradation was not expressed exclusively through Joint 3.

Other joints also exhibited coupled changes as the controller continued coordinating the six-axis manipulator.

This led to an important design decision:

> **The predictive maintenance framework should retain Joint 3 as the target joint while incorporating system-wide robot response.**

---

## Phase 5 — Health Indicator

A multijoint degradation framework was developed using robust deviations from healthy operation.

Four effort features were evaluated for every joint:

- Mean absolute torque
- Torque RMS
- Peak torque
- Torque standard deviation

A robust joint-level deviation score was calculated relative to healthy reference behaviour.

The framework then combined:

- Joint 3 degradation evidence
- Control-joint behaviour
- System-wide degradation behaviour

into an interpretable degradation index and corresponding health index.

The health index is an analytical measure of departure from healthy behaviour.

It is **not**:

- Failure probability
- Percentage remaining life
- Remaining Useful Life (RUL)

---

## Persistent Health Warning

A single abnormal cycle was deliberately not treated as a maintenance event.

Persistent warning logic required degradation evidence to remain present across rolling cycle windows.

This reduced sensitivity to isolated transient behaviour.

Persistent health warnings were identified in **6 of the 16 operating conditions**.

The strongest trajectories produced substantial warning lead before the dataset-defined evidently faulty region.

---

## Phase 6 — Isolation Forest Anomaly Detection

Isolation Forest was introduced as an **independent statistical evidence stream**.

One Isolation Forest model was trained for each operating condition.

This was important because different trajectories naturally produce different telemetry distributions.

### Model Design

Each model used **24 effort features**:

**6 joints × 4 effort metrics**

The Isolation Forest models learned statistically normal healthy behaviour and identified cycles that became increasingly unusual relative to that baseline.

The models used:

- Robust feature scaling
- 500 trees per operating condition
- Healthy reference data for model fitting
- Operation-specific anomaly thresholds

Isolation Forest does not know:

- Which joint is physically defective
- That lubricant is escaping
- What constitutes mechanical failure
- The meaning of the robot's task

It simply identifies statistically unusual multivariate behaviour.

This makes it analytically different from the engineered health indicator.

---

## Health Indicator vs Isolation Forest

The two evidence streams answer different questions.

**Health Indicator**

> How far has physical robot behaviour departed from its healthy reference?

**Isolation Forest**

> How statistically unusual is this cycle compared with learned healthy operation?

Their convergence provides stronger evidence than either method alone.

As health deteriorated, Isolation Forest anomaly prevalence increased substantially.

---

## Phase 7 — Multi-Signal Maintenance Alert Framework

The health indicator and Isolation Forest were combined into a persistent multi-signal maintenance framework.

Four states were defined:

| State | Interpretation |
|---|---|
| **NORMAL** | No persistent degradation evidence |
| **WATCH** | Persistent statistical anomaly evidence |
| **WARNING** | Persistent health degradation evidence |
| **CRITICAL** | Persistent convergence of health and anomaly evidence |

This structure prevents a single unusual observation from immediately becoming a maintenance recommendation.

---

## Diagnostic Probe Ranking

A major finding was that degradation was **not equally observable under every robot trajectory**.

The 16 operating conditions were ranked according to their usefulness as diagnostic probes.

The ranking incorporated:

- Health-warning robustness
- Isolation Forest persistence
- Cross-method convergence
- Predictive warning lead

### Primary Diagnostic Probes

The five strongest diagnostic conditions were:

1. `1053`
2. `1050`
3. `1051`
4. `1052`
5. `1101`

Operation `1053` achieved the highest overall diagnostic utility.

An important engineering implication is that normal production movements can potentially serve a dual purpose:

**performing production work while simultaneously acting as recurring diagnostic probes of robot condition.**

---

## Phase 8 — Predictive Maintenance Opportunity

The final phase translated analytical warning evidence into maintenance-planning opportunity.

Two distinct horizons were calculated.

### Early Awareness Lead

The number of operating cycles between the first persistent degradation signal and the dataset-defined evidently faulty region.

### Confirmed Maintenance Opportunity

The number of operating cycles between the actionable maintenance point and the evidently faulty region.

These measures are intentionally expressed in **observed robot cycles**.

They are not interpreted as Remaining Useful Life or physical time-to-failure.

---

## Key Predictive Maintenance Results

Five operating conditions produced strong, convergent and actionable degradation evidence.

| Operation | Maintenance Opportunity (Cycles) |
|---|---:|
| 1053 | 1,054 |
| 1050 | 1,013 |
| 1051 | 1,011 |
| 1052 | 1,004 |
| 1101 | 630 |

### Portfolio-Level KPIs

- **5 Strong Actionable Conditions**
- **5 Primary Diagnostic Probes**
- **1,334 cycles Median Early Awareness Lead**
- **1,011 cycles Median Confirmed Maintenance Opportunity**
- **630 cycles Minimum Confirmed Opportunity**
- **1,054 cycles Maximum Confirmed Opportunity**

Operation `2500` provided additional health-based evidence supporting targeted inspection, while several other operating conditions were better suited to monitoring or were found to have limited degradation observability.

---

# Maintenance Decision Framework

The final analytical framework maps persistent evidence to practical maintenance responses.

| Analytical Evidence | Maintenance Response |
|---|---|
| No persistent warning | Continue Operation |
| Statistical anomaly only | Increase Monitoring |
| Health degradation only | Targeted Inspection |
| Convergent degradation evidence | Prioritize Maintenance |

Across the 16 operating conditions, the final response distribution was:

- **5 — Prioritize Maintenance**
- **1 — Targeted Inspection**
- **3 — Increase Monitoring**
- **7 — Continue Operation**

This provides a decision-support layer rather than simply producing anomaly scores.

---

# Tableau Decision-Support Dashboard

The final analytical results were consolidated into a Tableau dashboard:

## Mitsubishi RV-7FM Predictive Maintenance Decision Support

The dashboard communicates:

- Strong actionable operating conditions
- Median confirmed maintenance opportunity
- Median early-awareness lead
- Primary diagnostic probes
- Maintenance opportunity by operation
- Diagnostic probe ranking
- Maintenance response distribution
- Predictive maintenance windows

The dashboard is designed to connect engineering telemetry with operational maintenance decisions.

---

# Key Engineering Findings

### 1. Operating context matters

Robot speed and trajectory materially influence normal telemetry. Global thresholds can therefore confuse legitimate operating differences with degradation.

### 2. Degradation is multivariate

Although Joint 3 is the degradation target, the robot responds as a coupled six-axis system.

### 3. Motion can remain controlled while effort behaviour changes

The controller can preserve commanded motion even while internal effort patterns begin departing from healthy behaviour.

### 4. Statistical anomaly and physical degradation are complementary

Isolation Forest and the engineered health indicator provide different forms of evidence. Their persistent convergence strengthens maintenance confidence.

### 5. Diagnostic sensitivity depends on trajectory

Some production movements expose degradation substantially earlier and more consistently than others.

### 6. Predictive maintenance requires persistence

Individual anomalous cycles are insufficient evidence for maintenance intervention. Sustained behaviour across multiple cycles provides more defensible warning logic.

### 7. Maintenance opportunity can be quantified without claiming RUL

The project identifies observed operating-cycle opportunity before the dataset-defined evidently faulty region while deliberately avoiding unsupported claims about physical component lifetime.

---

# Technology Stack

- **Python**
- **Pandas**
- **NumPy**
- **scikit-learn**
- **Isolation Forest**
- **Robust statistical feature engineering**
- **Time-series / sequential degradation analysis**
- **Tableau**

---

# Repository Structure

```text
Mitsubishi-ME-AD-Predictive-Maintenance/
│
├── analysis/
│   ├── 01_data_inventory/
│   ├── 02_signal_quality/
│   ├── 03_healthy_baseline/
│   ├── 04_degradation_analysis/
│   ├── 05_health_indicator/
│   ├── 06_anomaly_detection/
│   ├── 07_early_warning/
│   └── 08_predictive_maintenance/
│
├── dashboard/
│   ├── me_ad_tableau_analytical_dataset.csv
│   └── Mitsubishi_RV7FM_Predictive_Maintenance.twb
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── external/
│
├── outputs/
├── requirements.txt
├── Business Question.docx
└── README.md
```

Large raw and intermediate datasets are excluded from version control.

---

# Analytical Guardrails

The project deliberately avoids overstating what can be inferred from the experimental data.

In particular:

- The health index is **not a probability of failure**.
- Maintenance opportunity is **not Remaining Useful Life**.
- The evidently faulty boundary is a **dataset-defined experimental region**, not necessarily the instant of physical component failure.
- Correlation between telemetry change and degradation progression does not automatically establish a unique physical causal mechanism.
- Weak observability under a particular trajectory does not imply absence of degradation.
- Isolation Forest anomaly scores represent statistical unusualness, not mechanical diagnosis.

---

# Business Value

The project demonstrates how high-frequency industrial robot telemetry can be transformed from raw machine signals into progressively more useful operational information:

**Telemetry → Behaviour → Degradation Evidence → Anomaly Evidence → Persistent Warning → Maintenance Opportunity → Decision Support**

The result is an interpretable predictive-maintenance framework that connects machine behaviour with maintenance planning while preserving the physical and statistical limitations of the underlying evidence.

---

## Dataset Attribution

This project uses the **Mitsubishi Electric Anomaly Detection (ME-AD)** dataset associated with Mitsubishi Electric Research Laboratories.

The original dataset is not redistributed through this repository.

---

## Author

**Blessing Taurai Chikowore**

Industrial Analytics | Robotics | Predictive Maintenance | Smart Manufacturing
