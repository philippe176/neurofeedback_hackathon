/**
 * Neurofeedback BCI - Web Application JavaScript
 *
 * Handles:
 * - WebSocket communication with the server
 * - Real-time visualization updates using Plotly
 * - User interface controls
 */

// ========================================
// Constants
// ========================================

const CLASS_COLORS = {
    0: '#5c9fff',  // Left Hand - Blue
    1: '#ffb347',  // Right Hand - Orange
    2: '#77dd77',  // Left Leg - Green
    3: '#ff6b6b',  // Right Leg - Coral
};

const CLASS_NAMES = {
    0: 'Left Hand',
    1: 'Right Hand',
    2: 'Left Leg',
    3: 'Right Leg',
};

const THEME = {
    bg_dark: '#0a0e1a',
    bg_medium: '#111827',
    bg_light: '#1e293b',
    text_primary: '#f1f5f9',
    text_secondary: '#94a3b8',
    text_muted: '#64748b',
    accent: '#3b82f6',
    success: '#22c55e',
    warning: '#f59e0b',
    border: '#334155',
};

// ========================================
// State
// ========================================

let socket = null;
let isStreaming = false;
let currentClass = null;
let autoTracking = false;
let centroidWindow = 50;
let currentModelType = 'dnn';
let currentModelName = 'DNN Decoder';
let currentVizMethod = 'neural';
let currentVizName = 'Neural Projection';
let controlMode = 'buttons';
let controlModeName = 'Button Controls';
let pressedArrows = [];
let trainingPhase = 'calibration';
let trainingPhaseName = 'Guided Calibration';
let trainingPhaseDescription = 'Follow the shown task cue and repeat a stable strategy until each label forms a distinct cluster.';
let currentDifficulty = 'd1';
let currentDifficultyName = 'D1 - Cardinal Pulses';
let promptState = null;
let sessionState = null;
let calibrationState = null;
let coachState = null;
let hasCustomSaveName = false;
const activeKeyboardArrows = new Set();

// ========================================
// Initialization
// ========================================

document.addEventListener('DOMContentLoaded', () => {
    initializeSocket();
    initializePlots();
    initializeControls();
    loadInitialStatus();
});

function initializeSocket() {
    socket = io();

    socket.on('connect', () => {
        console.log('Connected to server');
        updateConnectionStatus(true);
    });

    socket.on('disconnect', () => {
        console.log('Disconnected from server');
        updateConnectionStatus(false);
    });

    socket.on('update', (data) => {
        handleUpdate(data);
    });

    socket.on('streaming_status', (data) => {
        isStreaming = data.streaming;
        updateStreamingButtons();
    });

    socket.on('class_changed', (data) => {
        currentClass = data.class_idx;
        updateClassButtons();
    });

    socket.on('tracking_changed', (data) => {
        syncControlState(data);
    });

    socket.on('error', (data) => {
        console.error('Server error:', data.message);
    });

    socket.on('control_mode_changed', (data) => {
        syncControlState(data);
    });

    socket.on('control_state_changed', (data) => {
        syncControlState(data);
    });

    socket.on('training_phase_changed', (data) => {
        syncTrainingState(data);
    });

    socket.on('difficulty_changed', (data) => {
        syncTrainingState(data);
        updateSampleCounter(data.sample_count || 0);
    });

    socket.on('model_changed', (data) => {
        currentModelType = data.model_type;
        currentModelName = data.model_name;
        updateModelSelection();
    });

    socket.on('viz_method_changed', (data) => {
        currentVizMethod = data.viz_method;
        currentVizName = data.viz_name;
        updateVisualizationSelection();
    });
}

function initializePlots() {
    // Initialize manifold plot
    const manifoldLayout = {
        paper_bgcolor: THEME.bg_medium,
        plot_bgcolor: THEME.bg_medium,
        font: { color: THEME.text_secondary },
        margin: { t: 20, r: 20, b: 40, l: 50 },
        xaxis: {
            title: 'Projection X',
            gridcolor: THEME.border,
            zerolinecolor: THEME.border,
        },
        yaxis: {
            title: 'Projection Y',
            gridcolor: THEME.border,
            zerolinecolor: THEME.border,
        },
        showlegend: true,
        legend: {
            x: 1,
            y: 1,
            xanchor: 'right',
            bgcolor: 'rgba(0,0,0,0.5)',
        },
    };

    Plotly.newPlot('manifold-plot', [], manifoldLayout, {
        responsive: true,
        displayModeBar: false,
    });

    // Initialize probability bar chart
    const probsLayout = {
        paper_bgcolor: THEME.bg_medium,
        plot_bgcolor: THEME.bg_medium,
        font: { color: THEME.text_secondary },
        margin: { t: 10, r: 20, b: 40, l: 40 },
        xaxis: {
            ticktext: ['LH', 'RH', 'LL', 'RL'],
            tickvals: [0, 1, 2, 3],
        },
        yaxis: {
            range: [0, 1.1],
            gridcolor: THEME.border,
        },
        bargap: 0.3,
    };

    Plotly.newPlot('probs-plot', [], probsLayout, {
        responsive: true,
        displayModeBar: false,
    });

    // Initialize metrics plot
    const metricsLayout = {
        paper_bgcolor: THEME.bg_medium,
        plot_bgcolor: THEME.bg_medium,
        font: { color: THEME.text_secondary },
        margin: { t: 10, r: 20, b: 40, l: 50 },
        xaxis: {
            title: 'Sample',
            gridcolor: THEME.border,
        },
        yaxis: {
            range: [0, 1.1],
            gridcolor: THEME.border,
        },
        showlegend: true,
        legend: {
            x: 1,
            y: 1,
            xanchor: 'right',
            bgcolor: 'rgba(0,0,0,0.5)',
        },
    };

    Plotly.newPlot('metrics-plot', [], metricsLayout, {
        responsive: true,
        displayModeBar: false,
    });
}

function initializeControls() {
    // Start/Stop buttons
    document.getElementById('btn-start').addEventListener('click', () => {
        socket.emit('start_streaming');
    });

    document.getElementById('btn-stop').addEventListener('click', () => {
        socket.emit('stop_streaming');
    });

    // Auto-tracking button
    document.getElementById('btn-tracking').addEventListener('click', () => {
        socket.emit('toggle_tracking');
    });

    document.querySelectorAll('.btn-phase').forEach(btn => {
        btn.addEventListener('click', () => {
            const nextPhase = btn.dataset.phase;
            if (nextPhase === trainingPhase) {
                return;
            }
            socket.emit('set_training_phase', { training_phase: nextPhase });
        });
    });

    document.getElementById('difficulty-select').addEventListener('change', (event) => {
        const nextDifficulty = event.target.value;
        if (nextDifficulty === currentDifficulty) {
            return;
        }
        socket.emit('set_difficulty', { difficulty: nextDifficulty });
    });

    document.querySelectorAll('.btn-mode').forEach(btn => {
        btn.addEventListener('click', () => {
            const nextMode = btn.dataset.controlMode;
            if (nextMode === controlMode) {
                return;
            }
            socket.emit('set_control_mode', { control_mode: nextMode });
        });
    });

    // Class selection buttons
    document.querySelectorAll('.btn-class').forEach(btn => {
        btn.addEventListener('click', () => {
            const classIdx = btn.dataset.class;
            const value = classIdx === 'null' ? null : parseInt(classIdx);
            socket.emit('set_class', { class_idx: value });
        });
    });

    // Centroid window slider
    const slider = document.getElementById('centroid-slider');
    const sliderValue = document.getElementById('centroid-value');

    slider.addEventListener('input', () => {
        sliderValue.textContent = slider.value;
    });

    slider.addEventListener('change', () => {
        centroidWindow = parseInt(slider.value);
        socket.emit('set_centroid_window', { window: centroidWindow });
    });

    document.querySelectorAll('.btn-model').forEach(btn => {
        btn.addEventListener('click', () => {
            if (btn.dataset.viz) {
                return;
            }
            const modelType = btn.dataset.model;
            if (modelType === currentModelType) {
                return;
            }
            socket.emit('set_model', { model_type: modelType });
        });
    });

    document.querySelectorAll('.btn-viz').forEach(btn => {
        btn.addEventListener('click', () => {
            const vizMethod = btn.dataset.viz;
            if (vizMethod === currentVizMethod) {
                return;
            }
            socket.emit('set_viz_method', { viz_method: vizMethod });
        });
    });

    const saveInput = document.getElementById('save-model-name');
    saveInput.addEventListener('input', () => {
        hasCustomSaveName = saveInput.value.trim().length > 0 && saveInput.value !== buildDefaultSaveName();
    });

    document.getElementById('btn-save-model').addEventListener('click', saveModelSnapshot);

    window.addEventListener('keydown', handleKeyboardControl, true);
    window.addEventListener('keyup', handleKeyboardControl, true);
    window.addEventListener('blur', releaseAllKeyboardArrows);
    document.addEventListener('visibilitychange', () => {
        if (document.hidden) {
            releaseAllKeyboardArrows();
        }
    });
}

async function loadInitialStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();

        isStreaming = data.streaming;
        currentClass = data.current_class;
        autoTracking = data.auto_tracking;
        controlMode = data.control_mode || 'buttons';
        controlModeName = (data.available_control_modes || {})[controlMode] || 'Button Controls';
        pressedArrows = data.pressed_arrows || [];
        trainingPhase = data.training_phase || 'calibration';
        trainingPhaseName = data.training_phase_name || 'Guided Calibration';
        trainingPhaseDescription = data.training_phase_description || trainingPhaseDescription;
        currentDifficulty = data.difficulty || 'd1';
        currentDifficultyName = data.difficulty_name || 'D1 - Cardinal Pulses';
        promptState = data.prompt || null;
        sessionState = data.session || null;
        calibrationState = data.calibration || null;
        coachState = data.coach || null;
        currentModelType = data.model_type || 'dnn';
        currentModelName = data.model_name || 'DNN Decoder';
        currentVizMethod = data.viz_method || 'neural';
        currentVizName = data.viz_name || 'Neural Projection';

        updateStreamingButtons();
        updateClassButtons();
        updateTrackingButton();
        updateControlModeSelection();
        updatePressedArrowDisplay();
        updateTrainingPhaseSelection();
        updateDifficultySelection();
        updatePromptCoachPanel();
        updateSessionProgress();
        updateCalibrationPanel();
        updateClassControlAvailability();
        updateModelSelection();
        updateVisualizationSelection();
        ensureDefaultSaveName();
        updateSampleCounter(data.sample_count || 0);
    } catch (error) {
        console.error('Failed to load initial status:', error);
    }
}

// ========================================
// Update Handlers
// ========================================

function handleUpdate(data) {
    updateSampleCounter(data.sample_idx);
    updateStateDisplay(data);
    updateManifoldPlot(data);
    updateProbabilitiesPlot(data);
    updateMetricsPlot(data);
    updateClusterInfo(data);
    updateTrainingStatus(data);
    syncTrainingState(data);

    // Update current class from server
    if (data.current_class !== currentClass) {
        currentClass = data.current_class;
        updateClassButtons();
    }

    if (data.auto_tracking !== autoTracking) {
        autoTracking = data.auto_tracking;
        updateTrackingButton();
    }

    syncControlState(data);

    if (data.model_type && data.model_type !== currentModelType) {
        currentModelType = data.model_type;
        currentModelName = data.model_name || currentModelName;
        updateModelSelection();
    }

    if (data.viz_method && data.viz_method !== currentVizMethod) {
        currentVizMethod = data.viz_method;
        currentVizName = data.viz_name || currentVizName;
        updateVisualizationSelection();
    }
}

function updateConnectionStatus(connected) {
    const elem = document.getElementById('connection-status');
    if (connected) {
        elem.textContent = 'Connected';
        elem.className = 'status-indicator connected';
    } else {
        elem.textContent = 'Disconnected';
        elem.className = 'status-indicator disconnected';
    }
}

function updateStreamingButtons() {
    document.getElementById('btn-start').disabled = isStreaming;
    document.getElementById('btn-stop').disabled = !isStreaming;
}

function updateSampleCounter(count) {
    document.getElementById('sample-counter').textContent = `Samples: ${count}`;
}

function updateStateDisplay(data) {
    const targetName = data.prompt && data.prompt.target_class_name
        ? data.prompt.target_class_name
        : (data.current_class_name || '--');
    document.getElementById('target-class').textContent = targetName;
    document.getElementById('current-class').textContent = data.current_class_name;
    document.getElementById('predicted-class').textContent = data.predicted_class_name;
    document.getElementById('confidence').textContent = `${(data.confidence * 100).toFixed(1)}%`;
    document.getElementById('class-scale').textContent = data.class_scale.toFixed(2);
    document.getElementById('strategy-quality').textContent = data.strategy_quality.toFixed(2);
}

function updateTrainingStatus(data) {
    const training = data.training;
    document.getElementById('num-updates').textContent = training.num_updates;
    document.getElementById('labeled-seen').textContent = training.labeled_seen;
    document.getElementById('balanced-acc').textContent =
        training.balanced_accuracy !== null ? `${(training.balanced_accuracy * 100).toFixed(1)}%` : '--';
    document.getElementById('macro-f1').textContent =
        training.macro_f1 !== null ? `${(training.macro_f1 * 100).toFixed(1)}%` : '--';
    document.getElementById('rl-enabled').textContent = training.rl_enabled ? 'Yes' : 'No';
    const rollingReward = data.session && typeof data.session.rolling_reward === 'number'
        ? `${(data.session.rolling_reward * 100).toFixed(0)}%`
        : '--';
    document.getElementById('rolling-reward').textContent = rollingReward;
}

function updateClassButtons() {
    document.querySelectorAll('.btn-class').forEach(btn => {
        const classIdx = btn.dataset.class;
        const value = classIdx === 'null' ? null : parseInt(classIdx);
        btn.classList.toggle('active', value === currentClass);
    });
}

function updateClassControlAvailability() {
    const manualOnly = trainingPhase !== 'manual';
    document.querySelectorAll('.btn-class').forEach(btn => {
        btn.disabled = manualOnly;
    });

    const help = document.getElementById('manual-class-help');
    if (!help) {
        return;
    }

    if (manualOnly) {
        help.textContent = 'Class buttons are disabled in guided phases because the target cue is scheduled automatically. Switch to Manual Sandbox to force classes yourself.';
    } else {
        help.textContent = 'Manual sandbox only. Click to simulate the corresponding mental state. These same classes are also available through the keyboard emulator hotkeys.';
    }
}

function updateTrackingButton() {
    const btn = document.getElementById('btn-tracking');
    const state = document.getElementById('tracking-state');

    if (autoTracking) {
        btn.textContent = 'Disable Auto-Tracking';
        btn.classList.add('active');
        state.textContent = 'ON';
        state.classList.add('on');
    } else {
        btn.textContent = 'Enable Auto-Tracking';
        btn.classList.remove('active');
        state.textContent = 'OFF';
        state.classList.remove('on');
    }
}

function updateTrainingPhaseSelection() {
    document.querySelectorAll('.btn-phase').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.phase === trainingPhase);
    });

    document.getElementById('training-phase-name').textContent = trainingPhaseName;
    document.getElementById('training-phase-description').textContent = trainingPhaseDescription;
    ensureDefaultSaveName();
}

function updateDifficultySelection() {
    const select = document.getElementById('difficulty-select');
    if (select && select.value !== currentDifficulty) {
        select.value = currentDifficulty;
    }
    document.getElementById('difficulty-name').textContent = currentDifficultyName;
}

function updateControlModeSelection() {
    document.querySelectorAll('.btn-mode').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.controlMode === controlMode);
    });

    document.getElementById('control-mode-name').textContent = controlModeName;
}

function updatePressedArrowDisplay() {
    const elem = document.getElementById('pressed-arrows');
    if (!pressedArrows.length) {
        elem.textContent = 'None';
        return;
    }

    const labels = pressedArrows.map(key => {
        switch (key) {
            case 'ArrowLeft':
                return 'Left';
            case 'ArrowRight':
                return 'Right';
            case 'ArrowUp':
                return 'Up';
            case 'ArrowDown':
                return 'Down';
            default:
                return key;
        }
    });
    elem.textContent = labels.join(', ');
}

function syncControlState(data) {
    if (data.control_mode) {
        controlMode = data.control_mode;
    }
    if (data.control_mode_name) {
        controlModeName = data.control_mode_name;
    } else if (controlMode === 'buttons') {
        controlModeName = 'Button Controls';
    } else if (controlMode === 'keyboard') {
        controlModeName = 'Keyboard Emulator';
    }
    if (Array.isArray(data.pressed_arrows)) {
        pressedArrows = data.pressed_arrows;
        syncLocalPressedArrows();
    }
    if (typeof data.auto_tracking === 'boolean' && data.auto_tracking !== autoTracking) {
        autoTracking = data.auto_tracking;
        updateTrackingButton();
    }
    updateControlModeSelection();
    updatePressedArrowDisplay();
}

function syncLocalPressedArrows() {
    activeKeyboardArrows.clear();
    pressedArrows.forEach(key => activeKeyboardArrows.add(key));
}

function syncTrainingState(data) {
    if (data.training_phase) {
        trainingPhase = data.training_phase;
    }
    if (data.training_phase_name) {
        trainingPhaseName = data.training_phase_name;
    }
    if (data.training_phase_description) {
        trainingPhaseDescription = data.training_phase_description;
    }
    if (data.difficulty) {
        currentDifficulty = data.difficulty;
    }
    if (data.difficulty_name) {
        currentDifficultyName = data.difficulty_name;
    }
    if (data.prompt) {
        promptState = data.prompt;
    }
    if (data.session) {
        sessionState = data.session;
    }
    if (data.calibration) {
        calibrationState = data.calibration;
    }
    if (data.coach) {
        coachState = data.coach;
    }

    updateTrainingPhaseSelection();
    updateDifficultySelection();
    updatePromptCoachPanel();
    updateSessionProgress();
    updateCalibrationPanel();
    updateClassControlAvailability();
}

function updatePromptCoachPanel() {
    const guided = !!(promptState && promptState.guided);
    const badge = document.getElementById('prompt-guided-badge');
    const windowState = document.getElementById('prompt-window-state');
    const targetClass = document.getElementById('prompt-target-class');
    const targetDetail = document.getElementById('prompt-target-detail');
    const nextClass = document.getElementById('prompt-next-class');
    const nextDetail = document.getElementById('prompt-next-detail');
    const progressBar = document.getElementById('prompt-progress-bar');
    const coachBox = document.getElementById('coach-box');

    if (guided) {
        badge.textContent = trainingPhase === 'calibration' ? 'Calibration Cue' : 'Neurofeedback Cue';
        targetClass.textContent = promptState.target_class_name || '--';
        targetDetail.textContent = promptState.window_open
            ? `Act now - prompt ends in ${promptState.seconds_to_prompt_end.toFixed(1)}s`
            : `Window opens in ${promptState.seconds_to_window_start.toFixed(1)}s`;
        nextClass.textContent = promptState.next_target_class_name || '--';
        nextDetail.textContent = `Next cue in ${promptState.seconds_to_next_prompt_start.toFixed(1)}s`;
        windowState.textContent = promptState.window_open ? 'Window Open' : 'Get Ready';
        windowState.className = `prompt-window ${promptState.window_open ? 'open' : 'waiting'}`;
        progressBar.style.width = `${Math.max(0, Math.min(100, (promptState.progress || 0) * 100))}%`;
        progressBar.style.background = promptState.window_open
            ? 'linear-gradient(90deg, #22c55e, #86efac)'
            : 'linear-gradient(90deg, #38bdf8, #60a5fa)';
    } else {
        badge.textContent = 'Manual Sandbox';
        targetClass.textContent = currentClass === null ? 'Pick a class' : (CLASS_NAMES[currentClass] || '--');
        targetDetail.textContent = 'Choose the imagined movement yourself.';
        nextClass.textContent = 'Free play';
        nextDetail.textContent = 'No scheduled prompt';
        windowState.textContent = 'Manual';
        windowState.className = 'prompt-window waiting';
        progressBar.style.width = '0%';
        progressBar.style.background = 'linear-gradient(90deg, #38bdf8, #60a5fa)';
    }

    const nextState = coachState && coachState.state ? coachState.state : 'hold';
    coachBox.className = `coach-box ${nextState}`;
    document.getElementById('coach-headline').textContent = coachState && coachState.headline
        ? coachState.headline
        : 'Calibration in progress';
    document.getElementById('coach-message').textContent = coachState && coachState.message
        ? coachState.message
        : 'Keep repeating the strategy that makes the model more confident.';
    document.getElementById('coach-score-label').textContent = coachState && coachState.score_label
        ? coachState.score_label
        : 'Neurofeedback Score';
    document.getElementById('coach-score').textContent = coachState && typeof coachState.score === 'number'
        ? `${(coachState.score * 100).toFixed(0)}%`
        : '--';
    const margin = coachState && typeof coachState.target_margin === 'number'
        ? coachState.target_margin
        : 0;
    document.getElementById('coach-margin').textContent = `Margin: ${margin.toFixed(2)}`;
}

function updateSessionProgress() {
    const session = sessionState || {};
    document.getElementById('session-level').textContent =
        session.guided && session.level !== null && session.level !== undefined ? session.level : '--';
    document.getElementById('session-streak').textContent = session.streak ?? 0;
    document.getElementById('session-best-streak').textContent = session.best_streak ?? 0;
    document.getElementById('session-hit-rate').textContent =
        typeof session.hit_rate === 'number' && session.guided ? `${(session.hit_rate * 100).toFixed(0)}%` : '--';
    document.getElementById('session-prompts').textContent =
        `${session.total_prompts ?? 0} / ${session.total_hits ?? 0}`;
}

function updateCalibrationPanel() {
    const calibration = calibrationState || {};
    const readiness = typeof calibration.readiness === 'number' ? calibration.readiness : 0;
    document.getElementById('calibration-state').textContent = calibration.ready
        ? 'Ready For Feedback'
        : 'Building Clusters';
    document.getElementById('calibration-score').textContent = `${(readiness * 100).toFixed(0)}%`;
    document.getElementById('calibration-progress-bar').style.width =
        `${Math.max(0, Math.min(100, readiness * 100))}%`;
    document.getElementById('calibration-message').textContent = calibration.message
        || 'Keep collecting attempts for every task so each label forms its own cluster.';

    const counts = calibration.label_counts || {};
    for (let cls = 0; cls < 4; cls++) {
        document.getElementById(`count-class-${cls}`).textContent = counts[String(cls)] ?? 0;
    }

    if (typeof calibration.mean_label_separation === 'number') {
        const minSep = typeof calibration.min_label_separation === 'number'
            ? calibration.min_label_separation.toFixed(2)
            : '--';
        const meanSep = calibration.mean_label_separation.toFixed(2);
        document.getElementById('label-separation').textContent = `Min ${minSep} | Mean ${meanSep}`;
    } else {
        document.getElementById('label-separation').textContent = '--';
    }
}

function updateModelSelection() {
    document.querySelectorAll('.btn-model[data-model]').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.model === currentModelType);
    });

    document.getElementById('active-model-name').textContent = currentModelName;
    document.getElementById('active-model-type').textContent = currentModelType;
    ensureDefaultSaveName();
}

function updateVisualizationSelection() {
    document.querySelectorAll('.btn-viz').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.viz === currentVizMethod);
    });

    document.getElementById('active-viz-name').textContent = currentVizName;
    document.getElementById('active-viz-type').textContent = currentVizMethod;
    document.getElementById('projection-title').textContent = currentVizName;
    ensureDefaultSaveName();
}

function updateClusterInfo(data) {
    const numCentroids = Object.keys(data.centroids).length;
    document.getElementById('cluster-count').textContent = `Clusters: ${numCentroids}/4`;

    const minSep = data.min_separation.toFixed(2);
    const meanSep = data.mean_separation.toFixed(2);
    const meanSpread = data.mean_spread.toFixed(2);
    document.getElementById('cluster-separation').textContent = `MinSep: ${minSep} | MeanSep: ${meanSep} | Spread: ${meanSpread}`;
}

function buildDefaultSaveName() {
    return `${currentModelType}_${currentVizMethod}_${trainingPhase}`;
}

function ensureDefaultSaveName(force = false) {
    const input = document.getElementById('save-model-name');
    if (!input) {
        return;
    }

    if (force || !hasCustomSaveName || input.value.trim() === '') {
        input.value = buildDefaultSaveName();
        hasCustomSaveName = false;
    }
}

function setSaveStatus(message, kind = '') {
    const elem = document.getElementById('save-model-status');
    elem.textContent = message;
    elem.className = `save-status${kind ? ` ${kind}` : ''}`;
}

async function saveModelSnapshot() {
    const button = document.getElementById('btn-save-model');
    const input = document.getElementById('save-model-name');
    const name = input.value.trim() || buildDefaultSaveName();

    button.disabled = true;
    setSaveStatus('Saving snapshot...');

    try {
        const response = await fetch('/api/save_model', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ name }),
        });
        const data = await response.json();
        if (!response.ok || !data.success) {
            throw new Error(data.error || 'Failed to save model snapshot');
        }
        input.value = data.default_name || name;
        hasCustomSaveName = false;
        setSaveStatus(`Saved as ${data.filename}`, 'success');
    } catch (error) {
        console.error('Failed to save model snapshot:', error);
        setSaveStatus(error.message || 'Failed to save model snapshot', 'error');
    } finally {
        button.disabled = false;
    }
}

function handleKeyboardControl(event) {
    if (controlMode !== 'keyboard') {
        return;
    }

    const target = event.target;
    if (target && (
        target.tagName === 'INPUT' ||
        target.tagName === 'TEXTAREA' ||
        target.tagName === 'SELECT' ||
        target.isContentEditable
    )) {
        return;
    }

    const isArrow = event.key.startsWith('Arrow');
    const isClassHotkey = ['0', '1', '2', '3', '4'].includes(event.key);
    if (!isArrow && !isClassHotkey) {
        return;
    }

    event.preventDefault();

    if (event.type === 'keydown' && isClassHotkey && !event.repeat) {
        const classValue = event.key === '0' ? null : parseInt(event.key, 10) - 1;
        socket.emit('set_class', { class_idx: classValue });
        return;
    }

    if (!isArrow) {
        return;
    }

    const alreadyPressed = activeKeyboardArrows.has(event.key);
    if (event.type === 'keydown') {
        if (alreadyPressed) {
            return;
        }
        activeKeyboardArrows.add(event.key);
        socket.emit('control_key', { key: event.key, pressed: true });
        return;
    }

    if (!alreadyPressed) {
        return;
    }
    activeKeyboardArrows.delete(event.key);
    socket.emit('control_key', { key: event.key, pressed: false });
}

function releaseAllKeyboardArrows() {
    if (!socket || controlMode !== 'keyboard' || activeKeyboardArrows.size === 0) {
        return;
    }

    [...activeKeyboardArrows].forEach(key => {
        socket.emit('control_key', { key, pressed: false });
    });
    activeKeyboardArrows.clear();
}

// ========================================
// Plotting Functions
// ========================================

function updateManifoldPlot(data) {
    const points = data.points;
    const labels = data.labels || [];
    const predictions = data.predictions;
    const centroids = data.centroids;

    if (points.length === 0) return;

    // Use only recent points based on centroid window
    const windowSize = Math.min(data.centroid_window, points.length);
    const recentPoints = points.slice(-windowSize);
    const recentPreds = predictions.slice(-windowSize);
    const recentLabels = labels.slice(-windowSize).map((label, index) =>
        label === null || label === undefined ? recentPreds[index] : label
    );

    // Separate points by class
    const traces = [];

    // Add scatter points for each class
    for (let cls = 0; cls < 4; cls++) {
        const classPoints = [];
        const alphas = [];

        for (let i = 0; i < recentPoints.length; i++) {
            if (recentLabels[i] === cls) {
                classPoints.push(recentPoints[i]);
                // Fade older points
                alphas.push(0.3 + 0.7 * (i / recentPoints.length));
            }
        }

        if (classPoints.length > 0) {
            traces.push({
                x: classPoints.map(p => p[0]),
                y: classPoints.map(p => p[1]),
                mode: 'markers',
                type: 'scatter',
                name: CLASS_NAMES[cls],
                marker: {
                    color: CLASS_COLORS[cls],
                    size: 8,
                    opacity: alphas,
                },
                showlegend: true,
            });
        }
    }

    // Add centroids
    for (const [cls, centroid] of Object.entries(centroids)) {
        const classIdx = parseInt(cls);
        traces.push({
            x: [centroid[0]],
            y: [centroid[1]],
            mode: 'markers+text',
            type: 'scatter',
            name: `${CLASS_NAMES[classIdx]} Centroid`,
            text: ['LH', 'RH', 'LL', 'RL'][classIdx],
            textposition: 'top center',
            textfont: {
                color: CLASS_COLORS[classIdx],
                size: 12,
                family: 'Arial Black',
            },
            marker: {
                color: CLASS_COLORS[classIdx],
                size: 20,
                symbol: 'circle',
                line: {
                    color: 'white',
                    width: 2,
                },
            },
            showlegend: false,
        });
    }

    // Add current point highlight
    const currentPoint = points[points.length - 1];
    traces.push({
        x: [currentPoint[0]],
        y: [currentPoint[1]],
        mode: 'markers',
        type: 'scatter',
        name: 'Current',
        marker: {
            color: 'white',
            size: 14,
            symbol: 'circle',
            line: {
                color: THEME.accent,
                width: 3,
            },
        },
        showlegend: false,
    });

    Plotly.react('manifold-plot', traces, {
        paper_bgcolor: THEME.bg_medium,
        plot_bgcolor: THEME.bg_medium,
        font: { color: THEME.text_secondary },
        margin: { t: 20, r: 20, b: 40, l: 50 },
        xaxis: {
            title: 'Projection X',
            gridcolor: THEME.border,
            zerolinecolor: THEME.border,
        },
        yaxis: {
            title: 'Projection Y',
            gridcolor: THEME.border,
            zerolinecolor: THEME.border,
        },
        showlegend: true,
        legend: {
            x: 1,
            y: 1,
            xanchor: 'right',
            bgcolor: 'rgba(0,0,0,0.5)',
        },
    });
}

function updateProbabilitiesPlot(data) {
    const probs = data.probabilities;
    const predicted = data.predicted_class;
    const target = data.prompt && data.prompt.target_class !== undefined && data.prompt.target_class !== null
        ? data.prompt.target_class
        : null;

    const colors = probs.map((_, i) =>
        i === predicted ? CLASS_COLORS[i] : CLASS_COLORS[i] + '80'
    );

    const trace = {
        x: [0, 1, 2, 3],
        y: probs,
        type: 'bar',
        marker: {
            color: colors,
            line: {
                color: probs.map((_, i) => i === target ? 'white' : (i === predicted ? '#dbeafe' : 'transparent')),
                width: probs.map((_, i) => i === target ? 3 : (i === predicted ? 1.5 : 0)),
            },
        },
        text: probs.map(p => `${(p * 100).toFixed(0)}%`),
        textposition: 'outside',
        textfont: {
            color: THEME.text_primary,
        },
    };

    Plotly.react('probs-plot', [trace], {
        paper_bgcolor: THEME.bg_medium,
        plot_bgcolor: THEME.bg_medium,
        font: { color: THEME.text_secondary },
        margin: { t: 30, r: 20, b: 40, l: 40 },
        xaxis: {
            ticktext: ['LH', 'RH', 'LL', 'RL'],
            tickvals: [0, 1, 2, 3],
        },
        yaxis: {
            range: [0, 1.2],
            gridcolor: THEME.border,
        },
        bargap: 0.3,
    });
}

function updateMetricsPlot(data) {
    const confidences = data.confidences;
    const rewards = data.rewards;
    const accuracies = data.accuracies;

    // Smooth the data
    const smoothed = (arr, window) => {
        const result = [];
        for (let i = 0; i < arr.length; i++) {
            const start = Math.max(0, i - window + 1);
            const slice = arr.slice(start, i + 1);
            result.push(slice.reduce((a, b) => a + b, 0) / slice.length);
        }
        return result;
    };

    const x = Array.from({ length: confidences.length }, (_, i) => i);

    const traces = [
        {
            x: x,
            y: smoothed(confidences, 10),
            type: 'scatter',
            mode: 'lines',
            name: 'Confidence',
            line: { color: THEME.success, width: 2 },
        },
        {
            x: x,
            y: smoothed(rewards, 10),
            type: 'scatter',
            mode: 'lines',
            name: 'Reward',
            line: { color: THEME.accent, width: 2 },
        },
        {
            x: x,
            y: smoothed(accuracies, 10),
            type: 'scatter',
            mode: 'lines',
            name: 'Accuracy',
            line: { color: THEME.warning, width: 2, dash: 'dash' },
        },
    ];

    Plotly.react('metrics-plot', traces, {
        paper_bgcolor: THEME.bg_medium,
        plot_bgcolor: THEME.bg_medium,
        font: { color: THEME.text_secondary },
        margin: { t: 10, r: 20, b: 40, l: 50 },
        xaxis: {
            title: 'Sample',
            gridcolor: THEME.border,
        },
        yaxis: {
            range: [0, 1.1],
            gridcolor: THEME.border,
        },
        showlegend: true,
        legend: {
            x: 0,
            y: 1,
            xanchor: 'left',
            bgcolor: 'rgba(0,0,0,0.5)',
        },
    });
}
