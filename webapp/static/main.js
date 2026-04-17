const CLASS_COLORS = {
    0: '#60a5fa',
    1: '#f59e0b',
    2: '#34d399',
    3: '#f87171',
};

const CLASS_NAMES = {
    0: 'Left Hand',
    1: 'Right Hand',
    2: 'Left Leg',
    3: 'Right Leg',
};

const ABBR = ['LH', 'RH', 'LL', 'RL'];

const THEME = {
    bg: '#0c1221',
    panel: '#121a2d',
    grid: '#243149',
    text: '#e2e8f0',
    muted: '#94a3b8',
    accent: '#7dd3fc',
    success: '#34d399',
    warning: '#fbbf24',
    danger: '#f87171',
};

let socket = null;
let isStreaming = false;
let isConnected = false;
let trainingPhase = 'calibration';
let trainingPhaseName = 'Guided Calibration';
let trainingPhaseDescription = '';
let currentModelType = 'dnn';
let currentModelName = 'DNN Decoder';
let currentVizMethod = 'neural';
let currentVizName = 'Neural Projection';
let centroidWindow = 200;
let streamState = {};
let calibrationState = {};
let coachState = {};
let sessionState = {};
let latestData = null;
let hasCustomSaveName = false;
let explorationTargetClass = 0;
let explorationState = null;

document.addEventListener('DOMContentLoaded', () => {
    initializeSocket();
    initializePlots();
    initializeControls();
    loadInitialStatus();
});

function initializeSocket() {
    socket = io();

    socket.on('connect', () => {
        isConnected = true;
        updateConnectionStatus();
    });

    socket.on('disconnect', () => {
        isConnected = false;
        updateConnectionStatus();
    });

    socket.on('status', (data) => {
        if (typeof data.streaming === 'boolean') {
            isStreaming = data.streaming;
        }
        if (data.stream) {
            streamState = data.stream;
        }
        updateHeaderStatus();
        updateStreamPanel();
        updateStreamingButtons();
    });

    socket.on('streaming_status', (data) => {
        isStreaming = !!data.streaming;
        if (data.stream) {
            streamState = data.stream;
        }
        updateHeaderStatus();
        updateStreamPanel();
        updateStreamingButtons();
    });

    socket.on('training_phase_changed', (data) => {
        syncTrainingState(data);
        renderPanels();
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

    socket.on('centroid_window_changed', (data) => {
        centroidWindow = data.window;
        updateCentroidSlider();
    });

    socket.on('exploration_class_changed', (data) => {
        explorationTargetClass = data.class_idx;
        document.getElementById('exploration-class-select').value = data.class_idx;
    });

    socket.on('update', (data) => {
        latestData = data;
        syncRuntimeState(data);
        renderAll();
    });

    socket.on('error', (data) => {
        console.error(data.message);
        const message = data.message || 'Unknown server error';
        document.getElementById('stream-message').textContent = message;
        document.getElementById('save-model-status').textContent = message;
        document.getElementById('save-model-status').className = 'helper-copy error';
    });
}

function initializeControls() {
    document.getElementById('btn-start').addEventListener('click', () => {
        socket.emit('start_streaming');
    });

    document.getElementById('btn-stop').addEventListener('click', () => {
        socket.emit('stop_streaming');
    });

    document.getElementById('btn-reset').addEventListener('click', resetDecoder);

    document.querySelectorAll('.btn-phase').forEach((btn) => {
        btn.addEventListener('click', () => {
            const nextPhase = btn.dataset.phase;
            if (nextPhase === trainingPhase) {
                return;
            }
            socket.emit('set_training_phase', { training_phase: nextPhase });
        });
    });

    document.querySelectorAll('.btn-model').forEach((btn) => {
        btn.addEventListener('click', () => {
            const modelType = btn.dataset.model;
            if (modelType === currentModelType) {
                return;
            }
            socket.emit('set_model', { model_type: modelType });
        });
    });

    document.querySelectorAll('.btn-viz').forEach((btn) => {
        btn.addEventListener('click', () => {
            const vizMethod = btn.dataset.viz;
            if (vizMethod === currentVizMethod) {
                return;
            }
            socket.emit('set_viz_method', { viz_method: vizMethod });
        });
    });


    document.getElementById('exploration-class-select').addEventListener('change', (e) => {
        explorationTargetClass = parseInt(e.target.value, 10);
        socket.emit('set_exploration_class', { class_idx: explorationTargetClass });
    });

    const saveInput = document.getElementById('save-model-name');
    saveInput.addEventListener('input', () => {
        hasCustomSaveName = saveInput.value.trim().length > 0 && saveInput.value !== buildDefaultSaveName();
    });

    document.getElementById('btn-save-model').addEventListener('click', saveModelSnapshot);
}

function initializePlots() {
    Plotly.newPlot('manifold-plot', [], manifoldLayout(), plotConfig());
    Plotly.newPlot('probs-plot', [], probsLayout(), plotConfig());
    Plotly.newPlot('metrics-plot', [], metricsLayout(), plotConfig());
}

async function loadInitialStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        isStreaming = !!data.streaming;
        syncRuntimeState(data);
        renderPanels();
        updateStreamingButtons();
        ensureDefaultSaveName(true);
        updateSampleCounter(data.sample_count || 0);
    } catch (error) {
        console.error('Failed to load initial status:', error);
    }
}

function syncRuntimeState(data) {
    if (typeof data.streaming === 'boolean') {
        isStreaming = data.streaming;
    }
    if (data.stream) {
        streamState = data.stream;
    }
    syncTrainingState(data);
    if (data.model_type) {
        currentModelType = data.model_type;
    }
    if (data.model_name) {
        currentModelName = data.model_name;
    }
    if (data.viz_method) {
        currentVizMethod = data.viz_method;
    }
    if (data.viz_name) {
        currentVizName = data.viz_name;
    }
    if (data.centroid_window) {
        centroidWindow = data.centroid_window;
    }
    if (data.exploration !== undefined) {
        explorationState = data.exploration;
    }
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
    if (data.calibration) {
        calibrationState = data.calibration;
    }
    if (data.coach) {
        coachState = data.coach;
    }
    if (data.session) {
        sessionState = data.session;
    }
}

function renderAll() {
    updateHeaderStatus();
    updateStreamingButtons();
    updateSampleCounter(latestData.sample_idx || 0);
    renderPanels();
    updateManifoldPlot(latestData);
    updateProbabilitiesPlot(latestData);
    updateMetricsPlot(latestData);
}

function renderPanels() {
    updateConnectionStatus();
    updateHeaderStatus();
    updateTrainingPhaseSelection();
    updateStreamPanel();
    updateModelSelection();
    updateVisualizationSelection();
    updateCentroidSlider();
    updateStatusCards();
    updateCalibrationPanel();
    updateSignalPanel();
    updateTrainingPanel();
    updateExplorationPanel();
}

function updateConnectionStatus() {
    const elem = document.getElementById('connection-status');
    if (isConnected) {
        elem.textContent = 'Browser Connected';
        elem.className = 'status-pill connected';
    } else {
        elem.textContent = 'Browser Disconnected';
        elem.className = 'status-pill disconnected';
    }
}

function updateHeaderStatus() {
    const elem = document.getElementById('stream-status');
    const state = streamState.state || 'idle';
    elem.textContent = labelForStreamState(state);
    elem.className = `status-pill ${state}`;
}

function updateStreamingButtons() {
    document.getElementById('btn-start').disabled = isStreaming;
    document.getElementById('btn-stop').disabled = !isStreaming;
}

function updateSampleCounter(count) {
    document.getElementById('sample-counter').textContent = `Samples: ${count}`;
}

function updateTrainingPhaseSelection() {
    document.querySelectorAll('.btn-phase').forEach((btn) => {
        btn.classList.toggle('active', btn.dataset.phase === trainingPhase);
    });
    document.getElementById('training-phase-description').textContent = trainingPhaseDescription;
    ensureDefaultSaveName();
}

function updateStreamPanel() {
    document.getElementById('stream-message').textContent = streamState.message || 'Waiting for stream.';
    document.getElementById('stream-age').textContent = formatAge(streamState.last_sample_age_s);

    if (latestData && latestData.difficulty_name) {
        document.getElementById('stream-difficulty').textContent = latestData.difficulty_name;
    } else {
        document.getElementById('stream-difficulty').textContent = 'Waiting for emulator';
    }
}

function updateModelSelection() {
    document.querySelectorAll('.btn-model').forEach((btn) => {
        btn.classList.toggle('active', btn.dataset.model === currentModelType);
    });
    ensureDefaultSaveName();
}

function updateVisualizationSelection() {
    document.querySelectorAll('.btn-viz').forEach((btn) => {
        btn.classList.toggle('active', btn.dataset.viz === currentVizMethod);
    });
    document.getElementById('projection-title').textContent = currentVizName;
    ensureDefaultSaveName();
}

function updateCentroidSlider() {
    // Centroid slider removed from UI; kept as no-op for socket compat
}

function updateStatusCards() {
    const data = latestData || {};
    const currentName = data.current_class_name || 'Waiting for stream';
    const predictedName = data.predicted_class_name || 'Waiting';
    const hasLabel = data.current_class !== undefined && data.current_class !== null;

    document.getElementById('intended-task').textContent = hasLabel ? currentName : 'Pick a task';
    if (data.transition_ignored && typeof data.transition_samples_remaining === 'number') {
        document.getElementById('intended-task-subtitle').textContent =
            `New label settling in. ${data.transition_samples_remaining} more samples are ignored before this class enters calibration/exploration.`;
    } else {
        document.getElementById('intended-task-subtitle').textContent = hasLabel
            ? 'Chosen in the emulator with 1-4. Keep using arrows until the model follows it.'
            : 'Use 1-4 in the emulator to declare the task you want to train.';
    }

    document.getElementById('predicted-task').textContent = predictedName;
    if (!latestData || latestData.predicted_class === null || latestData.predicted_class === undefined) {
        document.getElementById('predicted-task-subtitle').textContent = 'The prediction will appear as soon as live samples arrive.';
    } else if (hasLabel && latestData.predicted_class === latestData.current_class) {
        document.getElementById('predicted-task-subtitle').textContent = 'The decoder agrees with your intended task.';
    } else if (hasLabel) {
        document.getElementById('predicted-task-subtitle').textContent = 'If this is not the task you want, change strategy until the readout moves.';
    } else {
        document.getElementById('predicted-task-subtitle').textContent = 'Prediction is live, but the current sample does not carry a task label.';
    }

    const guidanceCard = document.getElementById('guidance-card');
    const guidanceState = coachState.state || 'hold';
    guidanceCard.className = `guidance-banner ${guidanceState}`;
    document.getElementById('guidance-headline').textContent = coachState.headline || 'Waiting for stream';
    document.getElementById('guidance-message').textContent = coachState.message || 'Start the emulator to receive feedback.';
    document.getElementById('guidance-score-label').textContent = coachState.score_label || 'Decoder Match';
    document.getElementById('guidance-score').textContent = typeof coachState.score === 'number'
        ? `${(coachState.score * 100).toFixed(0)}%`
        : '--';

    const zone = data.zone || {};
    document.getElementById('zone-name').textContent = zone.class_name || 'Waiting';
    if (zone.source === 'centroid' && typeof zone.distance === 'number') {
        document.getElementById('zone-detail').textContent = `Distance ${zone.distance.toFixed(2)} from zone center.`;
    } else if (zone.source === 'prediction') {
        document.getElementById('zone-detail').textContent = 'Estimating zones — showing prediction.';
    } else {
        document.getElementById('zone-detail').textContent = 'Your position on the map becomes the zone feedback.';
    }
}

function updateCalibrationPanel() {
    const readiness = calibrationState.readiness || 0;
    const targetPerClass = calibrationState.target_per_class || 0;
    document.getElementById('calibration-state').textContent = calibrationState.ready ? 'Ready' : 'Building';
    document.getElementById('calibration-state').className = `big-chip ${calibrationState.ready ? 'ready' : 'neutral'}`;
    document.getElementById('calibration-score').textContent = `${(readiness * 100).toFixed(0)}%`;
    document.getElementById('calibration-progress-bar').style.width = `${Math.max(0, Math.min(100, readiness * 100))}%`;
    document.getElementById('calibration-message').textContent = calibrationState.message || 'Stable attempts from all four tasks will build clear zones here.';

    const counts = calibrationState.label_counts || {};
    for (let cls = 0; cls < 4; cls += 1) {
        const count = counts[String(cls)] ?? 0;
        document.getElementById(`count-class-${cls}`).textContent = targetPerClass
            ? `${count}/${targetPerClass}`
            : `${count}`;
    }

    if (typeof calibrationState.mean_label_separation === 'number') {
        const minSep = typeof calibrationState.min_label_separation === 'number'
            ? calibrationState.min_label_separation.toFixed(2)
            : '--';
        const meanSep = calibrationState.mean_label_separation.toFixed(2);
        document.getElementById('label-separation').textContent = `Min ${minSep} | Mean ${meanSep}`;
    } else {
        document.getElementById('label-separation').textContent = '--';
    }
}

function updateSignalPanel() {
    const data = latestData || {};
    document.getElementById('signal-score').textContent = typeof data.signal_score === 'number'
        ? `${(data.signal_score * 100).toFixed(0)}%`
        : '--';
    document.getElementById('class-scale').textContent = typeof data.class_scale === 'number'
        ? data.class_scale.toFixed(2)
        : '--';
    document.getElementById('strategy-quality').textContent = typeof data.strategy_quality === 'number'
        ? data.strategy_quality.toFixed(2)
        : '--';
    document.getElementById('agreement-rate').textContent = typeof sessionState.rolling_alignment === 'number'
        ? `${(sessionState.rolling_alignment * 100).toFixed(0)}%`
        : '--';
    document.getElementById('confidence').textContent = typeof data.confidence === 'number'
        ? `${(data.confidence * 100).toFixed(1)}%`
        : '--';
    document.getElementById('rolling-reward').textContent = typeof sessionState.rolling_reward === 'number'
        ? `${(sessionState.rolling_reward * 100).toFixed(0)}%`
        : '--';
}

function updateTrainingPanel() {
    const training = latestData && latestData.training ? latestData.training : {};
    document.getElementById('num-updates').textContent = training.num_updates ?? 0;
    document.getElementById('labeled-seen').textContent = training.labeled_seen ?? 0;
    document.getElementById('balanced-acc').textContent =
        typeof training.balanced_accuracy === 'number' ? `${(training.balanced_accuracy * 100).toFixed(1)}%` : '--';
    document.getElementById('macro-f1').textContent =
        typeof training.macro_f1 === 'number' ? `${(training.macro_f1 * 100).toFixed(1)}%` : '--';
    document.getElementById('rl-enabled').textContent = training.rl_enabled ? 'Yes' : 'No';
}

function updateExplorationPanel() {
    const panel = document.getElementById('exploration-panel');
    if (trainingPhase !== 'exploration') {
        panel.style.display = 'none';
        return;
    }
    panel.style.display = '';

    if (explorationState && explorationState.target_class !== null) {
        document.getElementById('exploration-class-select').value = explorationState.target_class;
    }

    document.getElementById('exploration-count').textContent =
        explorationState ? (explorationState.n_collected || 0) : 0;

    const analysis = explorationState ? explorationState.analysis : null;
    if (!analysis) {
        document.getElementById('exploration-clusters').textContent = 'Collecting...';
        document.getElementById('exploration-best').textContent = '--';
        document.getElementById('exploration-cluster-list').innerHTML = '';
        return;
    }

    document.getElementById('exploration-clusters').textContent = analysis.n_clusters;

    const listEl = document.getElementById('exploration-cluster-list');
    listEl.innerHTML = analysis.clusters.map((c) => `
        <div class="key-metric${c.is_best ? ' highlight' : ''}">
            <span class="metric-label">Strategy ${c.cluster_id + 1} (${c.size} pts)</span>
            <span class="metric-value">${(c.confidence * 100).toFixed(1)}%${c.is_best ? ' ★' : ''}</span>
        </div>
    `).join('');

    if (analysis.best_cluster_id !== null && analysis.best_cluster_id !== undefined) {
        const best = analysis.clusters.find((c) => c.cluster_id === analysis.best_cluster_id);
        document.getElementById('exploration-best').textContent =
            `Strategy ${analysis.best_cluster_id + 1} (${(best.confidence * 100).toFixed(1)}%)`;
    } else {
        document.getElementById('exploration-best').textContent = 'No clear winner yet';
    }
}

async function resetDecoder() {
    try {
        await fetch('/api/reset', { method: 'POST' });
        latestData = null;
        calibrationState = {};
        coachState = {};
        sessionState = {};
        await loadInitialStatus();
        Plotly.react('manifold-plot', [], manifoldLayout(), plotConfig());
        Plotly.react('probs-plot', [], probsLayout(), plotConfig());
        Plotly.react('metrics-plot', [], metricsLayout(), plotConfig());
    } catch (error) {
        console.error('Failed to reset decoder:', error);
    }
}

function buildDefaultSaveName() {
    return `${currentModelType}_${currentVizMethod}_${trainingPhase}`;
}

function ensureDefaultSaveName(force = false) {
    const input = document.getElementById('save-model-name');
    if (force || !hasCustomSaveName || input.value.trim() === '') {
        input.value = buildDefaultSaveName();
        hasCustomSaveName = false;
    }
}

function setSaveStatus(message, kind = '') {
    const elem = document.getElementById('save-model-status');
    elem.textContent = message;
    elem.className = `helper-copy${kind ? ` ${kind}` : ''}`;
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
            headers: { 'Content-Type': 'application/json' },
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
        console.error(error);
        setSaveStatus(error.message || 'Failed to save model snapshot', 'error');
    } finally {
        button.disabled = false;
    }
}

function updateManifoldPlot(data) {
    const points = data.points || [];
    const clusterPoints = data.cluster_points || [];
    const clusterLabels = data.cluster_labels || [];
    if (!points.length && !clusterPoints.length) {
        Plotly.react('manifold-plot', [], manifoldLayout(), plotConfig());
        document.getElementById('cluster-count').textContent = 'Zones: 0/4';
        document.getElementById('cluster-separation').textContent = 'MinSep: -- | MeanSep: -- | Spread: --';
        return;
    }

    const traces = [];

    for (let cls = 0; cls < 4; cls += 1) {
        const classPoints = clusterPoints.filter((_, idx) => clusterLabels[idx] === cls);
        if (!classPoints.length) {
            continue;
        }
        traces.push({
            x: classPoints.map((point) => point[0]),
            y: classPoints.map((point) => point[1]),
            mode: 'markers',
            type: 'scatter',
            name: CLASS_NAMES[cls],
            marker: {
                color: CLASS_COLORS[cls],
                size: 9,
                opacity: 0.72,
            },
        });
    }

    if (points.length) {
        const trajectory = points.slice(-24);
        traces.push({
            x: trajectory.map((point) => point[0]),
            y: trajectory.map((point) => point[1]),
            mode: 'lines',
            type: 'scatter',
            name: 'Trajectory',
            line: {
                color: 'rgba(226, 232, 240, 0.45)',
                width: 3,
            },
            showlegend: false,
        });
    }

    const centroids = data.centroids || {};
    Object.entries(centroids).forEach(([cls, centroid]) => {
        const idx = parseInt(cls, 10);
        traces.push({
            x: [centroid[0]],
            y: [centroid[1]],
            mode: 'markers+text',
            type: 'scatter',
            text: [ABBR[idx]],
            textposition: 'top center',
            textfont: {
                color: CLASS_COLORS[idx],
                size: 12,
                family: 'Trebuchet MS, sans-serif',
            },
            marker: {
                color: CLASS_COLORS[idx],
                size: 20,
                line: {
                    color: '#f8fafc',
                    width: 2,
                },
            },
            name: `${CLASS_NAMES[idx]} Center`,
            showlegend: false,
        });
    });

    const referenceCentroids = data.reference_centroids || {};
    Object.entries(referenceCentroids).forEach(([cls, centroid]) => {
        const idx = parseInt(cls, 10);
        traces.push({
            x: [centroid[0]],
            y: [centroid[1]],
            mode: 'markers',
            type: 'scatter',
            marker: {
                color: 'rgba(0,0,0,0)',
                size: 24,
                symbol: 'circle-open',
                line: {
                    color: CLASS_COLORS[idx],
                    width: 2,
                },
            },
            name: `${CLASS_NAMES[idx]} Reference`,
            showlegend: false,
        });
    });

    if (points.length) {
        const currentPoint = points[points.length - 1];
        traces.push({
            x: [currentPoint[0]],
            y: [currentPoint[1]],
            mode: 'markers',
            type: 'scatter',
            marker: {
                color: '#f8fafc',
                size: 16,
                line: {
                    color: THEME.accent,
                    width: 4,
                },
            },
            name: 'Current',
            showlegend: false,
        });
    }

    if (trainingPhase === 'exploration' && explorationState && explorationState.analysis) {
        const ea = explorationState.analysis;
        const CLUSTER_COLORS = ['#c084fc', '#22d3ee', '#fb923c', '#a3e635', '#f472b6'];
        for (let cid = 0; cid < ea.n_clusters; cid++) {
            const clusterPts = ea.points_2d.filter((_, i) => ea.cluster_labels[i] === cid);
            const cInfo = ea.clusters.find((c) => c.cluster_id === cid);
            const color = CLUSTER_COLORS[cid % CLUSTER_COLORS.length];
            traces.push({
                x: clusterPts.map((p) => p[0]),
                y: clusterPts.map((p) => p[1]),
                mode: 'markers',
                type: 'scatter',
                name: `Strat ${cid + 1} (${(cInfo.confidence * 100).toFixed(0)}%)`,
                marker: {
                    color: color,
                    size: cInfo.is_best ? 10 : 7,
                    opacity: 0.7,
                    symbol: cInfo.is_best ? 'star' : 'circle',
                },
            });
            traces.push({
                x: [cInfo.centroid_2d[0]],
                y: [cInfo.centroid_2d[1]],
                mode: 'markers+text',
                type: 'scatter',
                text: [`S${cid + 1}`],
                textposition: 'top center',
                textfont: { color: color, size: 11 },
                marker: {
                    color: color,
                    size: 18,
                    symbol: cInfo.is_best ? 'star' : 'diamond',
                    line: { color: '#f8fafc', width: 2 },
                },
                showlegend: false,
            });
        }
    }

    Plotly.react('manifold-plot', traces, manifoldLayout(), plotConfig());

    const displayCount = clusterPoints.length;
    const targetPerClass = data.calibration_samples_per_class || 0;
    const graphState = data.graph_frozen ? 'Frozen calibration map' : 'Live calibration map';
    document.getElementById('cluster-count').textContent =
        `Zones: ${Object.keys(centroids).length}/4 | ${graphState} | Bank ${displayCount}${targetPerClass ? ` (${targetPerClass}/class target)` : ''}`;
    document.getElementById('cluster-separation').textContent =
        `MinSep: ${formatFixed(data.min_separation)} | MeanSep: ${formatFixed(data.mean_separation)} | Spread: ${formatFixed(data.mean_spread)}`;
}

function updateProbabilitiesPlot(data) {
    const probs = data.probabilities || [];
    if (!probs.length) {
        Plotly.react('probs-plot', [], probsLayout(), plotConfig());
        return;
    }

    const predicted = data.predicted_class;
    const intended = data.current_class;

    Plotly.react('probs-plot', [{
        x: [0, 1, 2, 3],
        y: probs,
        type: 'bar',
        marker: {
            color: probs.map((_, idx) => (idx === predicted ? CLASS_COLORS[idx] : `${CLASS_COLORS[idx]}80`)),
            line: {
                color: probs.map((_, idx) => {
                    if (idx === intended) {
                        return '#f8fafc';
                    }
                    return idx === predicted ? '#dbeafe' : 'transparent';
                }),
                width: probs.map((_, idx) => (idx === intended ? 3 : (idx === predicted ? 1.5 : 0))),
            },
        },
        text: probs.map((value) => `${(value * 100).toFixed(0)}%`),
        textposition: 'outside',
        textfont: { color: THEME.text },
    }], probsLayout(), plotConfig());
}

function updateMetricsPlot(data) {
    const confidences = data.confidences || [];
    const rewards = data.rewards || [];
    const agreements = data.agreements || [];
    if (!confidences.length) {
        Plotly.react('metrics-plot', [], metricsLayout(), plotConfig());
        return;
    }

    const x = Array.from({ length: confidences.length }, (_, idx) => idx + 1);
    const traces = [
        {
            x,
            y: smoothSeries(agreements, 10),
            type: 'scatter',
            mode: 'lines',
            name: 'Agreement',
            line: { color: THEME.success, width: 3 },
        },
        {
            x,
            y: smoothSeries(confidences, 10),
            type: 'scatter',
            mode: 'lines',
            name: 'Confidence',
            line: { color: THEME.accent, width: 2.5 },
        },
        {
            x,
            y: smoothSeries(rewards, 10),
            type: 'scatter',
            mode: 'lines',
            name: 'Reward',
            line: { color: THEME.warning, width: 2.5, dash: 'dot' },
        },
    ];

    Plotly.react('metrics-plot', traces, metricsLayout(), plotConfig());
}

function manifoldLayout() {
    return {
        paper_bgcolor: THEME.panel,
        plot_bgcolor: THEME.panel,
        font: { color: THEME.muted, family: 'Trebuchet MS, sans-serif' },
        margin: { t: 24, r: 20, b: 48, l: 52 },
        xaxis: {
            title: 'Latent Zone X',
            gridcolor: THEME.grid,
            zerolinecolor: THEME.grid,
        },
        yaxis: {
            title: 'Latent Zone Y',
            gridcolor: THEME.grid,
            zerolinecolor: THEME.grid,
        },
        legend: {
            orientation: 'h',
            y: 1.12,
            x: 0,
            bgcolor: 'rgba(0,0,0,0)',
        },
    };
}

function probsLayout() {
    return {
        paper_bgcolor: THEME.panel,
        plot_bgcolor: THEME.panel,
        font: { color: THEME.muted, family: 'Trebuchet MS, sans-serif' },
        margin: { t: 24, r: 20, b: 40, l: 40 },
        xaxis: {
            ticktext: ABBR,
            tickvals: [0, 1, 2, 3],
        },
        yaxis: {
            range: [0, 1.2],
            gridcolor: THEME.grid,
        },
        bargap: 0.28,
    };
}

function metricsLayout() {
    return {
        paper_bgcolor: THEME.panel,
        plot_bgcolor: THEME.panel,
        font: { color: THEME.muted, family: 'Trebuchet MS, sans-serif' },
        margin: { t: 18, r: 20, b: 40, l: 46 },
        xaxis: {
            title: 'Recent Samples',
            gridcolor: THEME.grid,
        },
        yaxis: {
            range: [0, 1.05],
            gridcolor: THEME.grid,
        },
        legend: {
            orientation: 'h',
            y: 1.15,
            x: 0,
            bgcolor: 'rgba(0,0,0,0)',
        },
    };
}

function plotConfig() {
    return {
        responsive: true,
        displayModeBar: false,
    };
}

function smoothSeries(values, windowSize) {
    return values.map((_, idx) => {
        const start = Math.max(0, idx - windowSize + 1);
        const slice = values.slice(start, idx + 1);
        const valid = slice.filter((value) => Number.isFinite(value));
        if (!valid.length) {
            return 0;
        }
        return valid.reduce((sum, value) => sum + value, 0) / valid.length;
    });
}

function labelForStreamState(state) {
    switch (state) {
        case 'live':
            return 'Live Stream';
        case 'waiting':
            return 'Waiting For Stream';
        case 'stale':
            return 'Stream Paused';
        default:
            return 'Idle';
    }
}

function formatAge(value) {
    if (value === null || value === undefined || !Number.isFinite(value)) {
        return '--';
    }
    if (value < 1) {
        return `${(value * 1000).toFixed(0)} ms ago`;
    }
    return `${value.toFixed(1)} s ago`;
}

function formatFixed(value) {
    return Number.isFinite(value) ? value.toFixed(2) : '--';
}
