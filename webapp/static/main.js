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

// ========================================
// Initialization
// ========================================

document.addEventListener('DOMContentLoaded', () => {
    initializeSocket();
    initializePlots();
    initializeControls();
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
        autoTracking = data.auto_tracking;
        updateTrackingButton();
    });

    socket.on('error', (data) => {
        console.error('Server error:', data.message);
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

    // Update current class from server
    if (data.current_class !== currentClass) {
        currentClass = data.current_class;
        updateClassButtons();
    }

    if (data.auto_tracking !== autoTracking) {
        autoTracking = data.auto_tracking;
        updateTrackingButton();
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
}

function updateClassButtons() {
    document.querySelectorAll('.btn-class').forEach(btn => {
        const classIdx = btn.dataset.class;
        const value = classIdx === 'null' ? null : parseInt(classIdx);
        btn.classList.toggle('active', value === currentClass);
    });
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

function updateClusterInfo(data) {
    const numCentroids = Object.keys(data.centroids).length;
    document.getElementById('cluster-count').textContent = `Clusters: ${numCentroids}/4`;

    const minSep = data.min_separation.toFixed(2);
    const meanSep = data.mean_separation.toFixed(2);
    document.getElementById('cluster-separation').textContent = `MinSep: ${minSep} | MeanSep: ${meanSep}`;
}

// ========================================
// Plotting Functions
// ========================================

function updateManifoldPlot(data) {
    const points = data.points;
    const predictions = data.predictions;
    const centroids = data.centroids;

    if (points.length === 0) return;

    // Use only recent points based on centroid window
    const windowSize = Math.min(data.centroid_window, points.length);
    const recentPoints = points.slice(-windowSize);
    const recentPreds = predictions.slice(-windowSize);

    // Separate points by class
    const traces = [];

    // Add scatter points for each class
    for (let cls = 0; cls < 4; cls++) {
        const classPoints = [];
        const alphas = [];

        for (let i = 0; i < recentPoints.length; i++) {
            if (recentPreds[i] === cls) {
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
                color: probs.map((_, i) => i === predicted ? 'white' : 'transparent'),
                width: probs.map((_, i) => i === predicted ? 2 : 0),
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
