const svg = document.getElementById('networkSvg');
const timelineInput = document.getElementById('timelineInput');
const playBtn = document.getElementById('playBtn');
const pauseBtn = document.getElementById('pauseBtn');
const stepBtn = document.getElementById('stepBtn');
const resetBtn = document.getElementById('resetBtn');
const speedInput = document.getElementById('speedInput');
const speedLabel = document.getElementById('speedLabel');
const x1Input = document.getElementById('x1Input');
const x2Input = document.getElementById('x2Input');
const w1Input = document.getElementById('w1Input');
const w2Input = document.getElementById('w2Input');
const bInput = document.getElementById('bInput');
const x1Value = document.getElementById('x1Value');
const x2Value = document.getElementById('x2Value');
const w1Value = document.getElementById('w1Value');
const w2Value = document.getElementById('w2Value');
const bValue = document.getElementById('bValue');
const phaseLabel = document.getElementById('phaseLabel');
const stepLabel = document.getElementById('stepLabel');
const outputLabel = document.getElementById('outputLabel');
const stateOutput = document.getElementById('stateOutput');

const tanh = (x) => Math.tanh(x);
const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
const format = (value, digits = 4) => Number.isFinite(value) ? value.toFixed(digits) : 'NaN';

const layout = {
    input: [
        { id: 'x1', label: 'x1', x: 110, y: 160 },
        { id: 'x2', label: 'x2', x: 110, y: 360 },
    ],
    weights: [
        { id: 'w1', label: 'w1', x: 110, y: 560 },
        { id: 'w2', label: 'w2', x: 390, y: 560 },
        { id: 'b', label: 'b', x: 670, y: 560 },
    ],
    intermediates: [
        { id: 'x1w1', label: 'x1*w1', x: 390, y: 140 },
        { id: 'x2w2', label: 'x2*w2', x: 390, y: 350 },
        { id: 'sum', label: 'x1w1+x2w2', x: 670, y: 245 },
        { id: 'n', label: 'n', x: 930, y: 245 },
    ],
    output: [{ id: 'o', label: 'o', x: 1180, y: 245 }],
};

const learningRate = 0.1;

const state = {
    x1: 2,
    x2: 0,
    w1: -3,
    w2: 1,
    b: 6.8813735870195432,
    speed: 1,
    isPlaying: false,
    frameIndex: 0,
    frames: [],
};

function buildComputation(x1, x2, w1, w2, b) {
    const x1w1 = x1 * w1;
    const x2w2 = x2 * w2;
    const sum = x1w1 + x2w2;
    const n = sum + b;
    const o = tanh(n);

    const doGrad = 1.0;
    const dn = (1 - o ** 2) * doGrad;
    const dx1w1x2w2 = dn;
    const db = dn;
    const dx1w1 = dx1w1x2w2;
    const dx2w2 = dx1w1x2w2;
    const dx1 = w1 * dx1w1;
    const dw1 = x1 * dx1w1;
    const dx2 = w2 * dx2w2;
    const dw2 = x2 * dx2w2;

    const nextValues = {
        w1: w1 - learningRate * dw1,
        w2: w2 - learningRate * dw2,
        b: b - learningRate * db,
    };

    return {
        x1,
        x2,
        w1,
        w2,
        b,
        forward: { x1w1, x2w2, sum, n, o },
        backward: { doGrad, dn, dx1w1x2w2, db, dx1w1, dx2w2, dx1, dw1, dx2, dw2 },
        nextValues,
    };
}

function buildFrames(computation) {
    const { x1, x2, w1, w2, b, forward, backward, nextValues } = computation;

    return [
        {
            phase: 'idle',
            title: 'Ready',
            activeNodes: [],
            activeEdges: [],
            summary: 'Adjust the neuron inputs and weights, then press Play to watch the exact micrograd graph.',
            data: { x1, x2, w1, w2, b },
        },
        {
            phase: 'forward',
            title: 'Multiply x1 and w1',
            activeNodes: ['x1', 'w1', 'x1w1'],
            activeEdges: ['x1-x1w1', 'w1-x1w1'],
            summary: 'Compute x1*w1.',
            data: { x1w1: forward.x1w1 },
        },
        {
            phase: 'forward',
            title: 'Multiply x2 and w2',
            activeNodes: ['x2', 'w2', 'x2w2'],
            activeEdges: ['x2-x2w2', 'w2-x2w2'],
            summary: 'Compute x2*w2.',
            data: { x2w2: forward.x2w2 },
        },
        {
            phase: 'forward',
            title: 'Add the products',
            activeNodes: ['x1w1', 'x2w2', 'sum'],
            activeEdges: ['x1w1-sum', 'x2w2-sum'],
            summary: 'Add x1*w1 and x2*w2.',
            data: { sum: forward.sum },
        },
        {
            phase: 'forward',
            title: 'Add the bias',
            activeNodes: ['sum', 'b', 'n'],
            activeEdges: ['sum-n', 'b-n'],
            summary: 'Add the bias term b to get n.',
            data: { n: forward.n },
        },
        {
            phase: 'forward',
            title: 'Apply tanh',
            activeNodes: ['n', 'o'],
            activeEdges: ['n-o'],
            summary: 'Pass n through tanh to get the output o.',
            data: { o: forward.o },
        },
        {
            phase: 'backward',
            title: 'Seed output gradient',
            activeNodes: ['o'],
            activeEdges: ['n-o'],
            summary: 'Start backprop by setting o.grad = 1.0.',
            data: { doGrad: backward.doGrad },
        },
        {
            phase: 'backward',
            title: 'Backprop through tanh',
            activeNodes: ['o', 'n'],
            activeEdges: ['n-o'],
            summary: 'Compute n.grad = 1 - o.data**2.',
            data: { dn: backward.dn },
        },
        {
            phase: 'backward',
            title: 'Backprop to the sum',
            activeNodes: ['n', 'sum', 'b'],
            activeEdges: ['sum-n', 'b-n'],
            summary: 'The gradient flowing into n is passed back to the sum and the bias.',
            data: { dx1w1x2w2: backward.dx1w1x2w2, db: backward.db },
        },
        {
            phase: 'backward',
            title: 'Split gradient to the products',
            activeNodes: ['sum', 'x1w1', 'x2w2'],
            activeEdges: ['x1w1-sum', 'x2w2-sum'],
            summary: 'The sum gradient is copied to both product nodes.',
            data: { dx1w1: backward.dx1w1, dx2w2: backward.dx2w2 },
        },
        {
            phase: 'backward',
            title: 'Final gradients',
            activeNodes: ['x1', 'x2', 'w1', 'w2'],
            activeEdges: ['x1-x1w1', 'w1-x1w1', 'x2-x2w2', 'w2-x2w2'],
            summary: 'Use the product rule to push gradients back to the inputs and weights.',
            data: { dx1: backward.dx1, dw1: backward.dw1, dx2: backward.dx2, dw2: backward.dw2 },
        },
    ].map((frame, index) => ({ ...frame, index, computation }));
}

function edgeBetween(source, target) {
    return `${source}-${target}`;
}

function createSvgElement(tag, attrs = {}) {
    const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const [key, value] of Object.entries(attrs)) {
        el.setAttribute(key, String(value));
    }
    return el;
}

function drawBaseGraph() {
    svg.innerHTML = '';

    const defs = createSvgElement('defs');
    defs.innerHTML = `
    <linearGradient id="edgeGlow" x1="0" x2="1">
      <stop offset="0%" stop-color="#6ee7b7" stop-opacity="0.9" />
      <stop offset="100%" stop-color="#66d9ef" stop-opacity="0.9" />
    </linearGradient>
        <marker id="arrowHead" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#d4e4ff"></path>
        </marker>
  `;
    svg.appendChild(defs);

    const edges = [
        [layout.input[0], layout.intermediates[0]],
        [layout.weights[0], layout.intermediates[0]],
        [layout.input[1], layout.intermediates[1]],
        [layout.weights[1], layout.intermediates[1]],
        [layout.intermediates[0], layout.intermediates[2]],
        [layout.intermediates[1], layout.intermediates[2]],
        [layout.intermediates[2], layout.intermediates[3]],
        [layout.weights[2], layout.intermediates[3]],
        [layout.intermediates[3], layout.output[0]],
    ];

    for (const [source, target] of edges) {
        const path = createSvgElement('path', {
            d: `M ${source.x + 58} ${source.y} C ${source.x + 180} ${source.y}, ${target.x - 180} ${target.y}, ${target.x - 58} ${target.y}`,
            class: 'edge',
            id: edgeBetween(source.id, target.id),
            'marker-end': 'url(#arrowHead)',
        });
        svg.appendChild(path);
    }

    for (const node of [...layout.input, ...layout.weights, ...layout.intermediates, ...layout.output]) {
        const group = createSvgElement('g', { class: 'node', id: `node-${node.id}` });
        group.appendChild(createSvgElement('circle', { cx: node.x, cy: node.y, r: 52, class: 'node-ring' }));
        group.appendChild(createSvgElement('text', { x: node.x, y: node.y - 4, 'text-anchor': 'middle', class: 'node-label' }));
        group.lastChild.textContent = node.label;
        group.appendChild(createSvgElement('text', { x: node.x, y: node.y + 20, 'text-anchor': 'middle', class: 'node-value', id: `value-${node.id}` }));
        svg.appendChild(group);
    }

    const annotations = [
        { x: 110, y: 72, text: 'Inputs' },
        { x: 390, y: 72, text: 'Products' },
        { x: 670, y: 72, text: 'Sum and bias' },
        { x: 1180, y: 72, text: 'Output' },
    ];

    for (const annotation of annotations) {
        const text = createSvgElement('text', {
            x: annotation.x,
            y: annotation.y,
            'text-anchor': 'middle',
            class: 'node-label',
            style: 'font-size: 20px; letter-spacing: 0.06em; fill: #a9bce8;',
        });
        text.textContent = annotation.text;
        svg.appendChild(text);
    }
}

function renderFrame(frame) {
    const c = frame.computation;
    const current = frame.data;
    const values = {
        x1: c.x1,
        x2: c.x2,
        w1: c.w1,
        w2: c.w2,
        b: c.b,
        x1w1: c.forward.x1w1,
        x2w2: c.forward.x2w2,
        sum: c.forward.sum,
        n: c.forward.n,
        o: c.forward.o,
    };

    const labels = {
        x1: format(c.x1),
        x2: format(c.x2),
        w1: format(c.w1),
        w2: format(c.w2),
        b: format(c.b),
        x1w1: format(c.forward.x1w1),
        x2w2: format(c.forward.x2w2),
        sum: format(c.forward.sum),
        n: format(c.forward.n),
        o: format(c.forward.o),
    };

    for (const id of Object.keys(values)) {
        const text = document.getElementById(`value-${id}`);
        if (text) {
            text.textContent = labels[id];
        }
    }

    for (const node of [...layout.input, ...layout.weights, ...layout.intermediates, ...layout.output]) {
        const el = document.getElementById(`node-${node.id}`);
        el?.classList.toggle('active', frame.activeNodes.includes(node.id));
        el?.classList.toggle('primary', node.id === 'o' && frame.phase !== 'idle');
    }

    for (const edge of svg.querySelectorAll('.edge')) {
        edge.classList.remove('active', 'forward-flow', 'backward-flow', 'update-flow');
        edge.removeAttribute('marker-start');
        edge.removeAttribute('marker-end');
    }

    for (const edgeId of frame.activeEdges) {
        const edge = document.getElementById(edgeId);
        if (!edge) {
            continue;
        }
        edge.classList.add('active');
        if (frame.phase === 'forward') {
            edge.classList.add('forward-flow');
            edge.setAttribute('marker-end', 'url(#arrowHead)');
        } else if (frame.phase === 'backward') {
            edge.classList.add('backward-flow');
            edge.setAttribute('marker-start', 'url(#arrowHead)');
        } else if (frame.phase === 'update') {
            edge.classList.add('update-flow');
            edge.setAttribute('marker-end', 'url(#arrowHead)');
        }
    }

    phaseLabel.textContent = frame.phase.charAt(0).toUpperCase() + frame.phase.slice(1);
    stepLabel.textContent = `${frame.index + 1} / ${state.frames.length}`;
    outputLabel.textContent = format(c.forward.o, 4);

    const serializable = {
        title: frame.title,
        phase: frame.phase,
        summary: frame.summary,
        x1: c.x1,
        x2: c.x2,
        w1: c.w1,
        w2: c.w2,
        b: c.b,
        x1w1: c.forward.x1w1,
        x2w2: c.forward.x2w2,
        sum: c.forward.sum,
        n: c.forward.n,
        gradients: c.backward,
        nextWeights: c.nextValues,
        extra: current,
    };

    stateOutput.textContent = JSON.stringify(serializable, null, 2);
    timelineInput.value = String(frame.index / (state.frames.length - 1));
}

function refresh() {
    state.frames = buildFrames(buildComputation(state.x1, state.x2, state.w1, state.w2, state.b));
    const frame = state.frames[state.frameIndex] ?? state.frames[0];
    renderFrame(frame);
}

function goToFrame(index) {
    state.frameIndex = clamp(index, 0, state.frames.length - 1);
    renderFrame(state.frames[state.frameIndex]);
}

function play() {
    if (state.isPlaying) {
        return;
    }

    state.isPlaying = true;
    const tick = () => {
        if (!state.isPlaying) {
            return;
        }

        if (state.frameIndex >= state.frames.length - 1) {
            state.isPlaying = false;
            return;
        }

        state.frameIndex += 1;
        renderFrame(state.frames[state.frameIndex]);
        const delay = 900 / state.speed;
        window.setTimeout(tick, delay);
    };

    tick();
}

function pause() {
    state.isPlaying = false;
}

function reset() {
    pause();
    state.frameIndex = 0;
    refresh();
}

function bindInput(input, valueEl, key) {
    const update = () => {
        state[key] = Number(input.value);
        valueEl.textContent = format(state[key], 1);
        state.frameIndex = 0;
        refresh();
    };

    input.addEventListener('input', update);
    update();
}

drawBaseGraph();

bindInput(x1Input, x1Value, 'x1');
bindInput(x2Input, x2Value, 'x2');
bindInput(w1Input, w1Value, 'w1');
bindInput(w2Input, w2Value, 'w2');
bindInput(bInput, bValue, 'b');

speedInput.addEventListener('input', () => {
    state.speed = Number(speedInput.value);
    speedLabel.textContent = `${state.speed.toFixed(2).replace(/\.00$/, '.0')}x`;
});

timelineInput.addEventListener('input', () => {
    pause();
    const ratio = Number(timelineInput.value);
    const index = Math.round(ratio * (state.frames.length - 1));
    goToFrame(index);
});

playBtn.addEventListener('click', play);
pauseBtn.addEventListener('click', pause);
stepBtn.addEventListener('click', () => {
    pause();
    goToFrame(state.frameIndex + 1);
});
resetBtn.addEventListener('click', reset);

state.speed = Number(speedInput.value);
speedLabel.textContent = `${state.speed.toFixed(2).replace(/\.00$/, '.0')}x`;
refresh();
