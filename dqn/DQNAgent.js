/**
 * DQN SDK — Game-Agnostic Deep Q-Network Agent
 *
 * Philosophy: "আমি শিখি, তুমি শেখাও"
 * SDK receives state[], returns action, receives reward.
 * How states and rewards are computed is 100% consumer's business.
 *
 * Features:
 *   - Configurable layer architecture
 *   - Double-DQN with target network
 *   - Experience replay with optional prioritized sampling
 *   - Optional expert policy blend (DQfD-style guided training)
 *   - Universal persistence (Node.js filesystem + Browser localStorage)
 *   - Epsilon-greedy exploration with configurable decay
 */

import { saveToDisk, loadFromDisk, saveToBrowser, loadFromBrowser } from './ModelPersistence.js';

/** Default configuration values */
const DEFAULTS = {
    stateSize: 75,
    actionSize: 10,
    layers: [
        { units: 128, activation: 'relu' },
        { units: 128, activation: 'relu' },
        { units: 64, activation: 'relu' }
    ],
    gamma: 0.95,
    epsilon: 1.0,
    epsilonMin: 0.08,
    epsilonDecay: 0.9998,
    learningRate: 0.001,
    batchSize: 32,
    memoryCapacity: 5000,
    trainInterval: 10,
    targetUpdateInterval: 50,
    rewardScale: 0.02,
    rewardClamp: 2.0,
    qClamp: 3.0,
    autoSaveInterval: 200,
    modelName: 'DQNAgent',
    modelVersion: '1.0.0',
    storageKey: 'localstorage://dqn_model',
    diskPath: 'models/dqn_weights.json'
};

export class DQNAgent {
    /**
     * @param {object} config - Agent configuration (merged with DEFAULTS)
     */
    constructor(config = {}) {
        this.config = { ...DEFAULTS, ...config };

        // Runtime state
        this.model = null;
        this.targetModel = null;
        this.isTfReady = false;
        this.trainingLoss = 0;
        this.lastAction = 0;
        this.stepCounter = 0;
        this.trainStepCount = 0;
        this.isTrainingBusy = false;
        this.epsilon = this.config.epsilon;
        this.isWeightsLoaded = false;

        // Experience replay circular buffer (O(1) transition insertion)
        this.memory = [];
        this.memoryIndex = 0;

        // Pluggable hooks (consumer provides these)
        this._expertPolicy = null;    // { getQValues, normalize, blendSchedule }
        this._criticalFilter = null;  // (experience) => boolean
    }

    // ═══════════════════════════════════════════════════════════════
    // Plugin Hooks
    // ═══════════════════════════════════════════════════════════════

    /**
     * Set expert policy for DQfD-style guided training.
     * @param {object} policy
     * @param {function} policy.getQValues - (state) => number[] Q-values
     * @param {function} policy.normalize  - (expertQ) => number[] normalized Q-values
     * @param {function} policy.blendSchedule - (epsilon) => number [0,1] expert weight
     */
    setExpertPolicy(policy) {
        this._expertPolicy = policy;
    }

    /**
     * Set critical experience filter for prioritized replay sampling.
     * @param {function} filterFn - (experience) => boolean
     */
    setCriticalFilter(filterFn) {
        this._criticalFilter = filterFn;
    }

    // ═══════════════════════════════════════════════════════════════
    // Model Architecture
    // ═══════════════════════════════════════════════════════════════

    /**
     * Build TF.js sequential model from config.layers.
     * @returns {tf.Sequential}
     */
    buildModel() {
        const model = tf.sequential();
        const layers = this.config.layers;

        for (let i = 0; i < layers.length; i++) {
            const layerConfig = {
                units: layers[i].units,
                activation: layers[i].activation || 'relu',
                kernelInitializer: layers[i].kernelInitializer || 'heNormal'
            };
            if (i === 0) {
                layerConfig.inputShape = [this.config.stateSize];
            }
            model.add(tf.layers.dense(layerConfig));
        }

        // Output layer: linear activation for Q-values
        model.add(tf.layers.dense({
            units: this.config.actionSize,
            activation: 'linear',
            kernelInitializer: 'heNormal'
        }));

        model.compile({
            optimizer: tf.train.adam(this.config.learningRate),
            loss: 'meanSquaredError'
        });

        return model;
    }

    // ═══════════════════════════════════════════════════════════════
    // Lifecycle
    // ═══════════════════════════════════════════════════════════════

    /**
     * Initialize TF.js, build Double-DQN networks, load saved weights.
     * @returns {Promise<boolean>}
     */
    async init() {
        // Dynamic import for Node.js if tf not globally defined
        if (typeof tf === 'undefined') {
            try {
                const tfModule = await import('@tensorflow/tfjs');
                globalThis.tf = tfModule.default || tfModule;
            } catch (_) { }
        }

        if (typeof tf === 'undefined') {
            console.warn('TensorFlow.js not loaded yet, waiting...');
            return false;
        }

        // Clean up existing models
        if (this.model && typeof this.model.dispose === 'function') {
            try { this.model.dispose(); } catch (_) { }
        }
        if (this.targetModel && typeof this.targetModel.dispose === 'function') {
            try { this.targetModel.dispose(); } catch (_) { }
        }

        // Build Main + Target networks
        this.model = this.buildModel();
        this.targetModel = this.buildModel();

        this.isTfReady = true;
        console.log('TensorFlow.js Double-DQN (Main & Target Networks) initialized successfully!');

        // Load weights from disk and sync target model
        await this.loadFromDisk();
        this.updateTargetModel();
        return true;
    }

    /**
     * Dispose TF.js models to free GPU/CPU memory.
     */
    dispose() {
        if (this.model && typeof this.model.dispose === 'function') {
            try { this.model.dispose(); } catch (_) { }
        }
        if (this.targetModel && typeof this.targetModel.dispose === 'function') {
            try { this.targetModel.dispose(); } catch (_) { }
        }
        this.model = null;
        this.targetModel = null;
        this.isTfReady = false;
    }

    /**
     * Full model reset to fresh Double-DQN state.
     * @returns {Promise<boolean>}
     */
    async reset() {
        this.stepCounter = 0;
        this.trainStepCount = 0;
        this.trainingLoss = 0;
        this.epsilon = this.config.epsilon;
        this.memory = [];
        this.memoryIndex = 0;

        if (typeof localStorage !== 'undefined') {
            const prefix = this.config.storageKey || 'dqn_agent';
            localStorage.removeItem(`${prefix}_steps`);
            localStorage.removeItem(`${prefix}_loss`);
            localStorage.removeItem(`${prefix}_epsilon`);
        }

        if (this.model && typeof this.model.dispose === 'function') {
            try { this.model.dispose(); } catch (_) { }
        }
        if (this.targetModel && typeof this.targetModel.dispose === 'function') {
            try { this.targetModel.dispose(); } catch (_) { }
        }

        this.model = this.buildModel();
        this.targetModel = this.buildModel();
        this.updateTargetModel();
        await this.saveToDisk();
        console.log('🔄 Model successfully reset to fresh Double-DQN state!');
        return true;
    }

    // ═══════════════════════════════════════════════════════════════
    // Action Selection
    // ═══════════════════════════════════════════════════════════════

    /**
     * Select action using epsilon-greedy policy.
     * @param {number[]} state - State vector
     * @param {object} [opts={}] - Options
     * @param {boolean} [opts.forcePureNeural=false] - Skip exploration, pure NN inference
     * @returns {number} Selected action index
     */
    act(state, opts = {}) {
        if (!state || state.length < this.config.stateSize) return 0;

        const forcePureNeural = typeof opts === 'boolean' ? opts : (opts && opts.forcePureNeural);

        // Pure neural network inference (no exploration, no expert)
        if (forcePureNeural) {
            if (this.isTfReady && this.model && typeof tf !== 'undefined') {
                return tf.tidy(() => {
                    const stateTensor = tf.tensor2d([state]);
                    const qValues = this.model.predict(stateTensor);
                    const action = qValues.argMax(1).dataSync()[0];
                    this.lastAction = action;
                    return action;
                });
            }
            throw new Error('Pure neural network mode failed: TensorFlow.js model is not ready.');
        }

        // Explore with probability epsilon
        if (Math.random() < this.epsilon) {
            // Blend: 70% expert-guided exploration, 30% random
            if (this._expertPolicy && Math.random() < 0.70) {
                const expertQ = this._expertPolicy.getQValues(state);
                this.lastAction = expertQ.indexOf(Math.max(...expertQ));
                return this.lastAction;
            }
            this.lastAction = Math.floor(Math.random() * this.config.actionSize);
            return this.lastAction;
        }

        // Exploit using neural network
        if (this.isTfReady && this.model && typeof tf !== 'undefined') {
            try {
                return tf.tidy(() => {
                    const stateTensor = tf.tensor2d([state]);
                    const qValues = this.model.predict(stateTensor);
                    const action = qValues.argMax(1).dataSync()[0];
                    this.lastAction = action;
                    return action;
                });
            } catch (err) {
                console.warn('TensorFlow prediction error, safely falling back:', err.message);
                this.isTfReady = false;
                this.init().catch(() => { });
            }
        }

        // Fallback to expert if available
        if (this._expertPolicy) {
            const expertQ = this._expertPolicy.getQValues(state);
            this.lastAction = expertQ.indexOf(Math.max(...expertQ));
            return this.lastAction;
        }

        this.lastAction = Math.floor(Math.random() * this.config.actionSize);
        return this.lastAction;
    }

    // ═══════════════════════════════════════════════════════════════
    // Experience Replay
    // ═══════════════════════════════════════════════════════════════

    /**
     * Store experience tuple into circular replay memory (O(1) insertion).
     */
    remember(state, action, reward, nextState, done) {
        const experience = { state, action, reward, nextState, done };
        if (this.memory.length < this.config.memoryCapacity) {
            this.memory.push(experience);
        } else {
            this.memory[this.memoryIndex] = experience;
            this.memoryIndex = (this.memoryIndex + 1) % this.config.memoryCapacity;
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Training
    // ═══════════════════════════════════════════════════════════════

    /**
     * Sync target network weights from main network (cloned).
     */
    updateTargetModel() {
        if (this.model && this.targetModel && typeof tf !== 'undefined') {
            try {
                const weights = this.model.getWeights();
                const clonedWeights = weights.map(w => w.clone());
                this.targetModel.setWeights(clonedWeights);
                clonedWeights.forEach(w => w.dispose());
            } catch (err) {
                console.warn('Target model weight sync notice:', err.message);
            }
        }
    }

    /**
     * Perform one mini-batch gradient descent step (Double-DQN).
     * Supports optional expert policy blending (DQfD) and prioritized sampling.
     * @returns {Promise<void>}
     */
    async trainBatch() {
        this.stepCounter++;
        if (!this.isTfReady || this.memory.length < this.config.batchSize || this.isTrainingBusy) return;
        if (this.stepCounter % this.config.trainInterval !== 0) return;

        this.isTrainingBusy = true;

        // Decay epsilon
        if (this.epsilon > this.config.epsilonMin) {
            this.epsilon *= Math.pow(this.config.epsilonDecay, this.config.trainInterval);
            if (this.epsilon < this.config.epsilonMin) this.epsilon = this.config.epsilonMin;
        }

        // Sample batch: prioritized if criticalFilter is set, otherwise uniform
        const batch = [];
        if (this._criticalFilter) {
            const criticalPool = this.memory.filter(this._criticalFilter);
            const targetCriticalCount = Math.min(Math.floor(this.config.batchSize / 2), criticalPool.length);
            for (let i = 0; i < targetCriticalCount; i++) {
                batch.push(criticalPool[Math.floor(Math.random() * criticalPool.length)]);
            }
        }
        while (batch.length < this.config.batchSize) {
            batch.push(this.memory[Math.floor(Math.random() * this.memory.length)]);
        }

        const states = batch.map(b => b.state);
        const nextStates = batch.map(b => b.nextState);

        let statesTensor = null;
        let nextStatesTensor = null;
        let targetTensor = null;

        try {
            statesTensor = tf.tensor2d(states);
            nextStatesTensor = tf.tensor2d(nextStates);

            const currentQValues = tf.tidy(() => this.model.predict(statesTensor).arraySync());
            const onlineNextQ = tf.tidy(() => this.model.predict(nextStatesTensor).arraySync());
            const targetNextQ = tf.tidy(() => (this.targetModel || this.model).predict(nextStatesTensor).arraySync());

            // Expert blend weight (0 if no expert policy)
            const hasExpert = !!this._expertPolicy;
            const expertWeight = hasExpert
                ? (this._expertPolicy.blendSchedule
                    ? this._expertPolicy.blendSchedule(this.epsilon)
                    : 0.40 + 0.40 * Math.min(1.0, this.epsilon / 1.0))
                : 0;
            const tdWeight = 1.0 - expertWeight;

            const { rewardScale, rewardClamp, qClamp, gamma } = this.config;

            const targetQ = currentQValues.map((qVal, idx) => {
                const { state, action, reward, done } = batch[idx];
                const scaledReward = Math.max(-rewardClamp, Math.min(rewardClamp, (reward || 0) * rewardScale));

                // Double-DQN: online selects action, target evaluates Q
                const bestNextAction = onlineNextQ[idx].indexOf(Math.max(...onlineNextQ[idx]));
                const doubleDqnTarget = targetNextQ[idx][bestNextAction] || 0;

                const updatedQ = [...qVal];
                if (done) {
                    updatedQ[action] = scaledReward;
                } else {
                    updatedQ[action] = Math.max(-qClamp, Math.min(qClamp, scaledReward + gamma * doubleDqnTarget));
                }

                // Expert policy blend (if configured)
                if (hasExpert && expertWeight > 0) {
                    const expertQ = this._expertPolicy.getQValues(state);
                    const normExpert = this._expertPolicy.normalize
                        ? this._expertPolicy.normalize(expertQ)
                        : expertQ;
                    return updatedQ.map((tdVal, aIdx) => tdWeight * tdVal + expertWeight * normExpert[aIdx]);
                }

                return updatedQ;
            });

            targetTensor = tf.tensor2d(targetQ);
            const history = await this.model.fit(statesTensor, targetTensor, {
                epochs: 1,
                verbose: 0
            });

            if (history && history.history && history.history.loss) {
                this.trainingLoss = history.history.loss[0];
            }

            this.trainStepCount++;
            // Target network sync
            if (this.trainStepCount % this.config.targetUpdateInterval === 0) {
                this.updateTargetModel();
            }

            // Auto-save
            if (this.config.autoSaveInterval > 0 && this.trainStepCount % this.config.autoSaveInterval === 0) {
                this.save().catch(() => { });
            }
        } catch (err) {
            console.error('Error during RL batch training:', err);
        } finally {
            if (statesTensor) statesTensor.dispose();
            if (nextStatesTensor) nextStatesTensor.dispose();
            if (targetTensor) targetTensor.dispose();
            this.isTrainingBusy = false;
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Persistence (delegates to ModelPersistence module)
    // ═══════════════════════════════════════════════════════════════

    async saveToDisk(filePath) {
        return saveToDisk(this, filePath || this.config.diskPath);
    }

    async loadFromDisk(filePath) {
        return loadFromDisk(this, filePath || this.config.diskPath);
    }

    async saveToBrowser() {
        return saveToBrowser(this);
    }

    async loadFromBrowser() {
        return loadFromBrowser(this);
    }

    /**
     * Save model weights and state (both disk and browser by default).
     * @param {object} [options={ disk: true, browser: true }]
     * @returns {Promise<boolean>}
     */
    async save(options = { disk: true, browser: true }) {
        const saveDisk = options.disk !== false;
        const saveBrowser = options.browser !== false;
        let ok = true;
        if (saveDisk) {
            const diskOk = await this.saveToDisk();
            if (!diskOk) ok = false;
        }
        if (saveBrowser) {
            const browserOk = await this.saveToBrowser();
            if (!browserOk) ok = false;
        }
        return ok;
    }

    /** Load from disk (primary source). */
    async load() {
        return this.loadFromDisk();
    }
}
