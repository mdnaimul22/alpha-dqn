/**
 * DQN SDK — Model Persistence Layer
 * Handles save/load of neural network weights across environments:
 *   - Node.js: filesystem (fs.readFileSync / writeFileSync)
 *   - Browser: HTTP fetch to server API + TF.js localStorage
 *
 * All methods accept a DQNAgent instance and operate on its model/config.
 */

/**
 * Save model weights and metadata to disk (JSON file).
 * In Node.js, writes directly via fs. In Browser, POSTs to /api/save-model.
 *
 * @param {object} agent - DQNAgent instance
 * @param {string} filePath - Relative path to JSON weights file
 * @returns {Promise<boolean>}
 */
export async function saveToDisk(agent, filePath) {
    let weightsData = [];
    if (agent.model && typeof tf !== 'undefined') {
        try {
            weightsData = agent.model.getWeights().map(w => ({
                name: w.name,
                shape: w.shape,
                dtype: w.dtype,
                values: Array.from(w.dataSync())
            }));
        } catch (err) {
            console.warn('Failed to extract model weights:', err.message);
            return false;
        }
    }

    if (agent.model && (!Array.isArray(weightsData) || weightsData.length === 0)) {
        console.warn('Refusing to save empty weights manifest: neural network model is active but yielded 0 weight tensors.');
        return false;
    }

    const data = {
        modelName: agent.config.modelName || 'DQNAgent',
        version: agent.config.modelVersion || '1.0.0',
        format: 'dqn_dense_weights',
        stateSize: agent.config.stateSize,
        actionSize: agent.config.actionSize,
        savedAt: new Date().toISOString(),
        hyperparameters: {
            gamma: agent.config.gamma,
            epsilonMin: agent.config.epsilonMin,
            learningRate: agent.config.learningRate
        },
        metadata: {
            trainingLoss: agent.trainingLoss,
            stepsTrained: agent.trainStepCount,
            environmentSteps: agent.stepCounter,
            epsilon: agent.epsilon
        },
        weights: weightsData
    };

    if (typeof process !== 'undefined' && process.versions && process.versions.node) {
        try {
            const fs = await import('node:fs');
            const path = await import('node:path');
            const fullPath = path.resolve(filePath);
            fs.writeFileSync(fullPath, JSON.stringify(data, null, 2), 'utf8');
            console.log(`💾 Auto-saved unified Disk Model to: ${fullPath}`);
            return true;
        } catch (err) {
            console.warn(`Failed to save disk model to ${filePath}:`, err.message);
            return false;
        }
    } else if (typeof fetch !== 'undefined') {
        try {
            const res = await fetch('/api/save-model', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            if (res.ok) {
                console.log(`💾 Auto-saved live model from Browser to disk: ${filePath}`);
                return true;
            }
            return false;
        } catch (err) {
            console.warn('Browser disk sync notice:', err.message);
            return false;
        }
    }

    console.warn(`saveToDisk failed: no supported persistence runtime found for ${filePath}`);
    return false;
}

/**
 * Load model weights and metadata from disk (JSON file).
 * In Node.js, reads via fs. In Browser, fetches via HTTP GET.
 *
 * @param {object} agent - DQNAgent instance
 * @param {string} filePath - Relative path to JSON weights file
 * @returns {Promise<boolean>}
 */
export async function loadFromDisk(agent, filePath) {
    let data = null;
    const storagePrefix = agent.config.storageKey || 'dqn_agent';

    if (typeof process !== 'undefined' && process.versions && process.versions.node) {
        try {
            const fs = await import('node:fs');
            const path = await import('node:path');
            const fullPath = path.resolve(filePath);
            if (fs.existsSync(fullPath)) {
                data = JSON.parse(fs.readFileSync(fullPath, 'utf8'));
            }
        } catch (err) {
            console.warn(`[Node.js] Notice on disk model at ${filePath}:`, err.message);
        }
    } else if (typeof fetch !== 'undefined') {
        try {
            const res = await fetch(`/${filePath}`);
            if (res.ok) {
                data = await res.json();
            }
        } catch (err) {
            console.warn(`[Browser] Notice on disk model fetch at /${filePath}:`, err.message);
        }
    }

    if (data && data.version) {
        if (data.metadata) {
            if (typeof data.metadata.stepsTrained === 'number') {
                agent.trainStepCount = data.metadata.stepsTrained;
                agent.stepCounter = typeof data.metadata.environmentSteps === 'number'
                    ? data.metadata.environmentSteps
                    : data.metadata.stepsTrained;
                if (typeof localStorage !== 'undefined') {
                    localStorage.setItem(`${storagePrefix}_steps`, String(agent.stepCounter));
                    localStorage.setItem(`${storagePrefix}_train_steps`, String(agent.trainStepCount));
                }
            }
            if (typeof data.metadata.trainingLoss === 'number') {
                agent.trainingLoss = data.metadata.trainingLoss;
                if (typeof localStorage !== 'undefined') {
                    localStorage.setItem(`${storagePrefix}_loss`, String(agent.trainingLoss));
                }
            }
            if (typeof data.metadata.epsilon === 'number') {
                agent.epsilon = data.metadata.epsilon;
                if (typeof localStorage !== 'undefined') {
                    localStorage.setItem(`${storagePrefix}_epsilon`, String(agent.epsilon));
                }
            } else if (agent.stepCounter > 0) {
                agent.epsilon = agent.config.epsilonMin;
            }
        } else if (agent.stepCounter > 0) {
            agent.epsilon = agent.config.epsilonMin;
        }

        // Restore neural network layer weights (with stateSize compatibility check)
        if (agent.model && typeof tf !== 'undefined' && Array.isArray(data.weights) && data.weights.length > 0) {
            if (data.stateSize && data.stateSize !== agent.config.stateSize) {
                console.warn(`[ModelPersistence] Incompatible state size: disk model has ${data.stateSize}, agent expects ${agent.config.stateSize}. Skipping weight restore.`);
            } else {
                try {
                    const weightTensors = data.weights.map(w => tf.tensor(w.values, w.shape, w.dtype || 'float32'));
                    agent.model.setWeights(weightTensors);
                    weightTensors.forEach(t => t.dispose());
                    agent.isWeightsLoaded = true;
                    console.log(`🧠 Successfully restored ${data.weights.length} Neural Network weight tensors to model.`);
                } catch (weightErr) {
                    console.warn('Could not restore dense weight tensors from disk payload:', weightErr.message);
                }
            }
        }

        console.log(`✅ Loaded unified Disk Model: ${data.modelName || 'DQN'} (v${data.version}, StateDim=${data.stateSize}, Steps=${agent.stepCounter}, Loss=${agent.trainingLoss}, Epsilon=${agent.epsilon.toFixed(4)})`);
        return true;
    }
    return false;
}

/**
 * Save model to browser localStorage / IndexedDB via TF.js.
 * Strictly operates on browser client-side storage without redundant network POST.
 *
 * @param {object} agent - DQNAgent instance
 * @returns {Promise<boolean>}
 */
export async function saveToBrowser(agent) {
    const storagePrefix = agent.config.storageKey || 'dqn_agent';

    if (typeof localStorage !== 'undefined') {
        localStorage.setItem(`${storagePrefix}_steps`, String(agent.stepCounter));
        localStorage.setItem(`${storagePrefix}_loss`, String(agent.trainingLoss));
        localStorage.setItem(`${storagePrefix}_epsilon`, String(agent.epsilon));
    }

    // If in Node.js or localStorage not present, return true gracefully
    const isNode = typeof process !== 'undefined' && process.versions && process.versions.node;
    if (isNode || typeof localStorage === 'undefined') {
        return true;
    }

    if (!agent.model || typeof tf === 'undefined') {
        console.warn('Cannot save model: TensorFlow.js is not initialized.');
        return false;
    }

    const browserKey = agent.config.storageKey || 'localstorage://dqn_model';
    try {
        await agent.model.save(browserKey);
        console.log(`✅ Deep Q-Network saved successfully to: ${browserKey}`);
        return true;
    } catch (err) {
        console.warn(`Could not save model to ${browserKey}:`, err.message);
        return false;
    }
}

/**
 * Load model from browser localStorage / IndexedDB via TF.js.
 * Synchronizes Target Model upon successful load to prevent stale targets.
 *
 * @param {object} agent - DQNAgent instance
 * @returns {Promise<boolean>}
 */
export async function loadFromBrowser(agent) {
    const storagePrefix = agent.config.storageKey || 'dqn_agent';

    if (typeof localStorage !== 'undefined') {
        const savedSteps = localStorage.getItem(`${storagePrefix}_steps`);
        if (savedSteps !== null) agent.stepCounter = parseInt(savedSteps, 10);
        const savedLoss = localStorage.getItem(`${storagePrefix}_loss`);
        if (savedLoss !== null) agent.trainingLoss = parseFloat(savedLoss);
        const savedEpsilon = localStorage.getItem(`${storagePrefix}_epsilon`);
        if (savedEpsilon !== null) {
            agent.epsilon = parseFloat(savedEpsilon);
        } else {
            agent.epsilon = agent.config.epsilonMin;
        }
    }

    if (typeof tf === 'undefined') return false;

    const browserKey = agent.config.storageKey || 'localstorage://dqn_model';
    try {
        const loaded = await tf.loadLayersModel(browserKey);
        if (loaded) {
            // Compatibility dry-run: test predict with dummy tensor
            let isCompatible = true;
            try {
                tf.tidy(() => {
                    const dummy = tf.zeros([1, agent.config.stateSize]);
                    loaded.predict(dummy);
                });
            } catch (_) {
                isCompatible = false;
            }

            if (!isCompatible) {
                console.warn(`⚠️ Incompatible checkpoint detected in ${browserKey}. Purging...`);
                try { await tf.io.removeModel(browserKey); } catch (_) { }
                await agent.init();
                return false;
            }

            // Dispose old model and take ownership of loaded model
            if (agent.model && typeof agent.model.dispose === 'function') {
                try { agent.model.dispose(); } catch (_) { }
            }

            agent.model = loaded;
            agent.model.compile({
                optimizer: tf.train.adam(agent.config.learningRate),
                loss: 'meanSquaredError'
            });
            agent.isTfReady = true;
            if (typeof agent.epsilon !== 'number' || isNaN(agent.epsilon)) {
                agent.epsilon = agent.config.epsilonMin;
            }

            // Synchronize Target Network with the newly loaded weights
            agent.updateTargetModel();

            console.log(`✅ Deep Q-Network loaded successfully from: ${browserKey} (Target Network Synced, Epsilon=${agent.epsilon})`);
            return true;
        }
    } catch (err) {
        console.info(`No saved model at ${browserKey}. Running with initialized baseline.`);
        return false;
    }
    return false;
}
