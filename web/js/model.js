/**
 * GradeCompass Client-side Inference Engine
 * Author: Abdul Hayy
 *
 * Implements client-side mathematical evaluation for:
 * 1. Feature encoding (matching Section 3.2 data contract)
 * 2. Feature standardization (StandardScaler)
 * 3. Linear Regression inference (predicted average score 0-100)
 * 4. Logistic Regression inference (pass/fail probability via sigmoid)
 */

/**
 * Loads the pre-trained model parameters from JSON.
 * @param {string} path
 * @returns {Promise<Object>}
 */
async function loadModelParams(path = "./assets/model_params.json") {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Failed to load model parameters from ${path}: HTTP ${response.status}`);
  }
  return await response.json();
}

/**
 * Normalizes a category key and selected option value into the corresponding
 * one-hot column name in feature_order.
 * @param {string} categoryKey
 * @param {string} value
 * @returns {string}
 */
function getFeatureName(categoryKey, value) {
  if (categoryKey === "gender") {
    return `gender_${value}`;
  }
  if (categoryKey === "race_ethnicity") {
    return `race_${value.replace(" ", "_")}`;
  }
  if (categoryKey === "parental_education") {
    const clean = value.replace(/'/g, "").replace(/\s+/g, "_");
    return `parent_edu_${clean}`;
  }
  if (categoryKey === "lunch") {
    return `lunch_${value.replace("/", "_")}`;
  }
  if (categoryKey === "test_preparation_course") {
    return `test_prep_${value}`;
  }
  return `${categoryKey}_${value}`;
}

/**
 * Encodes the 5 user dropdown selections into a 17-length one-hot vector,
 * strictly matching feature_order in model_params.json.
 * @param {Object} selection - { gender, race_ethnicity, parental_education, lunch, test_preparation_course }
 * @param {Object} params - model_params object
 * @returns {number[]}
 */
function encodeSelection(selection, params) {
  const featureOrder = params.feature_order;
  const activeFeatures = new Set();

  for (const [key, val] of Object.entries(selection)) {
    const featureName = getFeatureName(key, val);
    activeFeatures.add(featureName);
  }

  return featureOrder.map((col) => (activeFeatures.has(col) ? 1 : 0));
}

/**
 * Applies standard scaling (z = (x - mean) / std) to raw one-hot features.
 * @param {number[]} rawFeatures
 * @param {Object} scaler - { mean: number[], std: number[] }
 * @returns {number[]}
 */
function scaleFeatures(rawFeatures, scaler) {
  return rawFeatures.map((val, idx) => {
    const mean = scaler.mean[idx];
    const std = scaler.std[idx];
    return std === 0 ? 0 : (val - mean) / std;
  });
}

/**
 * Computes standard sigmoid activation: 1 / (1 + e^-z)
 * @param {number} z
 * @returns {number}
 */
function sigmoid(z) {
  return 1 / (1 + Math.exp(-z));
}

/**
 * Predicts expected average exam score (0-100) using Linear Regression.
 * Clamps output to [0, 100].
 * @param {number[]} scaledFeatures
 * @param {Object} linearParams - { weights: number[], bias: number }
 * @returns {number}
 */
function predictScore(scaledFeatures, linearParams) {
  let score = linearParams.bias;
  for (let i = 0; i < scaledFeatures.length; i++) {
    score += scaledFeatures[i] * linearParams.weights[i];
  }
  // Clamp to [0, 100]
  return Math.min(100, Math.max(0, score));
}

/**
 * Predicts pass probability (0.0 to 1.0) using Logistic Regression.
 * Threshold is 0.5 (>= 0.5 indicates Likely to Pass).
 * @param {number[]} scaledFeatures
 * @param {Object} logisticParams - { weights: number[], bias: number }
 * @returns {number}
 */
function predictPassProbability(scaledFeatures, logisticParams) {
  let z = logisticParams.bias;
  for (let i = 0; i < scaledFeatures.length; i++) {
    z += scaledFeatures[i] * logisticParams.weights[i];
  }
  return sigmoid(z);
}

// Export for module or browser window environments
if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    loadModelParams,
    getFeatureName,
    encodeSelection,
    scaleFeatures,
    sigmoid,
    predictScore,
    predictPassProbability
  };
}
