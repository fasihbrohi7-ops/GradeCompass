/**
 * GradeCompass UI Controller
 * Author: Abdul Hayy
 *
 * Coordinates DOM interactions, dropdown population from model metadata,
 * real-time inference triggering, progress gauge animations, and chart updates.
 */

let modelParams = null;

// Label mappings for human-readable dropdown presentation
const CATEGORY_META = {
  gender: {
    label: "Gender",
    formatOption: (opt) => opt.charAt(0).toUpperCase() + opt.slice(1)
  },
  race_ethnicity: {
    label: "Race / Ethnicity Group",
    formatOption: (opt) => opt.replace("group ", "Group ")
  },
  parental_education: {
    label: "Parental Level of Education",
    formatOption: (opt) => {
      const titles = {
        "some high school": "Some High School",
        "high school": "High School",
        "some college": "Some College",
        "associate's degree": "Associate's Degree",
        "bachelor's degree": "Bachelor's Degree",
        "master's degree": "Master's Degree"
      };
      return titles[opt] || opt;
    }
  },
  lunch: {
    label: "Lunch Type (Socioeconomic Indicator)",
    formatOption: (opt) => (opt === "standard" ? "Standard Lunch" : "Free / Reduced Lunch")
  },
  test_preparation_course: {
    label: "Test Preparation Course",
    formatOption: (opt) => (opt === "completed" ? "Completed Course" : "None (No Course)")
  }
};

/**
 * Initializes the application on DOM ready
 */
document.addEventListener("DOMContentLoaded", async () => {
  const loadingEl = document.getElementById("loading-state");
  const errorEl = document.getElementById("error-state");
  const appContainer = document.getElementById("app-content");

  try {
    modelParams = await loadModelParams("./assets/model_params.json");
    if (loadingEl) loadingEl.style.display = "none";
    if (appContainer) appContainer.style.display = "block";

    initDropdowns();
    attachEventListeners();
    updatePredictions();
  } catch (err) {
    console.error("Initialization error:", err);
    if (loadingEl) loadingEl.style.display = "none";
    if (errorEl) {
      errorEl.style.display = "block";
      const errMsg = document.getElementById("error-message");
      if (errMsg) errMsg.textContent = err.message;
    }
  }
});

/**
 * Dynamically populates the 5 dropdowns from modelParams.categorical_options
 */
function initDropdowns() {
  const form = document.getElementById("demographics-form");
  if (!form || !modelParams) return;

  const categories = Object.keys(modelParams.categorical_options);

  categories.forEach((catKey) => {
    const select = document.getElementById(`select-${catKey}`);
    if (!select) return;

    select.innerHTML = "";
    const options = modelParams.categorical_options[catKey];

    options.forEach((optValue) => {
      const optionEl = document.createElement("option");
      optionEl.value = optValue;
      const formatter = CATEGORY_META[catKey]?.formatOption;
      optionEl.textContent = formatter ? formatter(optValue) : optValue;
      select.appendChild(optionEl);
    });

    // Default to first option
    if (options.length > 0) {
      select.value = options[0];
    }
  });
}

/**
 * Retrieves the current user selection from the form
 * @returns {Object}
 */
function getCurrentSelection() {
  const categories = Object.keys(modelParams.categorical_options);
  const selection = {};

  categories.forEach((catKey) => {
    const select = document.getElementById(`select-${catKey}`);
    if (select) {
      selection[catKey] = select.value;
    }
  });

  return selection;
}

/**
 * Executes inference and updates UI elements with predictions
 */
function updatePredictions() {
  if (!modelParams) return;

  const selection = getCurrentSelection();

  // 1. Feature encoding & scaling
  const encoded = encodeSelection(selection, modelParams);
  const scaled = scaleFeatures(encoded, modelParams.scaler);

  // 2. Predict linear regression (average score 0-100)
  const predictedScore = predictScore(scaled, modelParams.linear_regression);

  // 3. Predict logistic regression (pass probability)
  const passProbability = predictPassProbability(scaled, modelParams.logistic_regression);
  const isLikelyToPass = passProbability >= 0.5;

  // 4. Update Score Display & Gauge
  const scoreNumberEl = document.getElementById("score-number");
  const scoreProgressEl = document.getElementById("score-progress");
  const scoreGaugeCircle = document.getElementById("score-gauge-circle");

  if (scoreNumberEl) {
    scoreNumberEl.textContent = predictedScore.toFixed(1);
  }

  // If using horizontal progress bar
  if (scoreProgressEl) {
    scoreProgressEl.style.width = `${Math.min(100, Math.max(0, predictedScore))}%`;
    scoreProgressEl.className = "progress-fill " + (predictedScore >= 60 ? "pass" : "fail");
  }

  // If using SVG circular gauge
  if (scoreGaugeCircle) {
    const radius = 54;
    const circumference = 2 * Math.PI * radius;
    const offset = circumference - (predictedScore / 100) * circumference;
    scoreGaugeCircle.style.strokeDashoffset = offset;
    scoreGaugeCircle.style.stroke = predictedScore >= 60 ? "#10b981" : "#ef4444";
  }

  // 5. Update Pass / Risk Badge and Probability
  const statusBadge = document.getElementById("status-badge");
  const probabilityNumberEl = document.getElementById("probability-number");
  const probabilityBarEl = document.getElementById("probability-bar");

  if (statusBadge) {
    if (isLikelyToPass) {
      statusBadge.textContent = "Likely to Pass";
      statusBadge.className = "badge badge-pass";
    } else {
      statusBadge.textContent = "At Risk";
      statusBadge.className = "badge badge-risk";
    }
  }

  if (probabilityNumberEl) {
    probabilityNumberEl.textContent = `${(passProbability * 100).toFixed(1)}%`;
  }

  if (probabilityBarEl) {
    probabilityBarEl.style.width = `${(passProbability * 100).toFixed(1)}%`;
    probabilityBarEl.className = "prob-fill " + (isLikelyToPass ? "pass" : "risk");
  }

  // 6. Update Test Preparation Impact Chart (holding other inputs fixed)
  const canvas = document.getElementById("prep-impact-chart");
  if (canvas) {
    // Clone selection for "none" vs "completed"
    const selNone = { ...selection, test_preparation_course: "none" };
    const selComp = { ...selection, test_preparation_course: "completed" };

    const encNone = encodeSelection(selNone, modelParams);
    const scaledNone = scaleFeatures(encNone, modelParams.scaler);
    const scoreNone = predictScore(scaledNone, modelParams.linear_regression);

    const encComp = encodeSelection(selComp, modelParams);
    const scaledComp = scaleFeatures(encComp, modelParams.scaler);
    const scoreComp = predictScore(scaledComp, modelParams.linear_regression);

    renderImpactChart(canvas, scoreNone, scoreComp, modelParams.linear_regression.pass_threshold || 60);
  }
}

/**
 * Resets all select dropdowns to their initial values
 */
function resetForm() {
  if (!modelParams) return;
  const categories = Object.keys(modelParams.categorical_options);

  categories.forEach((catKey) => {
    const select = document.getElementById(`select-${catKey}`);
    const options = modelParams.categorical_options[catKey];
    if (select && options && options.length > 0) {
      select.value = options[0];
    }
  });

  updatePredictions();
}

/**
 * Attaches change event listeners to all select controls and reset button
 */
function attachEventListeners() {
  const form = document.getElementById("demographics-form");
  if (form) {
    form.addEventListener("change", (e) => {
      if (e.target.tagName === "SELECT") {
        updatePredictions();
      }
    });
  }

  const resetBtn = document.getElementById("btn-reset");
  if (resetBtn) {
    resetBtn.addEventListener("click", resetForm);
  }

  window.addEventListener("resize", () => {
    updatePredictions();
  });
}
