const fs = require('fs');
const path = require('path');
const model = require('../web/js/model.js');

const paramsPath = path.join(__dirname, '..', 'web', 'assets', 'model_params.json');
const params = JSON.parse(fs.readFileSync(paramsPath, 'utf8'));

const testProfiles = [
  {
    name: "Standard profile 1",
    input: {
      gender: "female",
      race_ethnicity: "group B",
      parental_education: "bachelor's degree",
      lunch: "standard",
      test_preparation_course: "none"
    }
  },
  {
    name: "Standard profile 2",
    input: {
      gender: "male",
      race_ethnicity: "group A",
      parental_education: "some high school",
      lunch: "free/reduced",
      test_preparation_course: "completed"
    }
  },
  {
    name: "High achievement profile",
    input: {
      gender: "female",
      race_ethnicity: "group E",
      parental_education: "master's degree",
      lunch: "standard",
      test_preparation_course: "completed"
    }
  },
  {
    name: "At-risk profile",
    input: {
      gender: "male",
      race_ethnicity: "group C",
      parental_education: "high school",
      lunch: "free/reduced",
      test_preparation_course: "none"
    }
  }
];

console.log("Running JavaScript Model Inference Tests...");
let allPassed = true;

testProfiles.forEach((test, idx) => {
  const encoded = model.encodeSelection(test.input, params);
  const activeCount = encoded.reduce((acc, val) => acc + val, 0);

  if (activeCount !== 5) {
    console.error(`[FAIL] ${test.name}: Expected 5 active features, got ${activeCount}`);
    allPassed = false;
    return;
  }

  const scaled = model.scaleFeatures(encoded, params.scaler);
  const score = model.predictScore(scaled, params.linear_regression);
  const prob = model.predictPassProbability(scaled, params.logistic_regression);

  if (score < 0 || score > 100 || isNaN(score)) {
    console.error(`[FAIL] ${test.name}: Score out of bounds or NaN: ${score}`);
    allPassed = false;
    return;
  }

  if (prob < 0 || prob > 1 || isNaN(prob)) {
    console.error(`[FAIL] ${test.name}: Probability out of bounds or NaN: ${prob}`);
    allPassed = false;
    return;
  }

  console.log(`[PASS] ${test.name}:`);
  console.log(`       Predicted Score: ${score.toFixed(2)}/100`);
  console.log(`       Pass Probability: ${(prob * 100).toFixed(2)}% (${prob >= 0.5 ? 'Likely to Pass' : 'At Risk'})`);
});

if (allPassed) {
  console.log("\n[SUCCESS] All JS inference engine tests passed perfectly!");
  process.exit(0);
} else {
  console.error("\n[FAILURE] One or more tests failed.");
  process.exit(1);
}
