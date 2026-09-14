# GradeCompass - API & Schema Reference 📐

This document provides technical schemas, JSON definitions, and API specifications for GradeCompass's offline Python modules and client-side JavaScript inference engine.

---

## 1. `model_params.json` Schema Specification

The client-side inference engine relies on a single portable JSON file located at `web/assets/model_params.json`.

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "GradeCompassModelParams",
  "type": "object",
  "required": [
    "feature_order",
    "categorical_options",
    "scaler",
    "linear_regression",
    "logistic_regression",
    "metadata"
  ],
  "properties": {
    "feature_order": {
      "type": "array",
      "items": { "type": "string" },
      "minItems": 17,
      "maxItems": 17,
      "description": "Fixed 17-feature column ordering matching training/preprocessing.py"
    },
    "categorical_options": {
      "type": "object",
      "required": [
        "gender",
        "race_ethnicity",
        "parental_education",
        "lunch",
        "test_preparation_course"
      ],
      "properties": {
        "gender": { "type": "array", "items": { "type": "string" } },
        "race_ethnicity": { "type": "array", "items": { "type": "string" } },
        "parental_education": { "type": "array", "items": { "type": "string" } },
        "lunch": { "type": "array", "items": { "type": "string" } },
        "test_preparation_course": { "type": "array", "items": { "type": "string" } }
      }
    },
    "scaler": {
      "type": "object",
      "required": ["mean", "std"],
      "properties": {
        "mean": { "type": "array", "items": { "type": "number" }, "minItems": 17, "maxItems": 17 },
        "std": { "type": "array", "items": { "type": "number" }, "minItems": 17, "maxItems": 17 }
      }
    },
    "linear_regression": {
      "type": "object",
      "required": ["weights", "bias", "pass_threshold"],
      "properties": {
        "weights": { "type": "array", "items": { "type": "number" }, "minItems": 17, "maxItems": 17 },
        "bias": { "type": "number" },
        "pass_threshold": { "type": "integer" }
      }
    },
    "logistic_regression": {
      "type": "object",
      "required": ["weights", "bias"],
      "properties": {
        "weights": { "type": "array", "items": { "type": "number" }, "minItems": 17, "maxItems": 17 },
        "bias": { "type": "number" }
      }
    },
    "metadata": {
      "type": "object",
      "required": [
        "trained_on_rows",
        "linear_r2",
        "linear_mae",
        "logistic_accuracy",
        "logistic_precision",
        "logistic_recall",
        "dataset_source"
      ],
      "properties": {
        "trained_on_rows": { "type": "integer" },
        "linear_r2": { "type": "number" },
        "linear_mae": { "type": "number" },
        "logistic_accuracy": { "type": "number" },
        "logistic_precision": { "type": "number" },
        "logistic_recall": { "type": "number" },
        "dataset_source": { "type": "string" }
      }
    }
  }
}
```

---

## 2. JavaScript Inference API (`web/js/model.js`)

### `loadModelParams(path?: string): Promise<Object>`
Fetches and parses the serialized parameter object.
- **Parameters**: `path` (default: `"./assets/model_params.json"`).
- **Returns**: Parsed JSON object conforming to the schema above.
- **Throws**: `Error` if the HTTP response is not `200 OK`.

### `getFeatureName(categoryKey: string, value: string): string`
Maps a category key and raw option string into the corresponding column name from `feature_order`.
- **Example**: `getFeatureName("parental_education", "bachelor's degree")` &rarr; `"parent_edu_bachelors_degree"`.

### `encodeSelection(selection: Object, params: Object): number[]`
Encodes a 5-attribute student selection object into a 17-length binary vector.
- **Parameters**:
  - `selection`: `{ gender, race_ethnicity, parental_education, lunch, test_preparation_course }`
  - `params`: Parsed `model_params` object.
- **Returns**: `number[]` array of length 17 containing exactly five `1`s and twelve `0`s.

### `scaleFeatures(rawFeatures: number[], scaler: Object): number[]`
Applies standard standardization: $z_i = (x_i - \mu_i) / \sigma_i$.
- **Parameters**:
  - `rawFeatures`: 17-element binary vector.
  - `scaler`: `{ mean: number[], std: number[] }`
- **Returns**: Standardized feature vector.

### `predictScore(scaledFeatures: number[], linearParams: Object): number`
Calculates expected continuous score clamped to $[0, 100]$.
- **Parameters**:
  - `scaledFeatures`: 17-element standardized vector.
  - `linearParams`: `{ weights: number[], bias: number }`
- **Returns**: Expected score as a floating-point number clamped in $[0, 100]$.

### `predictPassProbability(scaledFeatures: number[], logisticParams: Object): number`
Calculates pass probability using standard logistic sigmoid.
- **Parameters**:
  - `scaledFeatures`: 17-element standardized vector.
  - `logisticParams`: `{ weights: number[], bias: number }`
- **Returns**: Probability in $[0.0, 1.0]$.

---

## 3. Python Module API (`training/`)

### `preprocessing.load_and_prepare(csv_path?: str)`
- **Loads**: Kaggle dataset and validates absence of null values.
- **Computes**: `average_score` (continuous target) and `passed` (binary classification target).
- **Encodes**: Feature matrix $X$ ($1,000 \times 17$) adhering strictly to `FEATURE_ORDER`.
- **Returns**: `(X: pd.DataFrame, y_score: pd.Series, y_passed: pd.Series)`

### `preprocessing.get_scaler(X_train: pd.DataFrame)`
- **Fits**: `sklearn.preprocessing.StandardScaler` on training split.
- **Returns**: Fitted `StandardScaler` instance.

### `preprocessing.encode_profile(profile: dict) -> list`
- **Encodes**: Single dictionary profile into the contractual 17-element binary list.
- **Returns**: `list[int]` of length 17.

### `merge_params.merge_model_parameters()`
- **Loads**: `training/linear_params.json` and `training/logistic_params.json`.
- **Validates**: All 17 dimension lengths, non-null values, and schema conformity.
- **Outputs**: Writes `web/assets/model_params.json` and copies `web/assets/model_metrics.md`.
