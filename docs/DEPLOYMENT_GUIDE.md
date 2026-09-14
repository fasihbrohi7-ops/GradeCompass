# GradeCompass - Production Deployment Guide 🚀

This document outlines deployment architectures, configuration settings, troubleshooting steps, and continuous deployment workflows for **GradeCompass**.

---

## 1. Vercel Deployment (Recommended)

GradeCompass is architected as a pure static web application. It requires zero serverless functions, zero backend nodes, and zero build commands at runtime.

### 1.1 One-Click Git Integration (Production)

1. **Push Repository**: Ensure all local commits on `main` are pushed to GitHub:
   ```bash
   git push origin main
   ```
2. **Access Vercel Dashboard**:
   - Navigate to [https://vercel.com/dashboard](https://vercel.com/dashboard).
   - Click **Add New... &rarr; Project**.
3. **Import Project**:
   - Select `fasihbrohi7-ops/GradeCompass`.
4. **Configure Project Settings**:
   - **Framework Preset**: Choose **`Other`** (Static Site).
   - **Root Directory**: Leave as `./` (the root directory).
   - *Note*: Because [`vercel.json`](../vercel.json) specifies `"outputDirectory": "web"`, Vercel will automatically serve the static bundle from `./web`.
   - **Build Command**: Leave empty / disabled.
   - **Output Directory**: Leave empty (overridden by `vercel.json`).
5. **Click Deploy**:
   - Deployment typically completes in under 20 seconds.

---

### 1.2 `vercel.json` Specification

The project includes a root [`vercel.json`](../vercel.json) file:

```json
{
  "outputDirectory": "web"
}
```

#### Optional Custom Header Rules
For enhanced security and caching policies in high-traffic deployments, `vercel.json` can be extended with HTTP headers:

```json
{
  "outputDirectory": "web",
  "cleanUrls": true,
  "headers": [
    {
      "source": "/assets/(.*)",
      "headers": [
        {
          "key": "Cache-Control",
          "value": "public, max-age=31536000, immutable"
        }
      ]
    }
  ]
}
```

---

## 2. Local Hosting Options

Because modern browsers enforce strict CORS policies on the `file://` protocol when calling `fetch('./assets/model_params.json')`, GradeCompass must be served via a local HTTP server.

### Option A: Python Built-In HTTP Server
```bash
# Navigate to repository root and serve the web/ directory
python -m http.server 8000 --directory web
```
Then visit `http://localhost:8000`.

### Option B: Node.js `serve`
```bash
npx serve web -p 3000
```
Then visit `http://localhost:3000`.

### Option C: VS Code Live Server
1. Install the **Live Server** extension by Ritwick Dey.
2. Right-click `web/index.html` in the file explorer.
3. Select **"Open with Live Server"**.

---

## 3. Post-Deployment Verification Checklist

After deploying to production:

- [ ] **HTTP 200 on Assets**: Open browser Developer Tools &rarr; **Network** tab and verify that `/assets/model_params.json` returns HTTP `200 OK`.
- [ ] **Case Sensitivity Check**: Verify all URLs use lowercase paths (`css/style.css`, `js/model.js`, `assets/model_params.json`). Linux-based edge CDN servers enforce strict case sensitivity.
- [ ] **Live Prediction Test**: Change each of the 5 dropdown selects; verify the circular gauge animates, the pass badge toggles, and the canvas sensitivity chart re-renders immediately.
- [ ] **Navigation Test**: Click **"About the Model"** and ensure navigation between `index.html` and `about.html` works without 404s.
- [ ] **Lighthouse Performance Audit**: Run a Chrome DevTools Lighthouse audit on mobile and desktop:
  - Performance: $\ge 95$
  - Accessibility: $\ge 95$
  - Best Practices: $\ge 95$
  - SEO: $\ge 95$

---

## 4. Retraining & Continuous Deployment (CI/CD)

Whenever dataset records are modified or models are re-trained:

```bash
# 1. Retrain Linear Regression
python training/train_linear.py

# 2. Retrain Logistic Regression
python training/train_logistic.py

# 3. Merge Parameters into web/assets/
python training/merge_params.py

# 4. Run Verification Suite
python tests/test_inference.py

# 5. Commit & Push to Main
git add .
git commit -m "feat(models): update model parameters and evaluation metrics"
git push origin main
```
Vercel's GitHub integration will automatically detect the push and trigger an instantaneous zero-downtime production deployment.
