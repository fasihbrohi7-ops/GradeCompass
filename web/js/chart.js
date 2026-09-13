/**
 * GradeCompass Test Prep Course Impact Chart
 * Author: Abdul Hayy
 *
 * Renders a crisp, responsive HTML5 canvas bar chart comparing predicted
 * average exam scores between test_preparation_course = 'none' vs 'completed',
 * holding all other demographic attributes fixed.
 */

/**
 * Draws the comparative bar chart on the specified canvas element.
 * @param {HTMLCanvasElement} canvas
 * @param {number} scoreNone - Predicted average score without test prep
 * @param {number} scoreCompleted - Predicted average score with test prep
 * @param {number} threshold - Pass threshold (default 60)
 */
function renderImpactChart(canvas, scoreNone, scoreCompleted, threshold = 60) {
  if (!canvas) return;

  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  // Handle High-DPI screens (Retina)
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const width = rect.width || 320;
  const height = rect.height || 220;

  canvas.width = width * dpr;
  canvas.height = height * dpr;
  ctx.scale(dpr, dpr);

  // Clear canvas
  ctx.clearRect(0, 0, width, height);

  const padding = { top: 35, right: 25, bottom: 45, left: 45 };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;

  // Background grid lines (0, 20, 40, 60, 80, 100)
  ctx.strokeStyle = "#e2e8f0";
  ctx.lineWidth = 1;
  ctx.fillStyle = "#94a3b8";
  ctx.font = "11px system-ui, -apple-system, sans-serif";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";

  const gridSteps = [0, 20, 40, 60, 80, 100];
  gridSteps.forEach((val) => {
    const y = padding.top + chartHeight - (val / 100) * chartHeight;
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(width - padding.right, y);
    ctx.stroke();

    ctx.fillText(`${val}`, padding.left - 8, y);
  });

  // Draw Pass Threshold Reference Line at 60
  const thresholdY = padding.top + chartHeight - (threshold / 100) * chartHeight;
  ctx.save();
  ctx.setLineDash([4, 4]);
  ctx.strokeStyle = "#ef4444";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(padding.left, thresholdY);
  ctx.lineTo(width - padding.right, thresholdY);
  ctx.stroke();
  ctx.restore();

  // Threshold label
  ctx.fillStyle = "#ef4444";
  ctx.font = "bold 10px system-ui, -apple-system, sans-serif";
  ctx.textAlign = "left";
  ctx.fillText(`Pass Mark (${threshold})`, width - padding.right - 85, thresholdY - 8);

  // Bars configuration
  const barWidth = Math.min(65, chartWidth / 4);
  const col1X = padding.left + chartWidth * 0.25 - barWidth / 2;
  const col2X = padding.left + chartWidth * 0.75 - barWidth / 2;

  const barNoneHeight = (Math.max(0, Math.min(100, scoreNone)) / 100) * chartHeight;
  const barCompHeight = (Math.max(0, Math.min(100, scoreCompleted)) / 100) * chartHeight;

  const barNoneY = padding.top + chartHeight - barNoneHeight;
  const barCompY = padding.top + chartHeight - barCompHeight;

  // Function to draw rounded rect bar
  function drawRoundedBar(x, y, w, h, radius, fillStyle) {
    if (h <= 0) return;
    const r = Math.min(radius, h / 2, w / 2);
    ctx.save();
    ctx.fillStyle = fillStyle;
    ctx.beginPath();
    ctx.moveTo(x, y + h);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h);
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }

  // Draw Bar 1: None
  const colorNone = scoreNone >= threshold ? "#3b82f6" : "#f59e0b";
  drawRoundedBar(col1X, barNoneY, barWidth, barNoneHeight, 6, colorNone);

  // Draw Bar 2: Completed
  const colorComp = scoreCompleted >= threshold ? "#10b981" : "#f59e0b";
  drawRoundedBar(col2X, barCompY, barWidth, barCompHeight, 6, colorComp);

  // Score value labels on top of bars
  ctx.font = "bold 13px system-ui, -apple-system, sans-serif";
  ctx.textAlign = "center";

  ctx.fillStyle = "#1e293b";
  ctx.fillText(scoreNone.toFixed(1), col1X + barWidth / 2, barNoneY - 8);

  ctx.fillStyle = "#0f766e";
  ctx.fillText(scoreCompleted.toFixed(1), col2X + barWidth / 2, barCompY - 8);

  // Category labels below bars
  ctx.font = "12px system-ui, -apple-system, sans-serif";
  ctx.fillStyle = "#475569";
  ctx.fillText("No Course", col1X + barWidth / 2, height - padding.bottom + 18);
  ctx.fillText("Completed", col2X + barWidth / 2, height - padding.bottom + 18);

  // Delta callout
  const delta = scoreCompleted - scoreNone;
  ctx.font = "italic 11px system-ui, -apple-system, sans-serif";
  ctx.fillStyle = delta >= 0 ? "#059669" : "#dc2626";
  const deltaText = delta >= 0 ? `+${delta.toFixed(1)} pts advantage` : `${delta.toFixed(1)} pts`;
  ctx.fillText(deltaText, col2X + barWidth / 2, height - padding.bottom + 32);
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { renderImpactChart };
}
