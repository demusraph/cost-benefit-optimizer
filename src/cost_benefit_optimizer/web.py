from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from .models import OptimizationRequest
from .engine import OptimizerEngine

app = FastAPI(
    title="CostBenefit Optimizer API",
    description="Enterprise Financial Threshold Calibrator and Payoff Engine",
    version="1.0.0"
)

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CostBenefit Optimizer | Financial Threshold Calibrator</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Plus Jakarta Sans', sans-serif; background-color: #09090B; color: #FFFFFF; }
        .font-mono { font-family: 'JetBrains Mono', monospace; }
    </style>
</head>
<body class="min-h-screen bg-[#09090B] text-white p-6 sm:p-12">
    <div class="max-w-5xl mx-auto space-y-8">
        <!-- Header -->
        <header class="border-b border-[#27272A] pb-6 flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div>
                <div class="text-xs uppercase tracking-widest text-emerald-400 font-mono mb-1">Decision Intelligence / FinTech</div>
                <h1 class="text-3xl font-bold tracking-tight">CostBenefit Optimizer</h1>
                <p class="text-zinc-400 text-sm mt-1">Calibrate classification thresholds for maximum Expected Monetary Value (EMV).</p>
            </div>
            <div class="flex items-center gap-3">
                <span class="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-emerald-950 text-emerald-400 border border-emerald-800">
                    Engine Active v1.0.0
                </span>
            </div>
        </header>

        <!-- KPI Metrics Grid -->
        <div class="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div class="bg-[#121214] border border-[#27272A] rounded-xl p-6">
                <span class="text-xs uppercase font-mono tracking-wider text-zinc-500">Optimal Threshold</span>
                <div id="res-threshold" class="text-3xl font-bold font-mono text-emerald-400 mt-2">0.3800</div>
                <span class="text-xs text-zinc-400 mt-1 block">vs. Standard 0.50 default</span>
            </div>
            <div class="bg-[#121214] border border-[#27272A] rounded-xl p-6">
                <span class="text-xs uppercase font-mono tracking-wider text-zinc-500">Max Projected EMV</span>
                <div id="res-emv" class="text-3xl font-bold font-mono text-white mt-2">$248,500.00</div>
                <span class="text-xs text-zinc-400 mt-1 block">Expected business payoff</span>
            </div>
            <div class="bg-[#121214] border border-[#27272A] rounded-xl p-6">
                <span class="text-xs uppercase font-mono tracking-wider text-zinc-500">Projected Profit Lift</span>
                <div id="res-lift" class="text-3xl font-bold font-mono text-blue-400 mt-2">+28.4%</div>
                <span class="text-xs text-zinc-400 mt-1 block">Net monetary gain over 0.5</span>
            </div>
        </div>

        <!-- Interactive Cost-Benefit Form -->
        <div class="bg-[#121214] border border-[#27272A] rounded-xl p-6 sm:p-8 space-y-6">
            <h2 class="text-lg font-semibold text-white">Financial Matrix Configuration ($ USD)</h2>
            <div class="grid grid-cols-2 md:grid-cols-4 gap-4 font-mono text-sm">
                <div>
                    <label class="block text-xs uppercase text-zinc-400 mb-1">True Positive ($ TP)</label>
                    <input id="input-tp" type="number" value="500" class="w-full bg-[#18181B] border border-[#27272A] rounded-lg px-3 py-2 text-white focus:outline-none focus:border-emerald-500">
                </div>
                <div>
                    <label class="block text-xs uppercase text-zinc-400 mb-1">False Positive ($ FP)</label>
                    <input id="input-fp" type="number" value="-200" class="w-full bg-[#18181B] border border-[#27272A] rounded-lg px-3 py-2 text-white focus:outline-none focus:border-emerald-500">
                </div>
                <div>
                    <label class="block text-xs uppercase text-zinc-400 mb-1">True Negative ($ TN)</label>
                    <input id="input-tn" type="number" value="0" class="w-full bg-[#18181B] border border-[#27272A] rounded-lg px-3 py-2 text-white focus:outline-none focus:border-emerald-500">
                </div>
                <div>
                    <label class="block text-xs uppercase text-zinc-400 mb-1">False Negative ($ FN)</label>
                    <input id="input-fn" type="number" value="-1000" class="w-full bg-[#18181B] border border-[#27272A] rounded-lg px-3 py-2 text-white focus:outline-none focus:border-emerald-500">
                </div>
            </div>
            <div class="flex justify-end pt-2">
                <button onclick="runSimulation()" class="bg-white text-black hover:bg-zinc-200 font-semibold px-6 py-2.5 rounded-lg text-sm transition">
                    Re-Optimize Threshold
                </button>
            </div>
        </div>
    </div>

    <script>
        function runSimulation() {
            const tp = parseFloat(document.getElementById('input-tp').value);
            const fp = parseFloat(document.getElementById('input-fp').value);
            const tn = parseFloat(document.getElementById('input-tn').value);
            const fn = parseFloat(document.getElementById('input-fn').value);

            // Synthetic demo population for instant client-side preview
            const synthetic_probs = [0.12, 0.25, 0.38, 0.44, 0.52, 0.67, 0.73, 0.81, 0.89, 0.95];
            const synthetic_labels = [0, 0, 0, 1, 0, 1, 1, 1, 1, 1];

            fetch('/optimize', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    probabilities: synthetic_probs,
                    labels: synthetic_labels,
                    costs: {tp: tp, fp: fp, tn: tn, fn: fn}
                })
            })
            .then(res => res.json())
            .then(data => {
                document.getElementById('res-threshold').innerText = data.optimal_threshold.toFixed(4);
                document.getElementById('res-emv').innerText = '$' + data.max_emv.toLocaleString('en-US', {minimumFractionDigits: 2});
                document.getElementById('res-lift').innerText = (data.profit_lift_pct >= 0 ? '+' : '') + data.profit_lift_pct + '%';
            })
            .catch(err => console.error(err));
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_TEMPLATE

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "cost-benefit-optimizer", "version": "1.0.0"}

@app.post("/optimize")
async def optimize(req: OptimizationRequest):
    try:
        return OptimizerEngine.run_optimization(req)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/sensitivity")
async def sensitivity(req: OptimizationRequest, variation: float = 0.2):
    try:
        return OptimizerEngine.sensitivity_analysis(req, variation_pct=variation)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))