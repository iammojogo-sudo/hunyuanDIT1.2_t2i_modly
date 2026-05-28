"use strict";
const path    = require("path");
const { spawn } = require("child_process");
const fs      = require("fs");
const os      = require("os");

// ─── Helpers ──────────────────────────────────────────────────────────────────

function run(exe, args, opts = {}) {
    return new Promise((resolve, reject) => {
        const proc = spawn(exe, args, { ...opts, stdio: ["ignore", "pipe", "pipe"] });
        let out = "", err = "";
        proc.stdout.on("data", (d) => { out += d.toString() });
        proc.stderr.on("data", (d) => { err += d.toString() });
        proc.on("error", reject);
        proc.on("close", (code) => {
            if (code === 0) resolve(out.trim());
            else reject(new Error(err.trim() || `exited with code ${code}`));
        });
    });
}

// ─── Processor ────────────────────────────────────────────────────────────────

const processor = async (input, params, context) => {
    const prompt = (input.text || "").trim();
    if (!prompt) throw new Error("hunyuandit-t2i: no prompt received. Connect a Text node.");

    const extDir = __dirname;
    const isWin  = process.platform === "win32";

    const pythonExe = isWin
        ? path.join(extDir, "venv", "Scripts", "python.exe")
        : path.join(extDir, "venv", "bin", "python");

    // ── First-run self-setup ──────────────────────────────────────────────────
    if (!fs.existsSync(pythonExe)) {
        context.progress(0, "First-time setup — installing dependencies...");
        context.log("venv not found — running setup automatically.");

        // Modly's embedded Python lives next to its resources
        const modlyPython = isWin
            ? path.join(process.resourcesPath, "python-embed", "python.exe")
            : path.join(process.resourcesPath, "python-embed", "python");

        if (!fs.existsSync(modlyPython))
            throw new Error(
                `Cannot locate Modly's Python at ${modlyPython}. ` +
                "Please use the Repair button."
            );

        // Detect GPU SM using Modly's Python (it already has torch)
        let gpuSm = 0;
        try {
            const sm = await run(modlyPython, ["-c",
                "import torch; c=torch.cuda.get_device_capability(0); print(c[0]*10+c[1])"
            ]);
            gpuSm = parseInt(sm) || 0;
        } catch (_) {
            gpuSm = 0;
        }

        context.log(`GPU SM detected: ${gpuSm}`);
        context.progress(1, `Installing packages (GPU SM ${gpuSm}) — this takes a few minutes...`);

        const setupScript = path.join(extDir, "setup.py");
        try {
            const setupLog = await run(modlyPython, [setupScript, modlyPython, extDir, String(gpuSm)]);
            context.log(setupLog);
        } catch (e) {
            throw new Error(`Setup failed: ${e.message}`);
        }

        context.log("Setup complete.");

        if (!fs.existsSync(pythonExe))
            throw new Error("Setup ran but venv still not found. Check logs.");
    }

    // ── Generate ──────────────────────────────────────────────────────────────
    const modelsDir = process.env.MODELS_DIR ||
        path.join(os.homedir(), ".modly", "models");

    const workspaceDir = context.workspaceDir ||
        path.join(os.homedir(), ".modly", "workspace");

    const paramsJson = JSON.stringify(params || {});

    context.log(`Prompt: "${prompt}"`);
    context.progress(5, "Starting generation worker...");

    return new Promise((resolve, reject) => {
        const worker = spawn(pythonExe, [
            path.join(extDir, "generator.py"),
            prompt,
            paramsJson,
            modelsDir,
            workspaceDir,
        ], {
            env: {
                ...process.env,
                MODELS_DIR:    modelsDir,
                WORKSPACE_DIR: workspaceDir,
                EXTENSION_DIR: extDir,
            },
        });

        let outputPath = null;
        let lineBuf    = "";

        worker.stdout.on("data", (chunk) => {
            lineBuf += chunk.toString();
            const lines = lineBuf.split("\n");
            lineBuf = lines.pop();
            for (const line of lines) {
                const trimmed = line.trim();
                if (!trimmed) continue;
                try {
                    const msg = JSON.parse(trimmed);
                    if      (msg.type === "progress") context.progress(msg.pct, msg.step || "");
                    else if (msg.type === "log")      context.log(msg.message || "");
                    else if (msg.type === "done")     outputPath = msg.output_path;
                    else if (msg.type === "error")    reject(new Error(msg.message || "Worker error"));
                } catch (_) {
                    context.log(`[worker] ${trimmed}`);
                }
            }
        });

        worker.stderr.on("data", (chunk) => {
            const t = chunk.toString().trim();
            if (t) context.log(`[stderr] ${t}`);
        });

        worker.on("error", (err) => reject(new Error(`Failed to start worker: ${err.message}`)));

        worker.on("close", (code) => {
            if (outputPath)  resolve({ filePath: outputPath });
            else if (code !== 0) reject(new Error(`Worker exited with code ${code}. Check logs.`));
            else reject(new Error("Worker finished but returned no output path."));
        });
    });
};

module.exports = processor;
