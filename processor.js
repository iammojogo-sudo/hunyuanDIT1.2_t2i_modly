"use strict";
const path = require("path");
const { spawn } = require("child_process");
const fs   = require("fs");
const os   = require("os");

const processor = async (input, params, context) => {
    const prompt = (input.text || "").trim();
    if (!prompt) throw new Error("hunyuandit-t2i: no prompt text received from connected Text node.");

    const extDir = __dirname;
    const isWin  = process.platform === "win32";
    const pythonExe = isWin
        ? path.join(extDir, "venv", "Scripts", "python.exe")
        : path.join(extDir, "venv", "bin", "python");

    if (!fs.existsSync(pythonExe))
        throw new Error(
            `hunyuandit-t2i: venv not found at ${pythonExe}. ` +
            "Please Repair the extension."
        );

    const workerScript = path.join(extDir, "generator.py");

    const modelsDir = process.env.MODELS_DIR ||
        path.join(os.homedir(), ".modly", "models");

    const workspaceDir = context.workspaceDir ||
        path.join(os.homedir(), ".modly", "workspace");

    const paramsJson = JSON.stringify(params || {});

    context.log(`HunyuanDiT t2i — prompt: "${prompt}"`);
    context.log(`Models dir: ${modelsDir}`);
    context.progress(2, "Starting generation worker...");

    return new Promise((resolve, reject) => {
        const worker = spawn(pythonExe, [
            workerScript,
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
                    if (msg.type === "progress") {
                        context.progress(msg.pct, msg.step || "");
                    } else if (msg.type === "log") {
                        context.log(msg.message || "");
                    } else if (msg.type === "done") {
                        outputPath = msg.output_path;
                    } else if (msg.type === "error") {
                        reject(new Error(msg.message || "Worker error"));
                    }
                } catch (_) {
                    context.log(`[worker] ${trimmed}`);
                }
            }
        });

        worker.stderr.on("data", (chunk) => {
            const text = chunk.toString().trim();
            if (text) context.log(`[stderr] ${text}`);
        });

        worker.on("error", (err) => {
            reject(new Error(`Failed to start worker: ${err.message}`));
        });

        worker.on("close", (code) => {
            if (outputPath) {
                context.log(`Generation complete: ${outputPath}`);
                resolve({ filePath: outputPath });
            } else if (code !== 0) {
                reject(new Error(`Worker exited with code ${code}. Check logs.`));
            } else {
                reject(new Error("Worker finished but returned no output path."));
            }
        });
    });
};

module.exports = processor;
