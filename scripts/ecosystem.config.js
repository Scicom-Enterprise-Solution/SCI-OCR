const path = require("path");

const rootDir = path.resolve(__dirname, "..");

module.exports = {
  apps: [
    {
      name: "sci-ocr-api",
      cwd: rootDir,
      script: path.join(rootDir, "scripts", "run_api_dev_bg.sh"),
      interpreter: "/usr/bin/env",
      interpreter_args: "bash",
      exec_mode: "fork",
      autorestart: true,
      watch: false,
      max_restarts: 10,
      restart_delay: 3000,
      env: {
        PYTHONUNBUFFERED: "1",
        API_DEV_HOST: process.env.API_DEV_HOST || "0.0.0.0",
      },
    },
  ],
};