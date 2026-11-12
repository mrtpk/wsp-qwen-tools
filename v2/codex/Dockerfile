# Codex CLI in a container. The container is the sandbox for the model's commands.
FROM node:22-bookworm
RUN npm install -g @openai/codex@0.167.0 && apt-get update && apt-get install -y --no-install-recommends python3 && rm -rf /var/lib/apt/lists/*
