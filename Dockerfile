# AMRShield Sentinel — Node API server + Python inference bridge in one container.
# Render (and most container hosts) build this directly; no other config needed
# beyond render.yaml's env vars.

FROM node:20-bookworm-slim

# Python 3 + pip for the LightGBM inference bridge (scripts/predict_cli.py).
# libgomp1 is required at runtime by LightGBM's compiled OpenMP backend —
# without it every prediction fails with "libgomp.so.1: cannot open shared object file".
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps first (cache layer — changes less often than app code)
COPY requirements.txt ./
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# Node deps
COPY package.json package-lock.json ./
RUN npm ci

# App source
COPY . .

# data/ is gitignored (see README "Edge Cases"); reconstruct the one file
# predict_cli.py actually needs at runtime from the committed model metadata.
RUN python3 scripts/ensure_feature_columns.py

RUN npm run build

ENV NODE_ENV=production
# LightGBM/numpy/scikit-learn each spin up their own BLAS/OpenMP thread pool by
# default; on a memory-constrained host (e.g. Render's free 512MB tier) that
# multiplies peak RSS per request for no real benefit at this request volume.
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
# Render injects PORT at runtime; server/index.ts already reads process.env.PORT.
EXPOSE 3000

CMD ["node", "dist/server/index.js"]
