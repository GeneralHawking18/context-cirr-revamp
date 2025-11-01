FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

WORKDIR /app

# ==========================================================
# 1️⃣ Cài hệ thống (và bật cache cho apt)
# ==========================================================
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && DEBIAN_FRONTEND=noninteractive \
    apt-get install -y \
      python3.10 \
      python3-pip \
      git \
      libgl1-mesa-glx \
      libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ==========================================================
# 2️⃣ Cài Python dependencies (cache pip)
# ==========================================================
# Copy riêng file requirements để cache layer này
COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && pip install -r requirements.txt

# ==========================================================
# 3️⃣ Cài thêm tool Python (chạy một lần duy nhất)
# ==========================================================
RUN curl -sSL https://install.python-poetry.org | python3 - --preview && \
    pip3 install --upgrade requests

# ==========================================================
# 4️⃣ Copy code sau cùng (để không phá cache pip)
# ==========================================================
COPY . .

# ==========================================================
# 5️⃣ Cấu hình mặc định
# ==========================================================
RUN ln -fs /usr/bin/python3 /usr/bin/python
CMD ["python", "--version"]
