FROM python:3.13

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
  libx11-xcb1 \
  libxkbcommon-x11-0 \
  libxcb1 \
  libxcb-cursor0 \
  libxcb-keysyms1 \
  libxcb-randr0 \
  libxcb-render-util0 \
  libxcb-render0 \
  libxcb-shape0 \
  libxcb-shm0 \
  libxcb-sync1 \
  libxcb-util1 \
  libxcb-xfixes0 \
  libxcb-xinerama0 \
  libxcb-icccm4 \
  libxcb-image0 \
  libgl1 \
  libglx0 \
  libglvnd0 \
  libegl1 \
  libopengl0 \
  libdbus-1-3 \
  libxkbcommon0 \
  libfontconfig1 \
  && rm -rf /var/lib/apt/lists/*

COPY uv.lock pyproject.toml ./ 

ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

RUN uv sync --frozen --no-dev

COPY . . 
ENV PATH="/app/.venv/bin:$PATH"

CMD ["uv", "run", "python3", "main.py"]
