FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIXI_NO_ANALYTICS=1 \
    PATH=/root/.pixi/bin:$PATH

RUN sed -i 's/^# deb /deb /' /etc/apt/sources.list \
    && apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    fontconfig \
    poppler-utils \
    imagemagick \
    texlive-xetex \
    texlive-latex-recommended \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-fonts-extra \
    texlive-pictures \
    texlive-lang-chinese \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    libsndfile1 \
    build-essential \
    pkg-config \
    cmake \
    unzip \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL -o /tmp/fira.zip https://github.com/mozilla/Fira/archive/refs/tags/4.202.zip \
    && unzip -q /tmp/fira.zip -d /tmp/fira \
    && mkdir -p /usr/local/share/fonts/Fira \
    && cp /tmp/fira/Fira-4.202/otf/FiraSans-*.otf /usr/local/share/fonts/Fira/ \
    && cp /tmp/fira/Fira-4.202/otf/FiraMono-*.otf /usr/local/share/fonts/Fira/ \
    && curl -fsSL -o /tmp/SourceHanSansSC.zip \
        https://github.com/adobe-fonts/source-han-sans/releases/download/2.004R/SourceHanSansSC.zip \
    && unzip -q /tmp/SourceHanSansSC.zip -d /tmp/sourcehansanssc \
    && mkdir -p /usr/local/share/fonts/SourceHanSansSC \
    && cp /tmp/sourcehansanssc/OTF/SimplifiedChinese/SourceHanSansSC-*.otf \
        /usr/local/share/fonts/SourceHanSansSC/ \
    && fc-cache -f \
    && rm -rf /tmp/fira /tmp/fira.zip /tmp/sourcehansanssc /tmp/SourceHanSansSC.zip

RUN curl -fsSL https://pixi.sh/install.sh | bash

WORKDIR /app

COPY pixi.toml pixi.lock ./
RUN pixi install --frozen

COPY . .

RUN mkdir -p /app/third_party \
    && git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git /app/third_party/CosyVoice

RUN mkdir -p /app/input /app/output /app/cache

ENTRYPOINT ["pixi", "run", "python", "main.py"]
