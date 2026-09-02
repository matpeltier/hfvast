FROM ghcr.io/ggml-org/llama.cpp:server-cuda
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends python3 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY gateway.py watchdog_container.py /opt/hfvast/
ENV HFVAST_ROOT=/opt/hfvast
