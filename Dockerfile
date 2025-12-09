FROM nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
