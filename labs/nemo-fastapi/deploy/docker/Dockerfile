FROM pytorch/pytorch:2.6.0-cuda12.8-cudnn9-runtime

WORKDIR /app

RUN pip install --no-cache-dir \
    "nemo-toolkit[asr]>=2.5,<2.7" \
    && pip cache purge

COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
COPY configs/ ./configs/

RUN pip install --no-cache-dir --no-deps -e . \
    && pip cache purge

EXPOSE 8000

ENTRYPOINT ["highperfasr"]
CMD ["serve", "--config", "/app/configs/serving-stream.yaml"]
