# make up       start servers
# make down     stop
# make health   check endpoints
# make smoke    quick test
# make logs     tail output

.PHONY: up down health smoke logs prefetch

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

health:
	@echo "Batch  :8000 — $$(curl -sf http://localhost:8000/health 2>/dev/null || echo 'not running')"
	@echo "Stream :8001 — $$(curl -sf http://localhost:8001/health 2>/dev/null || echo 'not running')"

smoke:
	@python3 -c "import struct,wave;f=wave.open('/tmp/_smoke.wav','wb');f.setnchannels(1);f.setsampwidth(2);f.setframerate(16000);f.writeframes(struct.pack('<'+'h'*16000,*([1000]*16000)));f.close()"
	@curl -sf -F "file=@/tmp/_smoke.wav" http://localhost:8000/v1/transcriptions && echo "" || echo "batch not available"
	@rm -f /tmp/_smoke.wav

prefetch:
	docker compose run --rm stream python -c "from nemo.collections.asr.models import ASRModel; ASRModel.from_pretrained('nvidia/canary-1b-flash')"
