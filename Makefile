# make up       start servers
# make down     stop
# make health   check endpoints
# make smoke    quick test
# make logs     tail output

.PHONY: up down health smoke logs

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
	@python3 -c "import json,urllib.request; data=json.load(urllib.request.urlopen('http://localhost:8001/health', timeout=5)); assert data.get('ready'), data; assert data.get('stream_model'), data; print('stream health ok')"
	@curl -sf http://localhost:8001/metrics/prometheus | grep -q '^highperfasr_active_streams ' && echo "stream metrics ok"
