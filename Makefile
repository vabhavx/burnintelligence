.PHONY: install test benchmark start clean

install:
	./scripts/setup.sh

test:
	. venv/bin/activate && python3 -c "from intelligence.processing.selftest import run_selftest; r=run_selftest(); print(f'{r[\"passed\"]}/{r[\"checks\"]} ok'); exit(0 if r['ok'] else 1)"

benchmark:
	. venv/bin/activate && python3 -m intelligence.validation.synthetic_benchmark

start:
	. venv/bin/activate && python3 -m intelligence.main continuous

once:
	. venv/bin/activate && python3 -m intelligence.main once

health:
	curl -s http://localhost:8000/api/health | python3 -m json.tool

clean:
	rm -rf venv intelligence/data/*.db intelligence/data/*.db-wal intelligence/data/*.db-shm logs/*.log
