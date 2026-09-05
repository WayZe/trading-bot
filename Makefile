# Учебный крипто-бот: make-обёртки над CLI и Docker.
# Запуск без цели показывает краткий список.

SYMBOL      ?= BTC/USDT
TIMEFRAME   ?= 4h
SINCE       ?= 2024-01-01
CONFIG      ?= config/backtest.yaml
FAST_VALUES ?= 10,15,20
SLOW_VALUES ?= 30,50
IS_DAYS     ?= 180
OOS_DAYS    ?= 60
IMAGE       ?= trading-bot

.DEFAULT_GOAL := help
.PHONY: help sync test lint download backtest report sweep walkforward smoke docker-build docker-run

help: ## показать этот список
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

sync: ## установить зависимости (uv sync)
	uv sync

test: ## запустить тесты
	uv run pytest

lint: ## проверить код линтером
	uv run ruff check .

download: ## скачать/обновить свечи (SYMBOL=, TIMEFRAME=, SINCE=)
	@if [ -f "data/bybit/$$(echo '$(SYMBOL)' | tr '/' '_')/$(TIMEFRAME).parquet" ]; then \
		uv run trading-bot download --symbol $(SYMBOL) --timeframe $(TIMEFRAME) --update; \
	else \
		uv run trading-bot download --symbol $(SYMBOL) --timeframe $(TIMEFRAME) --since $(SINCE); \
	fi

backtest: ## прогнать бэктест (CONFIG=)
	uv run trading-bot backtest --config $(CONFIG)

report: ## отчёт по последнему прогону
	uv run trading-bot report

sweep: ## sweep по сетке (FAST_VALUES=, SLOW_VALUES=)
	uv run trading-bot sweep --config $(CONFIG) \
		--param fast=$(FAST_VALUES) --param slow=$(SLOW_VALUES)

walkforward: ## walk-forward валидация (FAST_VALUES=, SLOW_VALUES=, IS_DAYS=, OOS_DAYS=)
	uv run trading-bot walkforward --config $(CONFIG) \
		--param fast=$(FAST_VALUES) --param slow=$(SLOW_VALUES) \
		--is-days $(IS_DAYS) --oos-days $(OOS_DAYS)

smoke: backtest report ## backtest + report одной командой

docker-build: ## собрать Docker-образ
	docker build -t $(IMAGE) .

docker-run: ## пробросить команду в контейнер: make docker-run CMD="backtest --config ..."
	@if [ -z "$(CMD)" ]; then \
		docker run --rm -v $$(pwd)/data:/app/data -v $$(pwd)/reports:/app/reports -v $$(pwd)/config:/app/config $(IMAGE) --help; \
	else \
		docker run --rm -v $$(pwd)/data:/app/data -v $$(pwd)/reports:/app/reports -v $$(pwd)/config:/app/config $(IMAGE) $(CMD); \
	fi
