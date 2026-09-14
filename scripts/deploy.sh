#!/usr/bin/env bash
# Автоматический деплой trading-bot на сервер: fast-forward до origin/main
# и пересборка контейнера. Вызывается по SSH из GitHub Actions (ключ
# с forced command в authorized_keys ссылается на этот файл); вручную —
# для аварийного обновления. Состояние торговли (data/, .env) не затрагивается.
set -euo pipefail

# Второй одновременный деплой подождёт своей очереди, а не сломается посередине.
exec 9>/tmp/trading-bot-deploy.lock
flock -w 300 9

cd /opt/trading-bot
git fetch origin main
git merge --ff-only origin/main
docker compose up -d --build
docker compose ps
