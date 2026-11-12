#!/usr/bin/env bash
# 一键启动: 首次运行自动建虚拟环境并安装依赖, 然后打开网页看板.
# 用法: ./start.sh            (参数会透传, 例如 ./start.sh --port 9000 --host 0.0.0.0)
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "首次运行: 创建 .venv 并安装依赖 ..."
  python3 -m venv .venv
  .venv/bin/pip install -q -r requirements.txt
fi
exec .venv/bin/python -m emo.cli web "$@"
