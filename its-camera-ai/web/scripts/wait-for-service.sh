#!/bin/bash

# Wait for service to be available
# Usage: ./wait-for-service.sh host:port [timeout]

HOST_PORT=$1
TIMEOUT=${2:-60}

if [ -z "$HOST_PORT" ]; then
  echo "Usage: $0 host:port [timeout]"
  exit 1
fi

HOST=$(echo $HOST_PORT | cut -d: -f1)
PORT=$(echo $HOST_PORT | cut -d: -f2)

echo "Waiting for $HOST:$PORT to be available (timeout: ${TIMEOUT}s)..."

for i in $(seq 1 $TIMEOUT); do
  if nc -z $HOST $PORT 2>/dev/null; then
    echo "✅ $HOST:$PORT is available"
    exit 0
  fi

  if [ $((i % 10)) -eq 0 ]; then
    echo "⏳ Still waiting for $HOST:$PORT (${i}s elapsed)..."
  fi

  sleep 1
done

echo "❌ Timeout: $HOST:$PORT is not available after ${TIMEOUT}s"
exit 1