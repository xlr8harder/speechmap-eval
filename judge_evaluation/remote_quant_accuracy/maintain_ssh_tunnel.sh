#!/usr/bin/env bash
set -uo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 LOCAL_PORT USER_HOST SSH_PORT STDOUT_LOG STDERR_LOG EVENT_LOG" >&2
  exit 2
fi

local_port=$1
user_host=$2
ssh_port=$3
stdout_log=$4
stderr_log=$5
event_log=$6
child_pid=

cleanup() {
  status=$?
  trap - EXIT INT TERM
  if [[ -n "$child_pid" ]]; then
    kill "$child_pid" >/dev/null 2>&1 || true
    wait "$child_pid" >/dev/null 2>&1 || true
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

attempt=0
while true; do
  attempt=$((attempt + 1))
  printf '%s\tattempt=%s\tevent=connecting\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)" "$attempt" >> "$event_log"
  ssh \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -o ConnectTimeout=10 \
    -o ConnectionAttempts=1 \
    -o ServerAliveInterval=15 \
    -o ServerAliveCountMax=3 \
    -o ExitOnForwardFailure=yes \
    -p "$ssh_port" \
    -N -L "127.0.0.1:${local_port}:127.0.0.1:8000" "$user_host" \
    >> "$stdout_log" 2>> "$stderr_log" &
  child_pid=$!
  wait "$child_pid"
  status=$?
  child_pid=
  printf '%s\tattempt=%s\tevent=disconnected\tstatus=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)" "$attempt" "$status" >> "$event_log"
  sleep 1
done
