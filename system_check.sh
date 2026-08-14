#!/usr/bin/env bash
# Jetson health snapshot: the settings and conditions that cause latency
# spikes, checked in one place.
#
#   ./system_check.sh          once
#   ./system_check.sh watch    every 2 s
set -u

check() {
echo "=== power / clocks ==============================================="
sudo nvpmodel -q 2>/dev/null | sed 's/^/  /'
echo "  jetson_clocks:"
sudo jetson_clocks --show 2>/dev/null | head -8 | sed 's/^/    /'

echo
echo "=== cpu ========================================================="
paste -d' ' \
  <(ls -d /sys/devices/system/cpu/cpu[0-9]* | sed 's|.*/||') \
  <(cat /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq 2>/dev/null \
    | awk '{printf "%.0fMHz\n", $1/1000}') \
  <(cat /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor 2>/dev/null) \
  | sed 's/^/  /'

echo
echo "=== thermal ====================================================="
for z in /sys/devices/virtual/thermal/thermal_zone*; do
  n=$(cat $z/type 2>/dev/null)
  t=$(cat $z/temp 2>/dev/null)
  [ -n "$t" ] && printf "  %-14s %.1f C\n" "$n" "$(echo "$t/1000" | bc -l)"
done

echo
echo "=== memory ======================================================"
free -m | sed 's/^/  /'
echo "  --- top RSS ---"
ps -eo rss,comm --sort=-rss | head -7 | awk '{printf "  %8.0f MB  %s\n", $1/1024, $2}'

echo
echo "=== throttling =================================================="
# Non-zero counters here mean the SoC has capped itself at some point.
for f in /sys/devices/platform/bus@0/*/thermal_throttling \
         /sys/kernel/debug/bpmp/debug/soctherm/*; do
  [ -r "$f" ] && echo "  $f = $(cat $f 2>/dev/null)"
done 2>/dev/null
echo "  (empty above is normal on many JetPack builds; use tegrastats"
echo "   and watch for GR3D_FREQ or cpu MHz dropping under sustained load)"

echo
echo "=== ros nodes ==================================================="
if command -v ros2 >/dev/null 2>&1; then
  ros2 node list 2>/dev/null | sed 's/^/  /'
fi
}

if [ "${1:-}" = "watch" ]; then
    watch -n2 "$0"
else
    check
fi
