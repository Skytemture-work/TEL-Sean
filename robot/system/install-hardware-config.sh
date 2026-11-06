#!/bin/sh
set -eu

if [ "$(id -u)" -ne 0 ]; then
    echo "Run with sudo: sudo $0" >&2
    exit 1
fi

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
project_dir=$(CDPATH= cd -- "$script_dir/../.." && pwd)

install -D -m 0644 "$project_dir/robot/systemd/can0-up.service" \
    /etc/systemd/system/can0-up.service
install -D -m 0644 "$project_dir/robot/udev/99-robot-serial.rules" \
    /etc/udev/rules.d/99-robot-serial.rules

systemctl daemon-reload
udevadm control --reload-rules
udevadm trigger --action=change --subsystem-match=tty
udevadm trigger --action=add --subsystem-match=net --sysname-match=can0

# Apply immediately when can0 is already present; udev starts it on later boots.
if [ -e /sys/class/net/can0 ]; then
    systemctl restart can0-up.service
fi

echo "Installed serial permissions and can0 startup configuration."
