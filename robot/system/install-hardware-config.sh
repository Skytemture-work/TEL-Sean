#!/bin/sh
set -eu

apply_now=true
if [ "${1-}" = "--no-apply" ] && [ "$#" -eq 1 ]; then
    apply_now=false
elif [ "$#" -ne 0 ]; then
    echo "Usage: $0 [--no-apply]" >&2
    exit 2
fi

if [ "$(id -u)" -ne 0 ]; then
    echo "Run with sudo: sudo $0" >&2
    exit 1
fi

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
project_dir=$(CDPATH= cd -- "$script_dir/../.." && pwd)

# can0.service is an alias, so one service owns the bus. Never overwrite a
# separately created unit: that would leave two CAN initializers running.
if [ -e /etc/systemd/system/can0.service ] || [ -L /etc/systemd/system/can0.service ]; then
    if [ "$(readlink -f /etc/systemd/system/can0.service)" != /etc/systemd/system/can0-up.service ]; then
        echo "A separate can0.service already exists. Resolve it before installing the can0-up.service alias." >&2
        exit 1
    fi
fi

install -D -m 0644 "$project_dir/robot/systemd/can0-up.service" \
    /etc/systemd/system/can0-up.service
install -D -m 0644 "$project_dir/robot/udev/99-robot-serial.rules" \
    /etc/udev/rules.d/99-robot-serial.rules

systemctl daemon-reload
systemctl enable can0-up.service
udevadm control --reload-rules
udevadm trigger --action=change --subsystem-match=tty

if [ "$apply_now" = true ]; then
    udevadm trigger --action=add --subsystem-match=net --sysname-match=can0
    # Starting an active oneshot is a no-op; do not interrupt a live bus.
    if [ -e /sys/class/net/can0 ]; then
        systemctl start can0-up.service
    fi
fi

echo "Installed serial permissions and can0 startup configuration."
