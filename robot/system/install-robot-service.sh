#!/bin/sh
set -eu

if [ "$(id -u)" -ne 0 ]; then
    echo "Run with sudo: sudo $0" >&2
    exit 1
fi

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
robot_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)
expected_dir=/home/robot/robot/TEL-Sean/robot
if [ "$robot_dir" != "$expected_dir" ]; then
    echo "robot.service expects $expected_dir; this checkout is $robot_dir." >&2
    exit 1
fi
for group in dialout video render; do
    getent group "$group" >/dev/null
    usermod -aG "$group" robot
done

# Install without reconfiguring an active CAN bus or starting the robot.
"$script_dir/install-hardware-config.sh" --no-apply
install -D -m 0644 "$robot_dir/systemd/robot.service" /etc/systemd/system/robot.service
systemctl daemon-reload
systemd-analyze verify /etc/systemd/system/can0-up.service /etc/systemd/system/robot.service
systemctl enable robot.service

echo "Enabled can0.service (alias of can0-up.service) and robot.service for next boot."
echo "robot.service has not been started or restarted by this installer."
echo "Starting it will run main.py and perform its existing motor startup positioning."
echo "Start manually when ready: sudo systemctl start robot.service"
