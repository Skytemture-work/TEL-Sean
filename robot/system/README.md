# Hardware startup configuration

This installs persistent access rules for USB ACM serial ports and configures
`can0` to 500 kbit/s and UP when its USB CAN adapter appears.

Install from the repository root:

```sh
sudo ./robot/system/install-hardware-config.sh
```

The udev rule applies `root:dialout` and mode `0660` to all `/dev/ttyACM*`
ports, including `ttyACM0` and `ttyACM1`. The current `robot` account already
belongs to `dialout`.

USB ACM ports are serial devices and do not have a network `UP` state. Only
`can0` is brought UP. The CAN bitrate is set to 500 kbit/s, matching the
currently active interface configuration.
