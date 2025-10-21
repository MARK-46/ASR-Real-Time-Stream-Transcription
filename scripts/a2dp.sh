#!/usr/bin/env bash
# =============================================================================
#  A2DP‑Sink Bluetooth‑headset setup (e.g. AirPods Pro)
#
#  What this script does:
#   • Installs PulseAudio (with Bluetooth modules) + BlueZ tools + expect
#   • Writes a *single* ordered block of Bluetooth modules into
#       /etc/pulse/system.pa  (no duplicate lines)
#   • Brings the Bluetooth adapter up and configures its name / class / MAC
#   • Uses **Expect** to feed a reliable command sequence to bluetoothctl
#   • Restarts PulseAudio and shows the final controller status
#
#  Run it once – it can be re‑run safely.
# =============================================================================

set -euo pipefail          # fail on errors, undefined vars, bad pipelines

# -------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------
log()   { echo -e "\e[1;34m[INFO]\e[0m $*"; }
error() { echo -e "\e[1;31m[ERROR]\e[0m $*" >&2; }

# -------------------------------------------------------------------------
# 1️⃣  Make sure we run as root (re‑exec via sudo if not)
# -------------------------------------------------------------------------
if [[ "$EUID" -ne 0 ]]; then
    log "Not root – re‑executing with sudo..."
    exec sudo bash "$0" "$@"
fi

# -------------------------------------------------------------------------
# 2️⃣  User‑configurable variables (change only here)
# -------------------------------------------------------------------------
DEV_NAME="AirPods Pro (MARK-46)"   # name that will appear in Bluetooth UI
DEV_MAC="1C:27:BE:4E:9A:AC"        # optional public MAC address (keep blank if not needed)
PULSE_CONF="/etc/pulse/system.pa" # PulseAudio system‑wide config file
BT_CLASS="0x240404"               # 0x240404 → Audio/Video + Wearable Headset

# Export the name so Expect can read it via $env(...)
export DEV_NAME

# -------------------------------------------------------------------------
# 3️⃣  Install required packages (includes expect)
# -------------------------------------------------------------------------
log "Updating APT cache..."
apt-get update -qq

log "Installing required packages (pulseaudio, bluez, expect …)"
DEPS=(
    pulseaudio
    pulseaudio-module-bluetooth
    bluez
    bluez-tools
    expect
    pavucontrol
)
apt-get install -y "${DEPS[@]}" >/dev/null

# -------------------------------------------------------------------------
# 4️⃣  PulseAudio – one clean block with the three Bluetooth modules
# -------------------------------------------------------------------------
log "Configuring PulseAudio for Bluetooth A2DP sink …"
# Remove any stale “module‑bluetooth‑*” lines (if they exist)
sed -i '/^load-module module-bluetooth-/d' "$PULSE_CONF"

# Append the correct modules – order matters!
cat <<'EOF' >> "$PULSE_CONF"
load-module module-bluetooth-discover
load-module module-bluetooth-policy
load-module module-bluetooth-device profile=a2dp_sink
EOF

# -------------------------------------------------------------------------
# 5️⃣  Bring up the Bluetooth adapter and enable the daemon
# -------------------------------------------------------------------------
log "Bringing up the Bluetooth adapter (hci0)…"
hciconfig hci0 up
hciconfig hci0 piscan

log "Enabling and starting bluetooth.service …"
systemctl enable --now bluetooth.service

# -------------------------------------------------------------------------
# 6️⃣  Set the controller’s name, public address & class via btmgmt
# -------------------------------------------------------------------------
log "Configuring controller name / address / class …"
btmgmt power off
btmgmt name "$DEV_NAME"

if [[ -n "$DEV_MAC" ]]; then
    btmgmt public-addr "$DEV_MAC"
fi

btmgmt class "$BT_CLASS" 0x0
btmgmt power on

# -------------------------------------------------------------------------
# 7️⃣  Interact with bluetoothctl **via Expect**
# -------------------------------------------------------------------------
log "Running bluetoothctl commands through Expect …"
expect <<'EOT'
    # --------------------------------------------------------------
    #  Expect script – each command waits for the bluetoothctl prompt
    #  before sending the next one.  This is more reliable than a
    #  simple here‑document because we guarantee the previous command
    #  has finished (including any “Agent registered” messages).
    # --------------------------------------------------------------

    log_user 1               ;# echo Expect’s interaction to the terminal
    set timeout 3           ;# generous timeout for slower adapters

    # Spawn bluetoothctl and wait for its first prompt
    spawn bluetoothctl
    expect -re {^\[bluetooth\]\#}

    # 1) Power on the controller
    send "power on\r"
    expect -re {^\[bluetooth\]\#}

    # 2) Register an agent (required for pairing)
    send "agent on\r"
    expect -re {^\[bluetooth\]\#}

    # 3) Make the just‑registered agent the default
    send "default-agent\r"
    expect -re {^\[bluetooth\]\#}

    # 4) Set a human‑readable alias (the name we exported)
    send "system-alias \"$env(DEV_NAME)\"\r"
    expect -re {^\[bluetooth\]\#}

    # 5) Allow pairing requests
    send "pairable on\r"
    expect -re {^\[bluetooth\]\#}

    # 6) Become discoverable so phones can see the headset
    send "discoverable on\r"
    expect -re {^\[bluetooth\]\#}

    # 7) Clean exit
    send "quit\r"
    expect eof
EOT

# -------------------------------------------------------------------------
# 8️⃣  Restart PulseAudio (kill any lingering instance, then start fresh)
# -------------------------------------------------------------------------
log "Restarting PulseAudio …"
pulseaudio -k || true   # ignore error if no daemon was running
pulseaudio --start

# -------------------------------------------------------------------------
# 9️⃣  Show final controller status and finish
# -------------------------------------------------------------------------
log "Current Bluetooth controller status:"
bluetoothctl show

log "✅  Setup complete – your headset should now appear as an A2DP sink."

exit 0
