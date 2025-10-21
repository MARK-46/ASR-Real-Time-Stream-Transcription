#!/bin/bash

pactl list short sinks

# Get the name of the currently active (default) audio output sink
# REAL_OUTPUT=$(pactl list short sinks | grep RUNNING | awk '{print $2}')
REAL_OUTPUT="alsa_output.pci-0000_00_1f.3.analog-stereo"

# Check if a real output sink was found
if [ -z "$REAL_OUTPUT" ]; then
  echo "No active output sink found. Exiting."
  exit 1
fi

echo "Active output sink: $REAL_OUTPUT"

# Check if the virtual sink already exists
if pactl list short sinks | grep -q "virtual_speaker"; then
  echo "Virtual speaker already exists. Skipping creation."
else
  # Load the virtual sink (virtual speaker)
  pactl load-module module-null-sink sink_name=virtual_speaker sink_properties=device.description="Virtual_Speaker"
  echo "Virtual speaker created."
fi

# Check if the loopback module already exists
if pactl list short modules | grep -q "module-loopback"; then
  echo "Loopback module already exists. Skipping creation."
else
  # Load the loopback module to send virtual speaker audio to the real output
  pactl load-module module-loopback source=virtual_speaker.monitor sink=$REAL_OUTPUT
  echo "Loopback module created."
fi

# Optionally, you can list sinks to confirm everything is working:
pactl list short sinks
echo "pavucontrol"


# sudo apt install pavucontrol
# pactl list short sinks
# pactl load-module module-null-sink sink_name=virtual_speaker sink_properties=device.description="Virtual_Speaker"
# pactl load-module module-loopback source=virtual_speaker.monitor sink=alsa_output.pci-0000_00_1f.3.analog-stereo
# pavucontrol
