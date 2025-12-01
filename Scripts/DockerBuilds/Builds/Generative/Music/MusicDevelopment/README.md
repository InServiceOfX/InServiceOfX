# Music Development Docker - Audio Setup for Mixxx

## Host-Side Audio Configuration (One-Time Setup)

### Step 1: Configure PulseAudio for Docker Access

**On host:**
```bash
cp /etc/pulse/default.pa ~/.config/pulse/default.pa
nano ~/.config/pulse/default.pa
```

### Step 2: Enable Anonymous PulseAudio Connections

Find this line:

```
load-module module-native-protocol-unix
```

**Replace it with:**

```
load-module module-native-protocol-unix auth-anonymous=1
```

**Important:** Replace the line (don't comment it out and add a new one).

### Step 3: Restart PulseAudio

**On host:**
```bash
pulseaudio --kill
pulseaudio --start
```

### Step 4: Verify Configuration

**On host:**
```bash
pulseaudio --check
pactl list modules short | grep native-protocol
```

## Running Mixxx with Audio

**On host:**
```bash
cd Scripts/DockerBuilds/Builds/Generative/Music/MusicDevelopment
python ../../../Utilities/DockerRun.py --gui --audio
```

## Troubleshooting

**In container:**
```bash
# Check available audio devices
aplay -l

# Test PulseAudio connection
pactl info

# List PulseAudio sinks
pactl list sinks short
```

**On host:**
```bash
# Verify PulseAudio is running
pulseaudio --check

# Check PulseAudio sinks
pactl list sinks short
```

Mixxx uses ALSA backend (configured in Mixxx preferences). PulseAudio socket is mounted for compatibility.