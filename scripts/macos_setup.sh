#!/bin/bash
# macOS: make live packet capture work for the unprivileged NIDS backend.
#
# Root cause this fixes: /dev/bpf* is root-only, the backend runs tcpdump
# with no sudo (no TTY for a password prompt), and macOS has no `setcap`
# (the Linux workaround). Wireshark ships a "ChmodBPF" launch daemon that
# chmods /dev/bpf* to be group-'admin' readable at boot, after which
# `tcpdump -i en0` works for any admin user with no sudo. The Wireshark
# cask also installs tshark + capinfos, which the packet table and the
# packet-count/duration readback (and the FILE ANALYSIS .pcap upload
# path) need on PATH.
#
# Idempotent — safe to re-run.
set -e

if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew is required. Install it first: https://brew.sh" >&2
  exit 1
fi

if brew list --cask wireshark >/dev/null 2>&1; then
  echo "Wireshark cask already installed."
else
  echo "Installing Wireshark (dumpcap, tshark, capinfos, ChmodBPF)..."
  brew install --cask wireshark
fi

# Load the ChmodBPF daemon. The plist path has moved across cask
# versions, so try both and don't fail the script if neither is present
# yet (the user can still run the bundled "Install ChmodBPF" pkg).
for plist in \
  /Library/LaunchDaemons/org.wireshark.ChmodBPF.plist \
  /Library/LaunchDaemons/org.wireshark.ChmodBPF.plist.disabled ; do
  if [ -f "$plist" ]; then
    echo "Loading $plist ..."
    sudo launchctl load -w "$plist" 2>/dev/null || true
  fi
done

CHMODBPF_PKG="/Applications/Wireshark.app/Contents/Resources/Extras/Install ChmodBPF.pkg"
if [ ! -c /dev/bpf0 ] || [ "$(stat -f '%Sg' /dev/bpf0)" != "admin" ]; then
  if [ -f "$CHMODBPF_PKG" ]; then
    echo
    echo "/dev/bpf0 is still not group-'admin'. Running the ChmodBPF installer"
    echo "(you will be prompted for your password):"
    sudo installer -pkg "$CHMODBPF_PKG" -target / || true
  fi
fi

echo
echo "=== /dev/bpf0 ==="
ls -l /dev/bpf0 || true
echo
echo "=== capture tools on PATH ==="
for t in tcpdump dumpcap tshark capinfos ; do
  printf '%-9s %s\n' "$t" "$(command -v "$t" || echo 'NOT FOUND')"
done
echo
echo "=== tcpdump -D ==="
tcpdump -D | head -8 || true
echo
echo "If /dev/bpf0 shows group 'admin' with a read bit, live capture will work"
echo "after you log out and back in (or reboot). If it still shows 'wheel',"
echo "open Wireshark once and accept its prompt to install ChmodBPF, then"
echo "re-run this script."
echo
echo "Fallback without Wireshark: 'sudo chmod +r /dev/bpf*' after each boot,"
echo "or start the backend under sudo."
