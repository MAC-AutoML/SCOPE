#!/usr/bin/env bash
# OpenCV needs libGL. Download the Debian packages into the project and extract them.
# This does not install packages into the operating system.
set -Eeuo pipefail
ROOT="${SCOPE_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
NATIVE="${ROOT}/.native"
LIB="${NATIVE}/usr/lib/x86_64-linux-gnu"
if test -f "${LIB}/libGL.so.1" && test -f "${LIB}/libGLX.so.0" && test -f "${LIB}/libGLdispatch.so.0"; then
  exit 0
fi
test "$(dpkg --print-architecture)" = amd64
mkdir -p "${NATIVE}/lists/partial" "${NATIVE}/cache/archives/partial" "${NATIVE}/debs"
APT_OPTIONS=(-o "Dir::State::lists=${NATIVE}/lists" -o "Dir::Cache=${NATIVE}/cache")
apt-get "${APT_OPTIONS[@]}" update
cd "${NATIVE}/debs"
apt-get "${APT_OPTIONS[@]}" download libgl1 libglx0 libglvnd0
for package in ./*.deb; do
  dpkg-deb -x "$package" "$NATIVE"
done
test -f "${LIB}/libGL.so.1"
test -f "${LIB}/libGLX.so.0"
test -f "${LIB}/libGLdispatch.so.0"
