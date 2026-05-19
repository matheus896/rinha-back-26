#!/bin/sh -e
rm -f "${SOCKET_PATH:?}"
export ARTIFACT_PATH="/artifact.bin"
exec /api
