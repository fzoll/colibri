#!/bin/sh
# Generate SPIR-V from GLSL and embed as a C header.
#
# Usage:  cd c && sh shaders/generate_spirv.sh
#
# Requires: glslangValidator (from Vulkan SDK)
#
# This produces shaders/matmul.spv which backend_vulkan.c loads at runtime
# if the embedded SPIR-V blob is a placeholder.  To embed the blob directly
# in the .c file, paste the output of the xxd step into the g_matmul_spirv[]
# array and update g_matmul_spirv_size.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COMP="$SCRIPT_DIR/matmul.comp"
SPV="$SCRIPT_DIR/matmul.spv"

if ! command -v glslangValidator >/dev/null 2>&1; then
    echo "glslangValidator not found. Install the Vulkan SDK." >&2
    exit 1
fi

echo "Compiling $COMP -> $SPV"
glslangValidator -V "$COMP" -o "$SPV"
echo "SPIR-V size: $(wc -c < "$SPV") bytes"

# Optional: generate embeddable C array
if command -v xxd >/dev/null 2>&1; then
    HDR="$SCRIPT_DIR/matmul_spv.h"
    echo "Generating $HDR"
    xxd -i "$SPV" | sed \
        -e 's/unsigned char [a-zA-Z_]*/static const uint32_t g_matmul_spirv[]/' \
        -e 's/unsigned int [a-zA-Z_]*/static const size_t g_matmul_spirv_size/' \
        > "$HDR"
    echo "Done. Replace the placeholder in backend_vulkan.c with the contents of $HDR"
else
    echo "xxd not found — skipping C header generation"
fi
