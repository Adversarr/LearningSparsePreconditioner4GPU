#!/bin/bash

# Define the output filename
OUTPUT="src.tar.gz"

# Check if the output file exists and remove it if it does
if [ -f "$OUTPUT" ]; then
    rm "$OUTPUT"
fi

# Create a temporary list of files to exclude based on .gitignore
EXCLUDE_FILE=".gitignore"
if [ ! -f "$EXCLUDE_FILE" ]; then
    echo "Error: .gitignore file not found."
    exit 1
fi

# Generate the tar command with all exclusions
EXCLUDE_PATTERNS=$(grep -v '^#' "$EXCLUDE_FILE" | sed '/^$/d' | sed 's/^/--exclude=/')
tar -czvf "$OUTPUT" $EXCLUDE_PATTERNS --exclude=ext *