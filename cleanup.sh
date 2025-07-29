#!/bin/bash

# Move simplified prototype files to main directory
cp -r simplified-prototype/* .

# Create static directory if it doesn't exist
mkdir -p static

# Keep essential files and directories
KEEP=(
  "index.html"
  "server.py"
  "static"
  "README.md"
  "requirements.txt"
  "venv"
  "cleanup.sh"
)

# Remove everything except the files/directories to keep
for item in *; do
  keep=false
  for keep_item in "${KEEP[@]}"; do
    if [ "$item" == "$keep_item" ]; then
      keep=true
      break
    fi
  done
  
  if [ "$keep" == false ]; then
    if [ -d "$item" ]; then
      echo "Removing directory: $item"
      rm -rf "$item"
    else
      echo "Removing file: $item"
      rm -f "$item"
    fi
  fi
done

# Remove the simplified-prototype directory as we've moved its contents
rm -rf simplified-prototype

echo "Cleanup complete! Only essential files remain."
