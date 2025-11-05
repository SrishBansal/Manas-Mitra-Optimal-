#!/bin/bash

# Remove all files matching patterns that are typically large
find . -type f -name "*.safetensors" -exec rm -f {} \;
find . -type f -name "*.dylib" -size +50M -exec rm -f {} \;
find . -type f -name "*.so" -size +50M -exec rm -f {} \;
find . -type f -name "*.pth" -size +50M -exec rm -f {} \;
find . -type f -name "*.bin" -size +50M -exec rm -f {} \;

# Clean up git history
git filter-repo --path-glob '*.safetensors' --invert-paths --force
git filter-repo --path-glob '*.dylib' --invert-paths --force
git filter-repo --path-glob '*.so' --invert-paths --force
git filter-repo --path-glob '*.pth' --invert-paths --force
git filter-repo --path-glob '*.bin' --invert-paths --force
git filter-repo --path venv_*/ --invert-paths --force
git filter-repo --path fine_tuned_emotion_model/ --invert-paths --force
git filter-repo --path results/ --invert-paths --force

# Clean up and optimize repository
git reflog expire --expire=now --all
git gc --prune=now --aggressive
