#!/bin/bash
# Deploy fallom package to PyPI
# Usage: ./deploy.sh [--test]

set -e

cd "$(dirname "$0")"

echo "🚀 Deploying Fallom SDK to PyPI"
echo ""

# Check for required tools
if ! command -v python &> /dev/null; then
    echo "❌ Python not found"
    exit 1
fi

# Install build tools if needed
echo "📦 Installing build tools..."
pip install --quiet build twine

# Clean old builds
echo "🧹 Cleaning old builds..."
rm -rf dist/ build/ *.egg-info fallom.egg-info/

# Build the package
echo "🔨 Building package..."
python -m build

# Get version from pyproject.toml
VERSION=$(grep 'version = ' pyproject.toml | head -1 | cut -d'"' -f2)
echo ""
echo "📋 Package version: $VERSION"
echo ""

# Check if --test flag is passed
if [[ "$1" == "--test" ]]; then
    echo "🧪 Uploading to TestPyPI..."
    python -m twine upload --repository testpypi dist/*
    echo ""
    echo "✅ Uploaded to TestPyPI!"
    echo "   Test install: pip install --index-url https://test.pypi.org/simple/ fallom"
else
    echo "📤 Uploading to PyPI..."
    python -m twine upload dist/*
    echo ""
    echo "✅ Published to PyPI!"
    echo "   Install: pip install fallom==$VERSION"
fi

echo ""
echo "🎉 Done!"

