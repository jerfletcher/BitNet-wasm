# BitNet C++ Runner

This script allows you to easily build and run the original BitNet C++ implementation from the 3rdparty folder without making any changes to the BitNet repository.

## Features

- **Docker Support**: Build and run BitNet in a Docker container (recommended)
- **Local Build**: Fall back to local build with automatic dependency management
- **Model Detection**: Automatically finds models in the `models/` directory
- **Conda Integration**: Uses conda environments when available
- **Multiple Build Strategies**: Tries Docker first, falls back to local build

## Quick Start

### Using Docker (Recommended)

```bash
# Run with Docker (builds automatically)
npm run bitnet:run

# Chat mode with Docker
npm run bitnet:chat

# Custom prompt with Docker
npm run bitnet:demo
```

### Using Local Build

```bash
# Run with local build
npm run bitnet:local

# Chat mode with local build
npm run bitnet:local-chat
```

## Manual Usage

```bash
# Show help
node run-bitnet-cpp.js --help

# Use Docker (recommended)
node run-bitnet-cpp.js --docker --prompt "Hello, world!"

# Use local build
node run-bitnet-cpp.js --prompt "What is AI?"

# Chat mode
node run-bitnet-cpp.js --conversation --docker

# Custom parameters
node run-bitnet-cpp.js --prompt "Explain quantum computing" --n-predict 200 --temperature 0.7 --docker
```

## Requirements

### For Docker (Recommended)
- Docker installed and running
- Model files in `models/BitNet-b1.58-2B-4T/` directory

### For Local Build
- Python 3.9+ or Conda
- CMake
- C++ compiler (gcc/clang)
- Model files in `models/BitNet-b1.58-2B-4T/` directory

## Options

- `--prompt, -p`: Text prompt for inference
- `--threads, -t`: Number of threads (default: 2)
- `--ctx-size, -c`: Context size (default: 512)
- `--temp`: Temperature (default: 0.8)
- `--n-predict, -n`: Number of tokens to predict (default: 64)
- `--conversation, -cnv`: Enable chat mode
- `--docker`: Force Docker usage
- `--help, -h`: Show help

## How It Works

1. **Environment Setup**: Checks for Docker, Python, Conda
2. **Build**: Uses Docker container or local CMake build
3. **Model Detection**: Finds GGUF models in models directory
4. **Inference**: Runs BitNet inference with specified parameters

The script automatically handles:
- Conda environment creation and management
- Python dependency installation
- Missing header file issues
- Docker image building and management
- Model path resolution

## Troubleshooting

### Docker Issues
- Ensure Docker is installed and running
- Check Docker permissions (may need `sudo` on Linux)

### Local Build Issues
- Install CMake: `brew install cmake` (macOS) or `apt install cmake` (Ubuntu)
- Install build tools: `xcode-select --install` (macOS)
- Check Python version: `python --version` (should be 3.9+)

### Model Issues
- Ensure model files are in `models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf`
- Download models using the BitNet setup scripts if needed
