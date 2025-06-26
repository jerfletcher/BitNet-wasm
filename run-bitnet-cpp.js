#!/usr/bin/env node

const { spawn, execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

class BitNetCppRunner {
    constructor() {
        this.bitnetDir = path.join(__dirname, '3rdparty', 'BitNet');
        this.modelsDir = path.join(__dirname, 'models');
        this.buildDir = path.join(this.bitnetDir, 'build');
        this.condaEnvName = 'bitnet-cpp';
        this.useDocker = false; // Set to true to use Docker instead of local build
    }

    log(message) {
        console.log(`[BitNet-CPP] ${message}`);
    }

    error(message) {
        console.error(`[BitNet-CPP ERROR] ${message}`);
    }

    async runCommand(command, options = {}) {
        return new Promise((resolve, reject) => {
            this.log(`Running: ${command}`);
            
            const proc = spawn(command, {
                stdio: 'inherit',
                shell: true,
                cwd: options.cwd || this.bitnetDir,
                ...options
            });

            proc.on('close', (code) => {
                if (code === 0) {
                    resolve();
                } else {
                    reject(new Error(`Command failed with exit code ${code}: ${command}`));
                }
            });

            proc.on('error', (err) => {
                reject(err);
            });
        });
    }

    checkDocker() {
        try {
            execSync('docker --version', { encoding: 'utf8' });
            this.log('Docker found');
            return true;
        } catch (error) {
            this.log('Docker not found - will use local build');
            return false;
        }
    }

    async buildWithDocker() {
        this.log('Building BitNet with Docker...');

        // Create a Dockerfile for BitNet
        const dockerfileContent = `
FROM ubuntu:22.04

# Install dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    cmake \\
    git \\
    python3 \\
    python3-pip \\
    wget \\
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /bitnet

# Copy BitNet source
COPY . .

# Build BitNet
RUN mkdir -p build && cd build && \\
    cmake .. && \\
    make -j$(nproc)

# Set the entrypoint
ENTRYPOINT ["./build/bin/llama-cli"]
`;

        const dockerfilePath = path.join(this.bitnetDir, 'Dockerfile.bitnet');
        fs.writeFileSync(dockerfilePath, dockerfileContent);

        try {
            // Build Docker image
            this.log('Building Docker image (this may take a while)...');
            await this.runCommand(`docker build -f Dockerfile.bitnet -t bitnet-cpp .`);

            this.log('Docker image built successfully');
            return true;
        } catch (error) {
            this.error(`Docker build failed: ${error.message}`);
            return false;
        } finally {
            // Clean up Dockerfile
            if (fs.existsSync(dockerfilePath)) {
                fs.unlinkSync(dockerfilePath);
            }
        }
    }

    async runInferenceWithDocker(modelPath, prompt, options = {}) {
        const {
            threads = 2,
            ctxSize = 512,
            temperature = 0.8,
            nPredict = 128
        } = options;

        // Get absolute paths
        const absoluteModelPath = path.resolve(modelPath);
        const modelDir = path.dirname(absoluteModelPath);
        const modelFile = path.basename(absoluteModelPath);

        const dockerArgs = [
            'docker', 'run', '--rm',
            '-v', `${modelDir}:/models`,
            'bitnet-cpp',
            '-m', `/models/${modelFile}`,
            '-p', prompt,
            '-t', threads.toString(),
            '-c', ctxSize.toString(),
            '--temp', temperature.toString(),
            '-n', nPredict.toString()
        ];

        try {
            await this.runCommand(dockerArgs.join(' '));
        } catch (error) {
            this.error(`Docker inference failed: ${error.message}`);
            throw error;
        }
    }

    checkPython() {
        try {
            const pythonVersion = execSync('python --version', { encoding: 'utf8' });
            this.log(`Python found: ${pythonVersion.trim()}`);
            return true;
        } catch (error) {
            try {
                const python3Version = execSync('python3 --version', { encoding: 'utf8' });
                this.log(`Python3 found: ${python3Version.trim()}`);
                return 'python3';
            } catch (error3) {
                this.error('Python not found. Please install Python 3.9+');
                return false;
            }
        }
    }

    checkConda() {
        try {
            execSync('conda --version', { encoding: 'utf8' });
            this.log('Conda found');
            return true;
        } catch (error) {
            this.log('Conda not found - will use system Python with pip');
            return false;
        }
    }

    async setupEnvironment() {
        this.log('Setting up BitNet environment...');

        if (!fs.existsSync(this.bitnetDir)) {
            this.error(`BitNet directory not found: ${this.bitnetDir}`);
            process.exit(1);
        }

        // Check Python
        const pythonCmd = this.checkPython();
        if (!pythonCmd) {
            process.exit(1);
        }

        // Check for requirements.txt
        const reqFile = path.join(this.bitnetDir, 'requirements.txt');
        if (!fs.existsSync(reqFile)) {
            this.error('requirements.txt not found in BitNet directory');
            process.exit(1);
        }

        // Setup environment
        const hasConda = this.checkConda();
        
        if (hasConda) {
            this.log('Setting up conda environment...');
            try {
                // Check if environment exists
                try {
                    execSync(`conda env list | grep ${this.condaEnvName}`, { encoding: 'utf8' });
                    this.log('Conda environment already exists');
                } catch {
                    this.log('Creating conda environment...');
                    await this.runCommand(`conda create -n ${this.condaEnvName} python=3.9 -y`);
                }

                // Install requirements
                this.log('Installing Python dependencies...');
                await this.runCommand(`conda run -n ${this.condaEnvName} pip install -r requirements.txt`);
            } catch (error) {
                this.error(`Failed to setup conda environment: ${error.message}`);
                process.exit(1);
            }
        } else {
            this.log('Installing dependencies with pip...');
            try {
                const pipCmd = pythonCmd === 'python3' ? 'pip3' : 'pip';
                await this.runCommand(`${pipCmd} install -r requirements.txt`);
            } catch (error) {
                this.error(`Failed to install pip dependencies: ${error.message}`);
                process.exit(1);
            }
        }
    }

    async findModel() {
        this.log('Looking for available models...');

        // Check for BitNet model in models directory
        const bitnetModelDir = path.join(this.modelsDir, 'BitNet-b1.58-2B-4T');
        const tinyModelDir = path.join(this.modelsDir, 'tiny');
        const testModelDir = path.join(this.modelsDir, 'test');

        let modelPath = null;

        // Look for GGUF files in different locations
        const searchPaths = [
            path.join(bitnetModelDir, 'ggml-model-i2_s.gguf'),
            path.join(bitnetModelDir, 'model.gguf'),
            path.join(tinyModelDir, 'model.gguf'),
            path.join(testModelDir, 'model.gguf'),
            // Also check build directory for downloaded models
            path.join(this.bitnetDir, 'models', 'BitNet-b1.58-2B-4T', 'ggml-model-i2_s.gguf')
        ];

        for (const searchPath of searchPaths) {
            if (fs.existsSync(searchPath)) {
                modelPath = searchPath;
                this.log(`Found model: ${modelPath}`);
                break;
            }
        }

        if (!modelPath) {
            this.error('No GGUF model found. Available options:');
            this.error('1. Download a model using setup_env.py');
            this.error('2. Place a .gguf file in models/BitNet-b1.58-2B-4T/');
            this.error('3. Use a test model in models/tiny/ or models/test/');
            return null;
        }

        return modelPath;
    }

    async downloadModel() {
        this.log('Downloading BitNet model...');
        
        const modelDir = path.join(this.bitnetDir, 'models', 'BitNet-b1.58-2B-4T');
        
        try {
            const hasConda = this.checkConda();
            if (hasConda) {
                await this.runCommand(`conda run -n ${this.condaEnvName} huggingface-cli download microsoft/BitNet-b1.58-2B-4T-gguf --local-dir ${modelDir}`);
                await this.runCommand(`conda run -n ${this.condaEnvName} python setup_env.py -md ${modelDir} -q i2_s`);
            } else {
                await this.runCommand(`huggingface-cli download microsoft/BitNet-b1.58-2B-4T-gguf --local-dir ${modelDir}`);
                await this.runCommand(`python setup_env.py -md ${modelDir} -q i2_s`);
            }
            
            return path.join(modelDir, 'ggml-model-i2_s.gguf');
        } catch (error) {
            this.error(`Failed to download model: ${error.message}`);
            return null;
        }
    }

    async buildProject() {
        this.log('Building BitNet project...');

        // Check if we should use Docker
        if (this.useDocker || this.checkDocker()) {
            this.log('Using Docker build approach...');
            const dockerSuccess = await this.buildWithDocker();
            if (dockerSuccess) {
                this.useDocker = true;
                return;
            } else {
                this.log('Docker build failed, falling back to local build...');
            }
        }

        // Fallback to local build
        await this.buildProjectLocal();
    }

    async buildProjectLocal() {
        this.log('Building BitNet project locally...');

        // Check if already built
        const llamaCliPath = path.join(this.buildDir, 'bin', 'llama-cli');
        if (fs.existsSync(llamaCliPath)) {
            this.log('BitNet already built');
            return;
        }

        try {
            // Create build directory if it doesn't exist
            if (!fs.existsSync(this.buildDir)) {
                fs.mkdirSync(this.buildDir, { recursive: true });
            }

            // Copy missing header file if needed
            const headerSrc = path.join(__dirname, 'include', 'bitnet-lut-kernels.h');
            const headerDest = path.join(this.bitnetDir, 'include', 'bitnet-lut-kernels.h');
            
            if (fs.existsSync(headerSrc) && !fs.existsSync(headerDest)) {
                this.log('Copying missing header file...');
                const includeDir = path.join(this.bitnetDir, 'include');
                if (!fs.existsSync(includeDir)) {
                    fs.mkdirSync(includeDir, { recursive: true });
                }
                fs.copyFileSync(headerSrc, headerDest);
            }
            
            this.log('Running CMake configuration...');
            await this.runCommand('cmake ..', { cwd: this.buildDir });
            
            this.log('Building with make...');
            await this.runCommand('make -j4', { cwd: this.buildDir });
            
            // Verify the build was successful
            if (!fs.existsSync(llamaCliPath)) {
                throw new Error('Build failed: llama-cli binary not found after build');
            }
            
            this.log('BitNet build completed successfully');
        } catch (error) {
            this.error(`Build failed: ${error.message}`);
            throw error;
        }
    }

    async runInference(modelPath, prompt = "You are a helpful assistant", options = {}) {
        this.log(`Running inference with model: ${modelPath}`);
        this.log(`Prompt: "${prompt}"`);

        // Use Docker if available and built
        if (this.useDocker) {
            try {
                return await this.runInferenceWithDocker(modelPath, prompt, options);
            } catch (error) {
                this.log('Docker inference failed, falling back to local binary...');
                this.useDocker = false;
            }
        }

        // Check if we have the local binary
        const llamaCliPath = path.join(this.buildDir, 'bin', 'llama-cli');
        if (fs.existsSync(llamaCliPath)) {
            return await this.runInferenceWithBinary(modelPath, prompt, options);
        }

        // Use traditional Python script approach as final fallback
        return await this.runInferenceLocal(modelPath, prompt, options);
    }

    async runInferenceWithBinary(modelPath, prompt, options = {}) {
        const {
            threads = 2,
            ctxSize = 512,
            temperature = 0.8,
            nPredict = 128
        } = options;

        const llamaCliPath = path.join(this.buildDir, 'bin', 'llama-cli');
        
        const args = [
            llamaCliPath,
            '-m', `"${modelPath}"`,
            '-p', `"${prompt}"`,
            '-t', threads.toString(),
            '-c', ctxSize.toString(),
            '--temp', temperature.toString(),
            '-n', nPredict.toString()
        ];

        try {
            await this.runCommand(args.join(' '));
        } catch (error) {
            this.error(`Binary inference failed: ${error.message}`);
            throw error;
        }
    }

    async runInferenceLocal(modelPath, prompt = "You are a helpful assistant", options = {}) {
        const {
            threads = 2,
            ctxSize = 512,
            temperature = 0.8,
            nPredict = 128,
            conversation = false
        } = options;

        // Escape the prompt for shell safety
        const escapedPrompt = prompt.replace(/"/g, '\\"');

        const args = [
            'run_inference.py',
            '-m', `"${modelPath}"`,
            '-p', `"${escapedPrompt}"`,
            '-t', threads.toString(),
            '-c', ctxSize.toString(),
            '-temp', temperature.toString(),
            '-n', nPredict.toString()
        ];

        if (conversation) {
            args.push('-cnv');
        }

        try {
            const hasConda = this.checkConda();
            if (hasConda) {
                await this.runCommand(`conda run -n ${this.condaEnvName} python ${args.join(' ')}`);
            } else {
                await this.runCommand(`python ${args.join(' ')}`);
            }
        } catch (error) {
            this.error(`Inference failed: ${error.message}`);
            throw error;
        }
    }

    async run(options = {}) {
        try {
            // Setup environment
            await this.setupEnvironment();

            // Build project
            await this.buildProject();

            // Find or download model
            let modelPath = await this.findModel();
            
            if (!modelPath) {
                this.log('No model found, attempting to download...');
                modelPath = await this.downloadModel();
                
                if (!modelPath) {
                    this.error('Failed to find or download a model');
                    process.exit(1);
                }
            }

            // Run inference
            const prompt = options.prompt || "You are a helpful assistant. Please introduce yourself.";
            await this.runInference(modelPath, prompt, {
                threads: options.threads || 2,
                ctxSize: options.ctxSize || 512,
                temperature: options.temperature || 0.8,
                nPredict: options.nPredict || 64,
                conversation: options.conversation || false
            });

            this.log('Inference completed successfully!');

        } catch (error) {
            this.error(`Failed to run BitNet: ${error.message}`);
            process.exit(1);
        }
    }
}

// CLI interface
if (require.main === module) {
    const args = process.argv.slice(2);
    const options = {};

    // Parse command line arguments
    for (let i = 0; i < args.length; i++) {
        const arg = args[i];
        switch (arg) {
            case '--prompt':
            case '-p':
                options.prompt = args[++i];
                break;
            case '--threads':
            case '-t':
                options.threads = parseInt(args[++i]);
                break;
            case '--ctx-size':
            case '-c':
                options.ctxSize = parseInt(args[++i]);
                break;
            case '--temperature':
            case '--temp':
                options.temperature = parseFloat(args[++i]);
                break;
            case '--n-predict':
            case '-n':
                options.nPredict = parseInt(args[++i]);
                break;
            case '--conversation':
            case '-cnv':
                options.conversation = true;
                break;
            case '--docker':
                options.forceDocker = true;
                break;
            case '--help':
            case '-h':
                console.log(`
Usage: node run-bitnet-cpp.js [options]

Options:
  -p, --prompt <text>       Prompt for inference (default: "You are a helpful assistant. Please introduce yourself.")
  -t, --threads <number>    Number of threads (default: 2)
  -c, --ctx-size <number>   Context size (default: 512)
  --temp <number>           Temperature (default: 0.8)
  -n, --n-predict <number>  Number of tokens to predict (default: 64)
  -cnv, --conversation      Enable conversation mode
  --docker                  Force Docker build/run (recommended)
  -h, --help                Show this help

Examples:
  node run-bitnet-cpp.js --docker
  node run-bitnet-cpp.js --prompt "What is artificial intelligence?" --docker
  node run-bitnet-cpp.js --prompt "Hello" --conversation --n-predict 100
`);
                process.exit(0);
                break;
        }
    }

    const runner = new BitNetCppRunner();
    
    // Force Docker if requested
    if (options.forceDocker) {
        runner.useDocker = true;
    }
    
    runner.run(options);
}

module.exports = BitNetCppRunner;
