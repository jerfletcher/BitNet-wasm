#!/usr/bin/env node

/**
 * Enhanced test script for BitNet WASM with real model loading
 * This script specifically tests the actual model file loading and inference
 */

const { chromium } = require('playwright');
const express = require('express');
const path = require('path');
const fs = require('fs');

class RealModelTester {
    constructor() {
        this.app = null;
        this.server = null;
        this.browser = null;
        this.page = null;
        this.port = 8081;
    }

    async setupServer() {
        console.log('🚀 Setting up test server...');
        
        this.app = express();
        
        // Enable CORS and proper headers for WASM
        this.app.use((req, res, next) => {
            res.header('Access-Control-Allow-Origin', '*');
            res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept');
            
            if (req.url.endsWith('.wasm')) {
                res.setHeader('Content-Type', 'application/wasm');
            }
            
            // Required headers for SharedArrayBuffer
            res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
            res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
            
            next();
        });

        // Serve static files
        this.app.use(express.static(__dirname));

        // Model info endpoint
        this.app.get('/api/model-info', (req, res) => {
            const modelPath = path.join(__dirname, 'models', 'BitNet-b1.58-2B-4T', 'ggml-model-i2_s.gguf');
            
            if (fs.existsSync(modelPath)) {
                const stats = fs.statSync(modelPath);
                res.json({
                    exists: true,
                    size: stats.size,
                    sizeGB: (stats.size / 1024 / 1024 / 1024).toFixed(2),
                    modified: stats.mtime,
                    path: 'models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf'
                });
            } else {
                res.json({ exists: false });
            }
        });

        return new Promise((resolve) => {
            this.server = this.app.listen(this.port, () => {
                console.log(`✅ Test server running on http://localhost:${this.port}`);
                resolve();
            });
        });
    }

    async setupBrowser() {
        console.log('🌐 Setting up browser...');
        
        this.browser = await chromium.launch({ 
            headless: false,  // Show browser for real model testing
            slowMo: 1000     // Slow down for better observation
        });
        
        this.page = await this.browser.newPage();
        
        // Listen for console messages from the page
        this.page.on('console', msg => {
            const text = msg.text();
            const type = msg.type();
            
            if (type === 'error') {
                console.log('❌ Browser Error:', text);
            } else if (type === 'warn') {
                console.log('⚠️  Browser Warning:', text);
            } else if (text.includes('[bitnet') || text.includes('BitNet') || text.includes('WASM')) {
                console.log('🔧 BitNet:', text);
            } else {
                // Log all other console messages for debugging
                console.log(`📝 Browser ${type}:`, text);
            }
        });

        // Listen for page errors
        this.page.on('pageerror', error => {
            console.log('💥 Page Error:', error.message);
        });
    }

    async checkFiles() {
        console.log('📁 Checking required files...');
        
        const files = {
            'bitnet.wasm': path.join(__dirname, 'bitnet.wasm'),
            'bitnet.js': path.join(__dirname, 'bitnet.js'),
            'test.html': path.join(__dirname, 'test.html'),
            'model': path.join(__dirname, 'models', 'BitNet-b1.58-2B-4T', 'ggml-model-i2_s.gguf')
        };

        for (const [name, filePath] of Object.entries(files)) {
            if (fs.existsSync(filePath)) {
                const stats = fs.statSync(filePath);
                if (name === 'model') {
                    console.log(`  ✅ ${name}: ${(stats.size / 1024 / 1024 / 1024).toFixed(2)} GB`);
                } else {
                    console.log(`  ✅ ${name}: ${(stats.size / 1024).toFixed(1)} KB`);
                }
            } else {
                console.log(`  ❌ ${name}: Missing`);
                if (name === 'bitnet.wasm' || name === 'bitnet.js') {
                    console.log('     Run ./build.sh to build WASM files');
                    return false;
                }
            }
        }
        return true;
    }

    async testRealModelLoading() {
        console.log('🧠 Testing real model loading...');
        
        await this.page.goto(`http://localhost:${this.port}/test.html`);
        
        // Wait for page to load
        await this.page.waitForLoadState('networkidle');
        
        // Wait for WASM initialization
        console.log('  ⏳ Waiting for WASM initialization...');
        await this.page.waitForSelector('#initialization.success', { timeout: 30000 });
        console.log('  ✅ WASM module loaded');

        // Run basic tests
        console.log('  ⏳ Running basic function tests...');
        await this.page.click('#test-basic');
        await this.page.waitForSelector('#basic-functions.success', { timeout: 10000 });
        console.log('  ✅ Basic functions work');

        // Select real model option
        console.log('  🎯 Selecting real model option...');
        await this.page.check('input[name="model-type"][value="real"]');
        
        // Start model loading (this will take a while)
        console.log('  📥 Starting real model loading (this may take several minutes)...');
        console.log('     Model size: ~1.2GB - be patient!');
        
        const startTime = Date.now();
        await this.page.click('#test-model');
        
        // Wait for model loading with extended timeout
        await this.page.waitForSelector('#model-loading.success', { timeout: 300000 }); // 5 minutes
        
        const loadTime = (Date.now() - startTime) / 1000;
        console.log(`  ✅ Real model loaded in ${loadTime.toFixed(1)} seconds`);

        // Get model info from the page
        const modelLog = await this.page.textContent('#model-log');
        const architectureMatch = modelLog.match(/Model architecture: vocab=(\d+), embedding=(\d+), layers=(\d+)/);
        
        if (architectureMatch) {
            console.log(`  📊 Model Architecture:`);
            console.log(`     Vocabulary size: ${architectureMatch[1]}`);
            console.log(`     Embedding dimension: ${architectureMatch[2]}`);
            console.log(`     Number of layers: ${architectureMatch[3]}`);
        }

        return true;
    }

    async testRealInference() {
        console.log('🔮 Testing real model inference...');
        
        const startTime = Date.now();
        await this.page.click('#test-inference');
        
        // Wait for inference to complete
        await this.page.waitForSelector('#inference-test.success', { timeout: 60000 });
        
        const inferenceTime = (Date.now() - startTime) / 1000;
        console.log(`  ✅ Real inference completed in ${inferenceTime.toFixed(1)} seconds`);

        // Get the inference results
        const inferenceLog = await this.page.textContent('#inference-log');
        const inputMatch = inferenceLog.match(/Input text: "([^"]+)"/);
        const outputMatch = inferenceLog.match(/Generated text: "([^"]+)"/);
        
        if (inputMatch && outputMatch) {
            console.log(`  📝 Input: "${inputMatch[1]}"`);
            console.log(`  🎯 Output: "${outputMatch[1]}"`);
            
            // Check if the output is different from input (real inference)
            if (outputMatch[1] !== inputMatch[1] && outputMatch[1].length > 0) {
                console.log(`  ✅ Model generated new text (not just echoing input)`);
                return true;
            } else {
                console.log(`  ⚠️  Output seems to be echoing input or empty`);
                return false;
            }
        }
        
        return false;
    }

    async runComprehensiveTest() {
        console.log('🎯 Running comprehensive real model test...');
        
        await this.page.click('#test-comprehensive');
        await this.page.waitForSelector('#comprehensive-test.success', { timeout: 120000 });
        
        const comprehensiveLog = await this.page.textContent('#comprehensive-log');
        const passMatch = comprehensiveLog.match(/Overall: (PASS|FAIL)/);
        
        if (passMatch && passMatch[1] === 'PASS') {
            console.log('  ✅ All comprehensive tests passed');
            return true;
        } else {
            console.log('  ❌ Some comprehensive tests failed');
            console.log('  📋 Test log excerpt:');
            console.log(comprehensiveLog.split('\n').slice(-10).join('\n'));
            return false;
        }
    }

    async cleanup() {
        console.log('🧹 Cleaning up...');
        
        if (this.page) {
            await this.page.close();
        }
        
        if (this.browser) {
            await this.browser.close();
        }
        
        if (this.server) {
            this.server.close();
        }
    }

    async run() {
        try {
            console.log('🚀 BitNet WASM Real Model Test');
            console.log('================================');
            
            // Pre-flight checks
            if (!await this.checkFiles()) {
                console.log('❌ Required files missing. Cannot proceed.');
                return false;
            }
            
            // Setup
            await this.setupServer();
            await this.setupBrowser();
            
            // Run tests
            let allPassed = true;
            
            console.log('\n📝 Test 1: Real Model Loading');
            if (!await this.testRealModelLoading()) {
                allPassed = false;
            }
            
            console.log('\n📝 Test 2: Real Model Inference');
            if (!await this.testRealInference()) {
                allPassed = false;
            }
            
            console.log('\n📝 Test 3: Comprehensive Tests');
            if (!await this.runComprehensiveTest()) {
                allPassed = false;
            }
            
            // Results
            console.log('\n🏁 Test Results');
            console.log('===============');
            if (allPassed) {
                console.log('✅ All tests passed! Real model is working correctly.');
                console.log('🎉 BitNet WASM with real model inference is functional!');
            } else {
                console.log('❌ Some tests failed. Check the browser and logs for details.');
            }
            
            return allPassed;
            
        } catch (error) {
            console.error('💥 Test failed with error:', error.message);
            return false;
        } finally {
            await this.cleanup();
        }
    }
}

// Run the test if called directly
if (require.main === module) {
    const tester = new RealModelTester();
    tester.run().then(success => {
        process.exit(success ? 0 : 1);
    }).catch(error => {
        console.error('Fatal error:', error);
        process.exit(1);
    });
}

module.exports = RealModelTester;
