#!/usr/bin/env node

/**
 * Bundle Size Analysis Script for Next.js Application
 *
 * This script analyzes the bundle size after build and:
 * - Checks against size limits
 * - Compares with previous builds
 * - Identifies large chunks
 * - Provides optimization suggestions
 */

const fs = require('fs');
const path = require('path');

// Configuration
const MAX_BUNDLE_SIZE = 500 * 1024; // 500KB
const MAX_INCREASE_PERCENT = 5; // 5% increase threshold
const CHUNK_SIZE_WARNING = 100 * 1024; // 100KB per chunk warning
const STATS_FILE = path.join(__dirname, '..', '.next', 'bundle-stats.json');
const BUILD_MANIFEST = path.join(__dirname, '..', '.next', 'build-manifest.json');
const STATIC_DIR = path.join(__dirname, '..', '.next', 'static');

// ANSI color codes
const colors = {
  reset: '\x1b[0m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  cyan: '\x1b[36m',
  magenta: '\x1b[35m',
};

function colorize(text, color) {
  return `${colors[color]}${text}${colors.reset}`;
}

function logError(message) {
  console.error(colorize(`❌ ${message}`, 'red'));
}

function logWarning(message) {
  console.warn(colorize(`⚠️  ${message}`, 'yellow'));
}

function logSuccess(message) {
  console.log(colorize(`✅ ${message}`, 'green'));
}

function logInfo(message) {
  console.log(colorize(`ℹ️  ${message}`, 'blue'));
}

function formatSize(bytes) {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
}

/**
 * Get file size for a given path
 */
function getFileSize(filePath) {
  try {
    const stats = fs.statSync(filePath);
    return stats.size;
  } catch (error) {
    return 0;
  }
}

/**
 * Analyze build manifest to get chunk information
 */
function analyzeBuildManifest() {
  if (!fs.existsSync(BUILD_MANIFEST)) {
    logError('Build manifest not found. Run `npm run build` first.');
    process.exit(1);
  }

  let manifest;
  try {
    manifest = JSON.parse(fs.readFileSync(BUILD_MANIFEST, 'utf8'));
  } catch (error) {
    logError(`Error reading build manifest: ${error.message}`);
    process.exit(1);
  }

  const chunks = [];
  let totalSize = 0;

  // Analyze pages
  if (manifest.pages) {
    Object.entries(manifest.pages).forEach(([page, assets]) => {
      assets.forEach(asset => {
        const assetPath = path.join(__dirname, '..', '.next', asset);
        const size = getFileSize(assetPath);

        if (size > 0) {
          chunks.push({
            name: asset,
            page,
            size,
            type: 'page',
          });
          totalSize += size;
        }
      });
    });
  }

  // Analyze shared chunks
  if (manifest.sharedFiles) {
    manifest.sharedFiles.forEach(asset => {
      const assetPath = path.join(__dirname, '..', '.next', asset);
      const size = getFileSize(assetPath);

      if (size > 0) {
        chunks.push({
          name: asset,
          page: 'shared',
          size,
          type: 'shared',
        });
        totalSize += size;
      }
    });
  }

  return { chunks, totalSize };
}

/**
 * Analyze static assets
 */
function analyzeStaticAssets() {
  const staticAssets = [];
  let totalStaticSize = 0;

  if (!fs.existsSync(STATIC_DIR)) {
    return { staticAssets, totalStaticSize };
  }

  function walkDirectory(dir, relativePath = '') {
    try {
      const files = fs.readdirSync(dir);

      files.forEach(file => {
        const filePath = path.join(dir, file);
        const relativeFilePath = path.join(relativePath, file);
        const stats = fs.statSync(filePath);

        if (stats.isDirectory()) {
          walkDirectory(filePath, relativeFilePath);
        } else {
          staticAssets.push({
            name: relativeFilePath,
            size: stats.size,
            type: path.extname(file).slice(1) || 'unknown',
          });
          totalStaticSize += stats.size;
        }
      });
    } catch (error) {
      // Ignore directory read errors
    }
  }

  walkDirectory(STATIC_DIR);

  return { staticAssets, totalStaticSize };
}

/**
 * Load previous bundle statistics
 */
function loadPreviousStats() {
  if (!fs.existsSync(STATS_FILE)) {
    return null;
  }

  try {
    return JSON.parse(fs.readFileSync(STATS_FILE, 'utf8'));
  } catch (error) {
    logWarning(`Error reading previous stats: ${error.message}`);
    return null;
  }
}

/**
 * Save current bundle statistics
 */
function saveCurrentStats(stats) {
  try {
    fs.writeFileSync(STATS_FILE, JSON.stringify(stats, null, 2));
  } catch (error) {
    logWarning(`Error saving stats: ${error.message}`);
  }
}

/**
 * Analyze bundle size and provide insights
 */
function analyzeBundleSize() {
  logInfo('📦 Analyzing bundle size...');

  const { chunks, totalSize } = analyzeBuildManifest();
  const { staticAssets, totalStaticSize } = analyzeStaticAssets();
  const previousStats = loadPreviousStats();

  console.log('\n' + '='.repeat(60));
  logInfo('Bundle Size Analysis Results');
  console.log('='.repeat(60));

  // Total bundle size
  console.log(`\nTotal JavaScript Bundle: ${colorize(formatSize(totalSize), 'cyan')}`);
  console.log(`Total Static Assets: ${colorize(formatSize(totalStaticSize), 'cyan')}`);
  console.log(`Combined Total: ${colorize(formatSize(totalSize + totalStaticSize), 'cyan')}`);

  // Check against limits
  let hasErrors = false;

  if (totalSize > MAX_BUNDLE_SIZE) {
    logError(`Bundle size (${formatSize(totalSize)}) exceeds limit (${formatSize(MAX_BUNDLE_SIZE)})`);
    hasErrors = true;
  } else {
    logSuccess(`Bundle size is within limits (${formatSize(MAX_BUNDLE_SIZE)})`);
  }

  // Compare with previous build
  if (previousStats) {
    const sizeDiff = totalSize - previousStats.totalSize;
    const percentChange = ((sizeDiff / previousStats.totalSize) * 100);

    console.log(`\nComparison with previous build:`);

    if (sizeDiff > 0) {
      const changeText = `+${formatSize(sizeDiff)} (+${percentChange.toFixed(2)}%)`;

      if (percentChange > MAX_INCREASE_PERCENT) {
        logError(`Bundle size increased by ${changeText} (limit: ${MAX_INCREASE_PERCENT}%)`);
        hasErrors = true;
      } else {
        logWarning(`Bundle size increased by ${changeText}`);
      }
    } else if (sizeDiff < 0) {
      logSuccess(`Bundle size decreased by ${formatSize(Math.abs(sizeDiff))} (${Math.abs(percentChange).toFixed(2)}%)`);
    } else {
      logInfo('Bundle size unchanged');
    }
  }

  // Analyze largest chunks
  console.log('\n📊 Largest JavaScript Chunks:');
  const sortedChunks = chunks.sort((a, b) => b.size - a.size).slice(0, 10);

  sortedChunks.forEach((chunk, index) => {
    const sizeColor = chunk.size > CHUNK_SIZE_WARNING ? 'red' : 'cyan';
    const sizeText = colorize(formatSize(chunk.size), sizeColor);
    console.log(`  ${index + 1}. ${chunk.name} (${chunk.page}) - ${sizeText}`);

    if (chunk.size > CHUNK_SIZE_WARNING) {
      console.log(`     ${colorize('⚠️  Large chunk detected', 'yellow')}`);
    }
  });

  // Analyze static assets
  if (staticAssets.length > 0) {
    console.log('\n📁 Largest Static Assets:');
    const sortedAssets = staticAssets.sort((a, b) => b.size - a.size).slice(0, 5);

    sortedAssets.forEach((asset, index) => {
      const sizeText = colorize(formatSize(asset.size), 'cyan');
      console.log(`  ${index + 1}. ${asset.name} (${asset.type}) - ${sizeText}`);
    });
  }

  // Asset type breakdown
  const assetTypes = {};
  staticAssets.forEach(asset => {
    assetTypes[asset.type] = (assetTypes[asset.type] || 0) + asset.size;
  });

  if (Object.keys(assetTypes).length > 0) {
    console.log('\n📈 Asset Type Breakdown:');
    Object.entries(assetTypes)
      .sort(([,a], [,b]) => b - a)
      .forEach(([type, size]) => {
        console.log(`  ${type}: ${colorize(formatSize(size), 'cyan')}`);
      });
  }

  // Optimization suggestions
  console.log('\n💡 Optimization Suggestions:');

  const largeChunks = chunks.filter(chunk => chunk.size > CHUNK_SIZE_WARNING);
  if (largeChunks.length > 0) {
    console.log('• Consider code splitting for large chunks:');
    largeChunks.slice(0, 3).forEach(chunk => {
      console.log(`  - ${chunk.name} (${formatSize(chunk.size)})`);
    });
  }

  const largeAssets = staticAssets.filter(asset => asset.size > 50 * 1024);
  if (largeAssets.length > 0) {
    console.log('• Optimize large static assets:');
    largeAssets.slice(0, 3).forEach(asset => {
      if (asset.type === 'png' || asset.type === 'jpg' || asset.type === 'jpeg') {
        console.log(`  - Compress image: ${asset.name} (${formatSize(asset.size)})`);
      } else {
        console.log(`  - Review asset: ${asset.name} (${formatSize(asset.size)})`);
      }
    });
  }

  console.log('• General optimization tips:');
  console.log('  - Use dynamic imports for code splitting');
  console.log('  - Optimize images with Next.js Image component');
  console.log('  - Remove unused dependencies');
  console.log('  - Use tree shaking for library imports');
  console.log('  - Consider lazy loading for non-critical components');

  // Save current stats
  const currentStats = {
    totalSize,
    totalStaticSize,
    timestamp: new Date().toISOString(),
    chunks: chunks.slice(0, 20), // Keep top 20 chunks
    staticAssets: staticAssets.slice(0, 10), // Keep top 10 assets
  };

  saveCurrentStats(currentStats);

  console.log('\n' + '='.repeat(60));

  if (hasErrors) {
    logError('Bundle size check failed! Address the issues above.');
    process.exit(1);
  } else {
    logSuccess('Bundle size check passed!');
  }
}

// Handle command line execution
if (require.main === module) {
  analyzeBundleSize();
}

module.exports = {
  analyzeBundleSize,
  analyzeBuildManifest,
  analyzeStaticAssets,
  formatSize,
};
