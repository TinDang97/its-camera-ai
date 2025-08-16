// Performance testing setup
const fs = require('fs');
const path = require('path');

// Create performance reports directory
const reportsDir = path.join(__dirname, '../reports/performance');
if (!fs.existsSync(reportsDir)) {
  fs.mkdirSync(reportsDir, { recursive: true });
}

// Create visual regression directories
const visualDir = path.join(__dirname, '../visual-regression');
const baselineDir = path.join(visualDir, 'baseline');
const outputDir = path.join(visualDir, 'output');
const diffDir = path.join(visualDir, 'diffs');

[visualDir, baselineDir, outputDir, diffDir].forEach(dir => {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
});

// Global test configuration
global.testConfig = {
  baseUrl: process.env.BASE_URL || 'http://localhost:3000',
  headless: process.env.HEADLESS !== 'false',
  slowMo: process.env.SLOW_MO ? parseInt(process.env.SLOW_MO) : 0,
  timeout: 60000,
  visualUpdate: process.env.VISUAL_UPDATE_BASELINE === 'true',
};

// Performance thresholds
global.performanceThresholds = {
  // Core Web Vitals (Google recommendations)
  lcp: { good: 2500, needsImprovement: 4000 }, // Largest Contentful Paint (ms)
  fid: { good: 100, needsImprovement: 300 },   // First Input Delay (ms)
  cls: { good: 0.1, needsImprovement: 0.25 },  // Cumulative Layout Shift (unitless)

  // Additional metrics
  fcp: { good: 1800, needsImprovement: 3000 }, // First Contentful Paint (ms)
  ttfb: { good: 800, needsImprovement: 1800 }, // Time to First Byte (ms)
  tbt: { good: 200, needsImprovement: 600 },   // Total Blocking Time (ms)

  // Resource thresholds
  maxResources: 100,
  maxTransferSize: 3 * 1024 * 1024, // 3MB
  maxJavaScriptSize: 1 * 1024 * 1024, // 1MB
};

// Visual regression thresholds
global.visualThresholds = {
  default: 0.1,      // 0.1% difference allowed
  charts: 0.15,      // Charts may have minor rendering differences
  responsive: 0.3,   // Responsive layouts may have more variation
  darkMode: 0.3,     // Theme changes cause significant differences
};

// Jest setup for performance testing
beforeAll(async () => {
  // Set longer timeout for performance tests
  jest.setTimeout(120000);

  // Clear any existing performance reports
  const perfReportsDir = path.join(__dirname, '../reports/performance');
  if (fs.existsSync(perfReportsDir)) {
    const files = fs.readdirSync(perfReportsDir);
    files.forEach(file => {
      if (file.endsWith('.json')) {
        fs.unlinkSync(path.join(perfReportsDir, file));
      }
    });
  }
});

afterAll(async () => {
  // Generate performance summary report
  const perfReportsDir = path.join(__dirname, '../reports/performance');
  const summaryPath = path.join(perfReportsDir, 'performance-summary.json');

  const summary = {
    timestamp: new Date().toISOString(),
    testRun: process.env.GITHUB_RUN_ID || 'local',
    browser: 'chrome',
    environment: process.env.NODE_ENV || 'test',
    thresholds: global.performanceThresholds,
    completed: true,
  };

  fs.writeFileSync(summaryPath, JSON.stringify(summary, null, 2));
  console.log(`Performance summary written to: ${summaryPath}`);
});

// Export utilities for tests
module.exports = {
  createPerformanceReport: (testName, metrics, scores) => {
    const reportsDir = path.join(__dirname, '../reports/performance');
    const reportPath = path.join(reportsDir, `${testName}-${Date.now()}.json`);

    const report = {
      testName,
      timestamp: new Date().toISOString(),
      metrics,
      scores,
      thresholds: global.performanceThresholds,
      passed: scores.overallScore !== 'poor',
    };

    fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
    return reportPath;
  },

  createVisualReport: (testName, results) => {
    const reportsDir = path.join(__dirname, '../reports/visual');
    if (!fs.existsSync(reportsDir)) {
      fs.mkdirSync(reportsDir, { recursive: true });
    }

    const reportPath = path.join(reportsDir, `${testName}-${Date.now()}.json`);

    const report = {
      testName,
      timestamp: new Date().toISOString(),
      results,
      thresholds: global.visualThresholds,
    };

    fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
    return reportPath;
  },
};