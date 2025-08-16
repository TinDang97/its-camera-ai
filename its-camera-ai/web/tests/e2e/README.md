# E2E Testing Suite - Visual Regression & Web Vitals

This directory contains comprehensive end-to-end tests including visual regression testing and Web Vitals performance measurement for the ITS Camera AI Web Dashboard.

## 📋 Test Categories

### 1. End-to-End Functional Tests
- **Authentication flows**: Login, logout, session management
- **Camera CRUD operations**: Create, read, update, delete cameras
- **Real-time WebSocket features**: Live data streaming, notifications
- **Analytics dashboards**: All 6 dashboard types with comprehensive functionality testing

### 2. Visual Regression Tests
- **Full page screenshots**: Complete dashboard layouts
- **Component screenshots**: Individual UI components
- **Responsive testing**: Multiple breakpoints (mobile, tablet, desktop)
- **Dark mode testing**: Theme variations
- **Error state testing**: Error and loading states

### 3. Web Vitals Performance Tests
- **Core Web Vitals**: LCP, FID, CLS measurement
- **Additional metrics**: FCP, TTFB, TBT, Speed Index
- **Performance budgets**: Automated threshold checking
- **Real-time monitoring**: Performance over time
- **Mobile performance**: 3G network simulation

## 🚀 Quick Start

### Prerequisites

```bash
# Install dependencies
yarn install

# Ensure test database is running
docker-compose up -d postgres redis
```

### Running Tests

```bash
# Run all E2E tests
yarn test:e2e

# Run tests with browser visible (headed mode)
yarn test:e2e:headed

# Run specific test categories
yarn test:visual              # Visual regression tests
yarn test:performance         # Web Vitals performance tests

# Update visual baselines (when UI changes are intentional)
yarn test:visual:update

# Run tests in Docker (CI environment)
yarn test:e2e:docker
```

### Test Structure

```
tests/e2e/
├── config/                   # Test configuration
│   ├── puppeteer.config.ts   # Puppeteer browser setup
│   ├── jest.setup.js         # Jest test setup
│   └── performance-setup.js  # Performance testing setup
├── fixtures/                 # Test data and mocks
│   └── test-data.ts          # Mock API responses and test users
├── pages/                    # Page Object Model
│   ├── BasePage.ts           # Base page class with common methods
│   ├── LoginPage.ts          # Authentication page objects
│   ├── CameraManagementPage.ts # Camera CRUD page objects
│   ├── AnalyticsDashboardPage.ts # Analytics dashboard page objects
│   └── WebSocketPage.ts      # Real-time features page objects
├── specs/                    # Test specifications
│   ├── auth.test.ts          # Authentication tests
│   ├── camera-crud.test.ts   # Camera management tests
│   ├── websocket-realtime.test.ts # WebSocket tests
│   ├── analytics-dashboards.test.ts # Dashboard tests
│   ├── visual-regression.test.ts # Visual regression tests
│   └── web-vitals.test.ts    # Performance tests
├── utils/                    # Testing utilities
│   ├── visualRegression.ts  # Visual regression testing utilities
│   └── webVitals.ts         # Web Vitals measurement utilities
├── visual-regression/        # Visual regression artifacts
│   ├── baseline/            # Baseline screenshots
│   ├── output/              # Current test screenshots
│   └── diffs/               # Difference images
└── reports/                 # Test reports and artifacts
    ├── performance/         # Performance test reports
    └── visual/              # Visual regression reports
```

## 🖼️ Visual Regression Testing

### How It Works

Visual regression testing captures screenshots of UI components and pages, then compares them pixel-by-pixel with baseline images to detect unexpected visual changes.

### Features

- **Pixel-perfect comparison**: Uses `pixelmatch` for accurate difference detection
- **Responsive testing**: Automatically tests multiple viewport sizes
- **Component isolation**: Test individual components or full pages
- **Baseline management**: Easy baseline creation and updates
- **Difference visualization**: Generated diff images highlight changes
- **Configurable thresholds**: Different sensitivity levels for various UI elements

### Usage Examples

```typescript
// Full page visual test
const result = await visualTester.compareScreenshot(
  'dashboard-overview',
  { threshold: 0.1 }
);

// Component visual test
const result = await visualTester.compareElementScreenshot(
  '[data-testid="kpi-card"]',
  'kpi-cards-component',
  { threshold: 0.05 }
);

// Responsive visual test
const results = await visualTester.compareResponsiveScreenshots(
  'dashboard-responsive',
  [
    { name: 'mobile', width: 375, height: 667 },
    { name: 'tablet', width: 768, height: 1024 },
    { name: 'desktop', width: 1920, height: 1080 },
  ]
);
```

### Managing Baselines

```bash
# Update baselines when UI changes are intentional
yarn test:visual:update

# Update specific test baseline
VISUAL_UPDATE_BASELINE=true yarn test:e2e --testPathPattern=specific-test
```

## ⚡ Web Vitals Performance Testing

### Core Web Vitals Measured

1. **Largest Contentful Paint (LCP)**: Loading performance
   - Good: ≤ 2.5 seconds
   - Needs Improvement: ≤ 4.0 seconds
   - Poor: > 4.0 seconds

2. **First Input Delay (FID)**: Interactivity
   - Good: ≤ 100 milliseconds
   - Needs Improvement: ≤ 300 milliseconds
   - Poor: > 300 milliseconds

3. **Cumulative Layout Shift (CLS)**: Visual stability
   - Good: ≤ 0.1
   - Needs Improvement: ≤ 0.25
   - Poor: > 0.25

### Additional Metrics

- **First Contentful Paint (FCP)**: Time to first content
- **Time to First Byte (TTFB)**: Server response time
- **Total Blocking Time (TBT)**: Main thread blocking time
- **Speed Index**: Visual loading speed
- **Resource metrics**: Count, size, and transfer metrics

### Performance Budgets

```typescript
const performanceBudgets = {
  lcp: 2500,    // 2.5 seconds
  fid: 100,     // 100 milliseconds
  cls: 0.1,     // 0.1 layout shift score
  fcp: 1800,    // 1.8 seconds
  ttfb: 800,    // 800 milliseconds
  tbt: 200,     // 200 milliseconds
};
```

### Usage Examples

```typescript
// Basic performance measurement
const report = await webVitalsTester.generateReport();

// Check performance budgets
const budgetCheck = webVitalsUtils.checkPerformanceBudget(
  report.metrics,
  performanceBudgets
);

// Monitor performance over time
const metrics = await webVitalsTester.monitorRealTimeWebVitals(30000);

// Generate recommendations
const recommendations = webVitalsUtils.generateRecommendations(report);
```

## 🎯 Test Configuration

### Environment Variables

```bash
# Test execution
HEADLESS=false           # Run tests with visible browser
SLOW_MO=100             # Slow down operations by 100ms
BASE_URL=http://localhost:3000  # Application base URL

# Visual regression
VISUAL_UPDATE_BASELINE=true     # Update visual baselines

# Performance testing
PERFORMANCE_THRESHOLD=0.8       # Performance score threshold
NETWORK_PRESET=fast3g          # Network throttling preset
```

### Puppeteer Configuration

```typescript
const config = {
  headless: process.env.HEADLESS !== 'false',
  slowMo: process.env.SLOW_MO ? parseInt(process.env.SLOW_MO) : 0,
  args: [
    '--no-sandbox',
    '--disable-setuid-sandbox',
    '--disable-dev-shm-usage',
    '--disable-web-security',
  ],
  defaultViewport: {
    width: 1920,
    height: 1080,
  },
};
```

## 📊 Test Reports

### Performance Reports

Performance tests generate comprehensive reports including:

- **Metrics summary**: All Web Vitals and additional metrics
- **Score breakdown**: Good/Needs Improvement/Poor classifications
- **Performance recommendations**: Actionable optimization suggestions
- **Historical comparison**: Track performance over time
- **Resource analysis**: Bundle size and resource utilization

### Visual Regression Reports

Visual tests generate:

- **Comparison results**: Pass/fail status with difference percentages
- **Diff images**: Visual highlighting of changes
- **Baseline management**: Easy identification of needed updates
- **Responsive analysis**: Cross-breakpoint consistency

### Report Locations

```
tests/e2e/reports/
├── performance/
│   ├── performance-summary.json
│   └── individual-test-reports.json
├── visual/
│   ├── visual-summary.json
│   └── individual-test-reports.json
└── jest-html-reports/
    ├── jest_html_reporters.html
    └── performance-report.html
```

## 🔧 Troubleshooting

### Common Issues

#### Visual Regression Failures

```bash
# Check diff images
ls tests/e2e/visual-regression/diffs/

# Update baselines after intentional changes
yarn test:visual:update
```

#### Performance Test Failures

```bash
# Check performance reports
cat tests/e2e/reports/performance/performance-summary.json

# Run with network throttling disabled
NETWORK_PRESET=none yarn test:performance
```

#### Browser Launch Issues

```bash
# Install Chromium dependencies (Linux)
sudo apt-get install -y libgbm-dev

# Use system Chrome instead of bundled Chromium
PUPPETEER_EXECUTABLE_PATH=/usr/bin/google-chrome yarn test:e2e
```

### Debug Mode

```bash
# Run tests with debug output
DEBUG=puppeteer:* yarn test:e2e

# Run single test with verbose output
yarn test:e2e --testNamePattern="specific test" --verbose
```

### Memory Issues

```bash
# Increase Node.js memory limit
NODE_OPTIONS="--max_old_space_size=4096" yarn test:e2e

# Run tests with fewer workers
yarn test:e2e --maxWorkers=1
```

## 🏗️ Continuous Integration

### GitHub Actions Integration

```yaml
- name: Run E2E Tests
  run: |
    yarn test:e2e:docker
    
- name: Run Visual Regression Tests
  run: |
    yarn test:visual
    
- name: Run Performance Tests
  run: |
    yarn test:performance
    
- name: Upload Test Artifacts
  uses: actions/upload-artifact@v3
  with:
    name: test-reports
    path: tests/e2e/reports/
```

### Docker Testing

```bash
# Run all tests in Docker environment
docker-compose -f docker-compose.test.yml up --abort-on-container-exit

# Run specific test suite
docker-compose -f docker-compose.test.yml up puppeteer-visual-tests
docker-compose -f docker-compose.test.yml up puppeteer-performance-tests
```

## 📚 Best Practices

### Visual Regression Testing

1. **Stabilize pages**: Hide dynamic content (timestamps, loading indicators)
2. **Consistent environment**: Use mocked API responses
3. **Appropriate thresholds**: Higher for charts/animations, lower for static content
4. **Baseline management**: Regular reviews and updates
5. **Cross-browser testing**: Test in multiple browsers when needed

### Performance Testing

1. **Controlled environment**: Consistent network conditions and hardware
2. **Meaningful metrics**: Focus on user-perceived performance
3. **Budget enforcement**: Fail tests when budgets are exceeded
4. **Continuous monitoring**: Track performance trends over time
5. **Realistic conditions**: Test with representative data sizes

### General E2E Testing

1. **Page Object Model**: Maintain clean, reusable page abstractions
2. **Robust selectors**: Use `data-testid` attributes for reliable element selection
3. **Wait strategies**: Proper waiting for dynamic content
4. **Error handling**: Graceful handling of test failures
5. **Parallel execution**: Optimize test execution time

## 🔗 Related Documentation

- [Jest Puppeteer Documentation](https://github.com/smooth-code/jest-puppeteer)
- [Web Vitals Documentation](https://web.dev/vitals/)
- [Visual Regression Testing Guide](https://percy.io/blog/visual-regression-testing)
- [Puppeteer API Reference](https://pptr.dev/)
- [Testing Library Best Practices](https://testing-library.com/docs/guiding-principles)

For additional support or questions, consult the main project documentation or contact the development team.