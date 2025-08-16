module.exports = {
  preset: 'jest-puppeteer',
  testEnvironment: 'jsdom',
  setupFilesAfterEnv: [
    '<rootDir>/tests/e2e/config/jest.setup.js',
    '@testing-library/jest-dom',
  ],
  testMatch: [
    '**/tests/e2e/specs/**/*.(test|spec).(js|ts)',
  ],
  testPathIgnorePatterns: [
    '<rootDir>/.next/',
    '<rootDir>/node_modules/',
  ],
  transform: {
    '^.+\\.(js|jsx|ts|tsx)$': ['ts-jest', {
      tsconfig: 'tsconfig.json',
    }],
  },
  moduleNameMapping: {
    '^@/(.*)$': '<rootDir>/$1',
    '^@/tests/(.*)$': '<rootDir>/tests/$1',
  },
  testTimeout: 60000,
  maxWorkers: 2,
  collectCoverage: false,
  verbose: true,
  reporters: [
    'default',
    ['jest-html-reporters', {
      publicPath: './tests/e2e/reports',
      filename: 'performance-report.html',
      expand: true,
      hideIcon: true,
      pageTitle: 'Web Vitals Performance Test Report',
    }],
    ['jest-junit', {
      outputDirectory: './tests/e2e/reports',
      outputName: 'performance-junit.xml',
      suiteName: 'Web Vitals Performance Tests',
    }],
  ],
  globals: {
    'ts-jest': {
      tsconfig: 'tsconfig.json',
    },
  },
  setupFilesAfterEnv: [
    '<rootDir>/tests/e2e/config/performance-setup.js',
  ],
};