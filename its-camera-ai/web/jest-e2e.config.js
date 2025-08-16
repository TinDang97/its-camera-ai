module.exports = {
  preset: 'jest-puppeteer',
  testEnvironment: 'node',
  testMatch: ['**/tests/e2e/specs/**/*.test.ts'],
  setupFilesAfterEnv: ['<rootDir>/tests/e2e/setup.ts'],
  transform: {
    '^.+\\.ts$': 'ts-jest',
  },
  moduleFileExtensions: ['ts', 'tsx', 'js', 'jsx', 'json'],
  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/$1',
  },
  testTimeout: 60000,
  maxWorkers: process.env.CI ? 2 : 4,
  reporters: [
    'default',
    [
      'jest-html-reporters',
      {
        publicPath: './tests/reports',
        filename: 'e2e-report.html',
        expand: true,
      },
    ],
    [
      'jest-junit',
      {
        outputDirectory: './tests/reports',
        outputName: 'e2e-results.xml',
      },
    ],
  ],
  coverageDirectory: './tests/reports/coverage',
  collectCoverageFrom: [
    'pages/**/*.{ts,tsx}',
    'components/**/*.{ts,tsx}',
    'app/**/*.{ts,tsx}',
    '!**/*.d.ts',
    '!**/node_modules/**',
    '!**/.next/**',
  ],
  coverageReporters: ['text', 'lcov', 'html'],
  globalSetup: '<rootDir>/tests/e2e/global-setup.ts',
  globalTeardown: '<rootDir>/tests/e2e/global-teardown.ts',
};