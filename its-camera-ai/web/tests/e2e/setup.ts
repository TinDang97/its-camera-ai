import { E2E_CONFIG } from './config/puppeteer.config';

// Global test configuration
jest.setTimeout(60000);

// Setup test environment
beforeAll(async () => {
  console.log('🚀 Starting E2E test suite...');
  console.log(`Base URL: ${E2E_CONFIG.baseUrl}`);
  console.log(`Headless: ${E2E_CONFIG.headless}`);
});

// Global teardown
afterAll(async () => {
  console.log('✅ E2E test suite completed');
});

// Export E2E configuration for use in tests
export { E2E_CONFIG };