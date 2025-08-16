const { E2E_CONFIG } = require('./tests/e2e/config/puppeteer.config.ts');

module.exports = {
  launch: {
    ...E2E_CONFIG.browser,
    defaultViewport: E2E_CONFIG.viewport,
  },
  browserContext: 'default',
  server: process.env.CI ? undefined : {
    command: 'yarn dev --port 3002',
    port: 3002,
    launchTimeout: 60000,
    debug: true,
  },
};