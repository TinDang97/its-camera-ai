export default async function globalTeardown() {
  console.log('🧹 Tearing down E2E test environment...');

  // Cleanup any global resources if needed
  // For example, stopping mock servers, cleaning up test data, etc.

  console.log('✅ E2E test environment teardown complete');
}