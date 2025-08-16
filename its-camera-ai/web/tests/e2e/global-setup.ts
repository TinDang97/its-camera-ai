import { E2E_CONFIG } from './config/puppeteer.config';
import fs from 'fs';
import path from 'path';

export default async function globalSetup() {
  console.log('🔧 Setting up E2E test environment...');

  // Create directories for test artifacts
  const dirs = [
    E2E_CONFIG.screenshots.path,
    E2E_CONFIG.videos.path,
    './tests/reports',
  ];

  for (const dir of dirs) {
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
      console.log(`📁 Created directory: ${dir}`);
    }
  }

  // Clean up previous test artifacts
  const cleanupDirs = [
    E2E_CONFIG.screenshots.path,
    E2E_CONFIG.videos.path,
  ];

  for (const dir of cleanupDirs) {
    if (fs.existsSync(dir)) {
      const files = fs.readdirSync(dir);
      for (const file of files) {
        const filePath = path.join(dir, file);
        if (fs.statSync(filePath).isFile()) {
          fs.unlinkSync(filePath);
        }
      }
      console.log(`🧹 Cleaned up directory: ${dir}`);
    }
  }

  console.log('✅ E2E test environment setup complete');
}