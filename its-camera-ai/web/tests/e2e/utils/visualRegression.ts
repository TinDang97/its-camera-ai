import { Page } from 'puppeteer';
import * as fs from 'fs/promises';
import * as path from 'path';
import { PNG } from 'pngjs';
import pixelmatch from 'pixelmatch';

export interface VisualRegressionOptions {
  threshold?: number;
  includeAA?: boolean;
  alpha?: number;
  aaColor?: [number, number, number];
  diffColor?: [number, number, number];
  diffMask?: boolean;
}

export interface VisualRegressionResult {
  passed: boolean;
  pixelDifference: number;
  percentageDifference: number;
  diffImagePath?: string;
  baselineImagePath: string;
  currentImagePath: string;
}

export class VisualRegressionTester {
  private page: Page;
  private baselineDir: string;
  private outputDir: string;
  private diffDir: string;

  constructor(page: Page, baseDir: string = './tests/e2e/visual-regression') {
    this.page = page;
    this.baselineDir = path.join(baseDir, 'baseline');
    this.outputDir = path.join(baseDir, 'output');
    this.diffDir = path.join(baseDir, 'diffs');
  }

  /**
   * Initialize directories for visual regression testing
   */
  async initializeDirectories(): Promise<void> {
    await this.ensureDirectoryExists(this.baselineDir);
    await this.ensureDirectoryExists(this.outputDir);
    await this.ensureDirectoryExists(this.diffDir);
  }

  /**
   * Take a screenshot and compare with baseline
   */
  async compareScreenshot(
    testName: string,
    options: VisualRegressionOptions = {}
  ): Promise<VisualRegressionResult> {
    await this.initializeDirectories();

    const sanitizedTestName = this.sanitizeFilename(testName);
    const baselineImagePath = path.join(this.baselineDir, `${sanitizedTestName}.png`);
    const currentImagePath = path.join(this.outputDir, `${sanitizedTestName}.png`);
    const diffImagePath = path.join(this.diffDir, `${sanitizedTestName}-diff.png`);

    // Take current screenshot
    await this.page.screenshot({
      path: currentImagePath,
      fullPage: true,
      type: 'png',
    });

    // Check if baseline exists
    const baselineExists = await this.fileExists(baselineImagePath);

    if (!baselineExists) {
      // Copy current as baseline for first run
      await fs.copyFile(currentImagePath, baselineImagePath);
      return {
        passed: true,
        pixelDifference: 0,
        percentageDifference: 0,
        baselineImagePath,
        currentImagePath,
      };
    }

    // Compare images
    return await this.compareImages(
      baselineImagePath,
      currentImagePath,
      diffImagePath,
      options
    );
  }

  /**
   * Compare element screenshot with baseline
   */
  async compareElementScreenshot(
    selector: string,
    testName: string,
    options: VisualRegressionOptions = {}
  ): Promise<VisualRegressionResult> {
    await this.initializeDirectories();

    const element = await this.page.$(selector);
    if (!element) {
      throw new Error(`Element with selector "${selector}" not found`);
    }

    const sanitizedTestName = this.sanitizeFilename(`${testName}-element`);
    const baselineImagePath = path.join(this.baselineDir, `${sanitizedTestName}.png`);
    const currentImagePath = path.join(this.outputDir, `${sanitizedTestName}.png`);
    const diffImagePath = path.join(this.diffDir, `${sanitizedTestName}-diff.png`);

    // Take element screenshot
    await element.screenshot({
      path: currentImagePath,
      type: 'png',
    });

    // Check if baseline exists
    const baselineExists = await this.fileExists(baselineImagePath);

    if (!baselineExists) {
      // Copy current as baseline for first run
      await fs.copyFile(currentImagePath, baselineImagePath);
      return {
        passed: true,
        pixelDifference: 0,
        percentageDifference: 0,
        baselineImagePath,
        currentImagePath,
      };
    }

    // Compare images
    return await this.compareImages(
      baselineImagePath,
      currentImagePath,
      diffImagePath,
      options
    );
  }

  /**
   * Take screenshot at multiple breakpoints
   */
  async compareResponsiveScreenshots(
    testName: string,
    breakpoints: Array<{ name: string; width: number; height: number }> = [
      { name: 'mobile', width: 375, height: 667 },
      { name: 'tablet', width: 768, height: 1024 },
      { name: 'desktop', width: 1920, height: 1080 },
    ],
    options: VisualRegressionOptions = {}
  ): Promise<Record<string, VisualRegressionResult>> {
    const results: Record<string, VisualRegressionResult> = {};

    for (const breakpoint of breakpoints) {
      // Set viewport
      await this.page.setViewport({
        width: breakpoint.width,
        height: breakpoint.height,
      });

      // Wait for responsive changes to apply
      await this.page.waitForTimeout(500);

      // Compare screenshot
      const result = await this.compareScreenshot(
        `${testName}-${breakpoint.name}`,
        options
      );

      results[breakpoint.name] = result;
    }

    // Reset to default viewport
    await this.page.setViewport({ width: 1920, height: 1080 });

    return results;
  }

  /**
   * Update baseline screenshot
   */
  async updateBaseline(testName: string): Promise<void> {
    await this.initializeDirectories();

    const sanitizedTestName = this.sanitizeFilename(testName);
    const currentImagePath = path.join(this.outputDir, `${sanitizedTestName}.png`);
    const baselineImagePath = path.join(this.baselineDir, `${sanitizedTestName}.png`);

    const currentExists = await this.fileExists(currentImagePath);
    if (!currentExists) {
      throw new Error(`Current screenshot not found: ${currentImagePath}`);
    }

    // Copy current to baseline
    await fs.copyFile(currentImagePath, baselineImagePath);
  }

  /**
   * Clean up old diff images
   */
  async cleanupDiffs(): Promise<void> {
    try {
      const files = await fs.readdir(this.diffDir);
      const diffFiles = files.filter(file => file.endsWith('-diff.png'));

      for (const file of diffFiles) {
        await fs.unlink(path.join(this.diffDir, file));
      }
    } catch (error) {
      // Directory might not exist, which is fine
    }
  }

  /**
   * Compare two images and generate diff
   */
  private async compareImages(
    baselinePath: string,
    currentPath: string,
    diffPath: string,
    options: VisualRegressionOptions = {}
  ): Promise<VisualRegressionResult> {
    const defaultOptions: Required<VisualRegressionOptions> = {
      threshold: 0.1,
      includeAA: false,
      alpha: 0.1,
      aaColor: [255, 255, 0],
      diffColor: [255, 0, 255],
      diffMask: false,
    };

    const opts = { ...defaultOptions, ...options };

    // Read images
    const baselineBuffer = await fs.readFile(baselinePath);
    const currentBuffer = await fs.readFile(currentPath);

    const baselineImg = PNG.sync.read(baselineBuffer);
    const currentImg = PNG.sync.read(currentBuffer);

    // Check dimensions match
    if (
      baselineImg.width !== currentImg.width ||
      baselineImg.height !== currentImg.height
    ) {
      throw new Error(
        `Image dimensions don't match. Baseline: ${baselineImg.width}x${baselineImg.height}, Current: ${currentImg.width}x${currentImg.height}`
      );
    }

    // Create diff image
    const { width, height } = baselineImg;
    const diff = new PNG({ width, height });

    // Compare pixels
    const pixelDifference = pixelmatch(
      baselineImg.data,
      currentImg.data,
      diff.data,
      width,
      height,
      opts
    );

    const totalPixels = width * height;
    const percentageDifference = (pixelDifference / totalPixels) * 100;

    // Save diff image if there are differences
    let diffImagePath: string | undefined;
    if (pixelDifference > 0) {
      diffImagePath = diffPath;
      await fs.writeFile(diffImagePath, PNG.sync.write(diff));
    }

    const passed = percentageDifference <= opts.threshold;

    return {
      passed,
      pixelDifference,
      percentageDifference,
      diffImagePath,
      baselineImagePath: baselinePath,
      currentImagePath,
    };
  }

  /**
   * Ensure directory exists
   */
  private async ensureDirectoryExists(dirPath: string): Promise<void> {
    try {
      await fs.access(dirPath);
    } catch {
      await fs.mkdir(dirPath, { recursive: true });
    }
  }

  /**
   * Check if file exists
   */
  private async fileExists(filePath: string): Promise<boolean> {
    try {
      await fs.access(filePath);
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Sanitize filename for filesystem
   */
  private sanitizeFilename(filename: string): string {
    return filename
      .replace(/[^a-z0-9]/gi, '-')
      .replace(/-+/g, '-')
      .replace(/^-|-$/g, '')
      .toLowerCase();
  }
}

/**
 * Visual regression testing utilities
 */
export const visualRegressionUtils = {
  /**
   * Wait for all images to load
   */
  async waitForImages(page: Page): Promise<void> {
    await page.waitForFunction(() => {
      const images = Array.from(document.querySelectorAll('img'));
      return images.every(img => img.complete && img.naturalHeight !== 0);
    });
  },

  /**
   * Wait for animations to complete
   */
  async waitForAnimations(page: Page): Promise<void> {
    await page.waitForFunction(() => {
      const animations = document.getAnimations();
      return animations.every(animation => animation.playState === 'finished');
    });
  },

  /**
   * Hide dynamic content for consistent screenshots
   */
  async hideDynamicContent(page: Page): Promise<void> {
    await page.addStyleTag({
      content: `
        [data-testid="last-updated"],
        [data-testid="timestamp"],
        [data-testid="current-time"],
        .live-indicator,
        .timestamp,
        .loading-animation {
          visibility: hidden !important;
        }
      `,
    });
  },

  /**
   * Stabilize page for screenshot
   */
  async stabilizePage(page: Page): Promise<void> {
    await this.waitForImages(page);
    await this.waitForAnimations(page);
    await this.hideDynamicContent(page);
    await page.waitForTimeout(1000); // Additional stability wait
  },
};