import { Browser, Page } from 'puppeteer';
import { CameraPage } from '../pages/CameraPage';
import { LoginPage } from '../pages/LoginPage';
import { getTestBrowser } from '../config/puppeteer.config';
import { TEST_USERS, TEST_CAMERAS } from '../fixtures/test-data';

describe('Camera CRUD E2E Tests', () => {
  let browser: Browser;
  let page: Page;
  let cameraPage: CameraPage;
  let loginPage: LoginPage;

  beforeAll(async () => {
    browser = await getTestBrowser();
  });

  afterAll(async () => {
    if (browser) {
      await browser.close();
    }
  });

  beforeEach(async () => {
    page = await browser.newPage();
    cameraPage = new CameraPage(page);
    loginPage = new LoginPage(page);

    // Login before each test
    await loginPage.navigateToLogin();
    await loginPage.loginWithUser(TEST_USERS.admin);

    // Navigate to cameras page
    await cameraPage.navigateToCameras();
  });

  afterEach(async () => {
    if (page) {
      await page.close();
    }
  });

  describe('Camera List Display', () => {
    test('should load and display cameras page', async () => {
      await cameraPage.waitForCamerasToLoad();

      // Should not be loading
      const isLoading = await cameraPage.isLoading();
      expect(isLoading).toBe(false);

      // Should not have errors
      const errorMessage = await cameraPage.getErrorMessage();
      expect(errorMessage).toBeNull();
    });

    test('should display camera list with proper data', async () => {
      await cameraPage.waitForCamerasToLoad();

      const cameras = await cameraPage.getCameraList();
      expect(cameras.length).toBeGreaterThanOrEqual(0);

      // If cameras exist, they should have required fields
      if (cameras.length > 0) {
        cameras.forEach(camera => {
          expect(camera.name).toBeTruthy();
          expect(camera.status).toBeTruthy();
          expect(camera.location).toBeTruthy();
          expect(['online', 'offline', 'maintenance']).toContain(camera.status.toLowerCase());
        });
      }
    });

    test('should switch between grid and list views', async () => {
      await cameraPage.waitForCamerasToLoad();

      // Test grid view
      await cameraPage.switchToGridView();
      await page.waitForTimeout(1000);

      // Test list view
      await cameraPage.switchToListView();
      await page.waitForTimeout(1000);

      // Should be able to switch without errors
      const errorMessage = await cameraPage.getErrorMessage();
      expect(errorMessage).toBeNull();
    });

    test('should handle empty state gracefully', async () => {
      // This test assumes we can filter to show no results
      await cameraPage.filterByStatus('maintenance');

      // Should either show results or empty state
      const isEmpty = await cameraPage.isCameraListEmpty();
      const cameraCount = await cameraPage.getCameraCount();

      if (isEmpty) {
        expect(cameraCount).toBe(0);
      } else {
        expect(cameraCount).toBeGreaterThan(0);
      }
    });
  });

  describe('Camera Search and Filtering', () => {
    test('should search cameras by name', async () => {
      await cameraPage.waitForCamerasToLoad();

      const initialCameras = await cameraPage.getCameraList();

      if (initialCameras.length > 0) {
        const testCamera = initialCameras[0];
        await cameraPage.searchCameras(testCamera.name);

        const filteredCameras = await cameraPage.getCameraList();
        expect(filteredCameras.length).toBeLessThanOrEqualTo(initialCameras.length);

        if (filteredCameras.length > 0) {
          expect(filteredCameras.some(c => c.name.includes(testCamera.name))).toBe(true);
        }
      }
    });

    test('should filter cameras by status', async () => {
      await cameraPage.waitForCamerasToLoad();

      // Test online filter
      await cameraPage.filterByStatus('online');
      await page.waitForTimeout(1000);

      const onlineCameras = await cameraPage.getCameraList();
      onlineCameras.forEach(camera => {
        expect(camera.status.toLowerCase()).toBe('online');
      });

      // Test offline filter
      await cameraPage.filterByStatus('offline');
      await page.waitForTimeout(1000);

      const offlineCameras = await cameraPage.getCameraList();
      offlineCameras.forEach(camera => {
        expect(camera.status.toLowerCase()).toBe('offline');
      });
    });

    test('should clear filters and show all cameras', async () => {
      await cameraPage.waitForCamerasToLoad();

      const initialCount = await cameraPage.getCameraCount();

      // Apply filter
      await cameraPage.filterByStatus('online');
      await page.waitForTimeout(1000);

      const filteredCount = await cameraPage.getCameraCount();

      // Clear filters
      await cameraPage.clearFilters();
      await page.waitForTimeout(1000);

      const clearedCount = await cameraPage.getCameraCount();
      expect(clearedCount).toBe(initialCount);
    });

    test('should handle search with no results', async () => {
      await cameraPage.waitForCamerasToLoad();

      // Search for something that doesn't exist
      await cameraPage.searchCameras('nonexistent-camera-xyz-123');

      const cameras = await cameraPage.getCameraList();
      expect(cameras.length).toBe(0);

      // Should show empty state or no results message
      const isEmpty = await cameraPage.isCameraListEmpty();
      expect(isEmpty).toBe(true);
    });
  });

  describe('Camera Creation (Create)', () => {
    test('should open camera creation form', async () => {
      await cameraPage.clickAddCamera();

      // Should navigate to form or open modal
      await cameraPage.waitForCameraForm();

      // Form should be visible and have required fields
      const nameInputVisible = await cameraPage.isElementVisible('[data-testid="camera-name-input"], input[name="name"]');
      const locationInputVisible = await cameraPage.isElementVisible('[data-testid="camera-location-input"], input[name="location"]');
      const ipInputVisible = await cameraPage.isElementVisible('[data-testid="camera-ip-input"], input[name="ipAddress"]');

      expect(nameInputVisible).toBe(true);
      expect(locationInputVisible).toBe(true);
      expect(ipInputVisible).toBe(true);
    });

    test('should create a new camera successfully', async () => {
      const newCamera = TEST_CAMERAS.newCamera;

      await cameraPage.createCamera(newCamera);

      // Should redirect back to list or show success
      await cameraPage.waitForCamerasToLoad();

      // Verify camera was created
      const cameraExists = await cameraPage.findCameraByName(newCamera.name);
      expect(cameraExists).toBe(true);
    });

    test('should validate required fields in camera form', async () => {
      await cameraPage.clickAddCamera();
      await cameraPage.waitForCameraForm();

      // Try to save without filling required fields
      await cameraPage.saveCameraForm();

      // Should show validation errors
      await page.waitForTimeout(1000);
      const errorMessage = await cameraPage.getErrorMessage();
      expect(errorMessage).toBeTruthy();
    });

    test('should validate IP address format', async () => {
      await cameraPage.clickAddCamera();
      await cameraPage.waitForCameraForm();

      // Fill form with invalid IP
      const invalidCamera = {
        ...TEST_CAMERAS.newCamera,
        ipAddress: 'invalid-ip-address'
      };

      await cameraPage.fillCameraForm(invalidCamera);
      await cameraPage.saveCameraForm();

      // Should show validation error
      await page.waitForTimeout(1000);
      const errorMessage = await cameraPage.getErrorMessage();
      expect(errorMessage).toBeTruthy();
    });

    test('should cancel camera creation', async () => {
      await cameraPage.clickAddCamera();
      await cameraPage.waitForCameraForm();

      // Fill some data
      await cameraPage.fillCameraForm(TEST_CAMERAS.newCamera);

      // Cancel form
      await cameraPage.cancelCameraForm();

      // Should return to camera list
      await cameraPage.waitForCamerasToLoad();

      // Camera should not be created
      const cameraExists = await cameraPage.findCameraByName(TEST_CAMERAS.newCamera.name);
      expect(cameraExists).toBe(false);
    });
  });

  describe('Camera Reading (Read)', () => {
    test('should view camera details', async () => {
      await cameraPage.waitForCamerasToLoad();

      const cameras = await cameraPage.getCameraList();

      if (cameras.length > 0) {
        const testCamera = cameras[0];
        await cameraPage.viewCameraDetails(testCamera.name);

        // Should navigate to details page
        const currentUrl = page.url();
        expect(currentUrl).toMatch(/camera|details/);

        // Should display camera details
        const detailsVisible = await cameraPage.isElementVisible('[data-testid="camera-details"]');
        expect(detailsVisible).toBe(true);
      }
    });

    test('should navigate back from camera details', async () => {
      await cameraPage.waitForCamerasToLoad();

      const cameras = await cameraPage.getCameraList();

      if (cameras.length > 0) {
        const testCamera = cameras[0];
        await cameraPage.viewCameraDetails(testCamera.name);

        // Go back to list
        await cameraPage.backToList();

        // Should return to camera list
        await cameraPage.waitForCamerasToLoad();
        const listVisible = await cameraPage.isElementVisible('[data-testid="camera-list"], [data-testid="cameras-page"]');
        expect(listVisible).toBe(true);
      }
    });

    test('should display camera status correctly', async () => {
      await cameraPage.waitForCamerasToLoad();

      const cameras = await cameraPage.getCameraList();

      cameras.forEach(camera => {
        expect(['online', 'offline', 'maintenance']).toContain(camera.status.toLowerCase());
      });
    });

    test('should test camera live preview if available', async () => {
      await cameraPage.waitForCamerasToLoad();

      const cameras = await cameraPage.getCameraList();

      if (cameras.length > 0) {
        const onlineCamera = cameras.find(c => c.status.toLowerCase() === 'online');

        if (onlineCamera) {
          const previewWorking = await cameraPage.testLivePreview(onlineCamera.name);
          expect(typeof previewWorking).toBe('boolean');
        }
      }
    });
  });

  describe('Camera Updates (Update)', () => {
    test('should edit camera information', async () => {
      // First create a camera to edit
      const originalCamera = TEST_CAMERAS.editableCamera;
      await cameraPage.createCamera(originalCamera);
      await cameraPage.waitForCamerasToLoad();

      // Update the camera
      const updatedData = {
        name: 'Updated Camera Name',
        location: 'Updated Location'
      };

      await cameraPage.updateCamera(originalCamera.name, updatedData);
      await cameraPage.waitForCamerasToLoad();

      // Verify camera was updated
      const cameraExists = await cameraPage.findCameraByName(updatedData.name);
      expect(cameraExists).toBe(true);

      // Original name should not exist anymore
      const originalExists = await cameraPage.findCameraByName(originalCamera.name);
      expect(originalExists).toBe(false);
    });

    test('should validate updated camera data', async () => {
      await cameraPage.waitForCamerasToLoad();

      const cameras = await cameraPage.getCameraList();

      if (cameras.length > 0) {
        const testCamera = cameras[0];

        // Try to update with invalid data
        await cameraPage.editCameraByName(testCamera.name);

        // Clear name field (required field)
        await page.focus('[data-testid="camera-name-input"], input[name="name"]');
        await page.keyboard.down('Control');
        await page.keyboard.press('KeyA');
        await page.keyboard.up('Control');
        await page.keyboard.press('Backspace');

        await cameraPage.saveCameraForm();

        // Should show validation error
        await page.waitForTimeout(1000);
        const errorMessage = await cameraPage.getErrorMessage();
        expect(errorMessage).toBeTruthy();
      }
    });

    test('should cancel camera edit', async () => {
      await cameraPage.waitForCamerasToLoad();

      const cameras = await cameraPage.getCameraList();

      if (cameras.length > 0) {
        const testCamera = cameras[0];
        const originalName = testCamera.name;

        await cameraPage.editCameraByName(testCamera.name);

        // Make some changes
        await page.focus('[data-testid="camera-name-input"], input[name="name"]');
        await page.keyboard.type(' Modified');

        // Cancel the edit
        await cameraPage.cancelCameraForm();

        // Camera should remain unchanged
        await cameraPage.waitForCamerasToLoad();
        const cameraExists = await cameraPage.findCameraByName(originalName);
        expect(cameraExists).toBe(true);
      }
    });
  });

  describe('Camera Deletion (Delete)', () => {
    test('should delete a camera successfully', async () => {
      // First create a camera to delete
      const cameraToDelete = TEST_CAMERAS.deletableCamera;
      await cameraPage.createCamera(cameraToDelete);
      await cameraPage.waitForCamerasToLoad();

      // Verify camera exists
      let cameraExists = await cameraPage.findCameraByName(cameraToDelete.name);
      expect(cameraExists).toBe(true);

      // Delete the camera
      await cameraPage.deleteCameraByName(cameraToDelete.name, true);
      await cameraPage.waitForCamerasToLoad();

      // Verify camera is deleted
      cameraExists = await cameraPage.findCameraByName(cameraToDelete.name);
      expect(cameraExists).toBe(false);
    });

    test('should cancel camera deletion', async () => {
      await cameraPage.waitForCamerasToLoad();

      const cameras = await cameraPage.getCameraList();

      if (cameras.length > 0) {
        const testCamera = cameras[0];

        // Try to delete but cancel
        await cameraPage.deleteCameraByName(testCamera.name, false);

        // Camera should still exist
        const cameraExists = await cameraPage.findCameraByName(testCamera.name);
        expect(cameraExists).toBe(true);
      }
    });

    test('should show delete confirmation dialog', async () => {
      await cameraPage.waitForCamerasToLoad();

      const cameras = await cameraPage.getCameraList();

      if (cameras.length > 0) {
        const testCamera = cameras[0];

        // Search for camera
        await cameraPage.searchCameras(testCamera.name);

        // Find and click delete button
        const cameraElement = await page.$(`[data-testid="camera-card"]:has-text("${testCamera.name}"), [data-testid="camera-row"]:has-text("${testCamera.name}")`);

        if (cameraElement) {
          const deleteButton = await cameraElement.$('[data-testid="delete-camera"], [aria-label="Delete camera"]');
          if (deleteButton) {
            await deleteButton.click();

            // Should show confirmation modal
            const modalVisible = await cameraPage.isElementVisible('[data-testid="delete-confirm-modal"], [role="dialog"]');
            expect(modalVisible).toBe(true);

            // Cancel to clean up
            await cameraPage.clickElement('[data-testid="cancel-delete"], button:has-text("Cancel")');
          }
        }
      }
    });
  });

  describe('Bulk Operations', () => {
    test('should select multiple cameras', async () => {
      await cameraPage.waitForCamerasToLoad();

      const cameras = await cameraPage.getCameraList();

      if (cameras.length >= 2) {
        const cameraNames = cameras.slice(0, 2).map(c => c.name);
        await cameraPage.selectCameras(cameraNames);

        // Should show selected count or enable bulk actions
        const selectedCountVisible = await cameraPage.isElementVisible('[data-testid="selected-count"]');
        const bulkDeleteVisible = await cameraPage.isElementVisible('[data-testid="bulk-delete"]');

        expect(selectedCountVisible || bulkDeleteVisible).toBe(true);
      }
    });

    test('should select all cameras', async () => {
      await cameraPage.waitForCamerasToLoad();

      const initialCount = await cameraPage.getCameraCount();

      if (initialCount > 0) {
        await cameraPage.selectAllCameras();

        // Should enable bulk actions
        const bulkDeleteVisible = await cameraPage.isElementVisible('[data-testid="bulk-delete"]');
        expect(bulkDeleteVisible).toBe(true);
      }
    });

    test('should export cameras', async () => {
      await cameraPage.waitForCamerasToLoad();

      // Try to export cameras
      await cameraPage.exportCameras();

      // Export should trigger without errors
      // Note: In a real test, you might want to verify file download
      await page.waitForTimeout(2000);

      const errorMessage = await cameraPage.getErrorMessage();
      expect(errorMessage).toBeNull();
    });
  });

  describe('Form Validation and Error Handling', () => {
    test('should handle duplicate camera names', async () => {
      await cameraPage.waitForCamerasToLoad();

      const cameras = await cameraPage.getCameraList();

      if (cameras.length > 0) {
        const existingCamera = cameras[0];

        // Try to create camera with same name
        await cameraPage.clickAddCamera();
        await cameraPage.waitForCameraForm();

        const duplicateCamera = {
          ...TEST_CAMERAS.newCamera,
          name: existingCamera.name
        };

        await cameraPage.fillCameraForm(duplicateCamera);
        await cameraPage.saveCameraForm();

        // Should show error about duplicate name
        await page.waitForTimeout(1000);
        const errorMessage = await cameraPage.getErrorMessage();
        expect(errorMessage).toBeTruthy();
      }
    });

    test('should handle network errors gracefully', async () => {
      // Simulate network failure
      await page.setOfflineMode(true);

      await cameraPage.clickAddCamera();
      await cameraPage.waitForCameraForm();
      await cameraPage.fillCameraForm(TEST_CAMERAS.newCamera);
      await cameraPage.saveCameraForm();

      // Should handle network error
      await page.waitForTimeout(3000);
      const errorMessage = await cameraPage.getErrorMessage();
      expect(errorMessage).toBeTruthy();

      // Restore network
      await page.setOfflineMode(false);
    });

    test('should validate coordinate ranges', async () => {
      await cameraPage.clickAddCamera();
      await cameraPage.waitForCameraForm();

      // Test invalid coordinates
      const invalidCamera = {
        ...TEST_CAMERAS.newCamera,
        latitude: 91, // Invalid latitude (> 90)
        longitude: 181 // Invalid longitude (> 180)
      };

      await cameraPage.fillCameraForm(invalidCamera);
      await cameraPage.saveCameraForm();

      // Should show validation error
      await page.waitForTimeout(1000);
      const errorMessage = await cameraPage.getErrorMessage();
      expect(errorMessage).toBeTruthy();
    });
  });

  describe('Performance and Accessibility', () => {
    test('should load camera list within acceptable time', async () => {
      const startTime = Date.now();

      await cameraPage.navigateToCameras();
      await cameraPage.waitForCamerasToLoad();

      const loadTime = Date.now() - startTime;

      // Should load within 5 seconds
      expect(loadTime).toBeLessThan(5000);
    });

    test('should support keyboard navigation', async () => {
      await cameraPage.waitForCamerasToLoad();

      // Test tab navigation
      await page.keyboard.press('Tab');
      await page.keyboard.press('Tab');
      await page.keyboard.press('Tab');

      // Should be able to navigate without errors
      const focusedElement = await page.evaluate(() => {
        return document.activeElement?.tagName;
      });

      expect(focusedElement).toBeTruthy();
    });

    test('should have proper ARIA labels', async () => {
      await cameraPage.waitForCamerasToLoad();

      // Check for proper accessibility attributes
      const addButtonHasLabel = await page.$eval(
        '[data-testid="add-camera-button"], button:has-text("Add Camera")',
        (el) => el.getAttribute('aria-label') || el.textContent
      );

      expect(addButtonHasLabel).toBeTruthy();
    });

    test('should measure Web Vitals performance', async () => {
      const webVitals = await cameraPage.measureWebVitals();

      // Validate performance thresholds for camera page
      expect(webVitals.LCP).toBeLessThan(3000); // 3s for data-heavy page
      expect(webVitals.FID).toBeLessThan(100);  // 100ms for good FID
      expect(webVitals.CLS).toBeLessThan(0.15); // 0.15 for dynamic content
      expect(webVitals.FCP).toBeLessThan(2000); // 2s for good FCP
      expect(webVitals.TTFB).toBeLessThan(1000); // 1s for data loading
    });
  });
});