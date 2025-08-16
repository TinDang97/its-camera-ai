import { Page } from 'puppeteer';
import { BasePage } from './BasePage';
import { TestCamera } from '../fixtures/test-data';

export class CameraPage extends BasePage {
  private selectors = {
    // Page navigation
    cameraListPage: '[data-testid="camera-list"], [data-testid="cameras-page"]',
    addCameraButton: '[data-testid="add-camera-button"], button:has-text("Add Camera")',

    // Camera list/grid elements
    cameraGrid: '[data-testid="camera-grid"], .camera-grid',
    cameraList: '[data-testid="camera-list"], .camera-list',
    cameraCard: '[data-testid="camera-card"]',
    cameraRow: '[data-testid="camera-row"]',

    // Camera card/row elements
    cameraName: '[data-testid="camera-name"]',
    cameraStatus: '[data-testid="camera-status"]',
    cameraLocation: '[data-testid="camera-location"]',
    cameraModel: '[data-testid="camera-model"]',
    cameraUptime: '[data-testid="camera-uptime"]',
    cameraActions: '[data-testid="camera-actions"]',

    // Action buttons
    editButton: '[data-testid="edit-camera"], [aria-label="Edit camera"]',
    deleteButton: '[data-testid="delete-camera"], [aria-label="Delete camera"]',
    viewDetailsButton: '[data-testid="view-camera"], [aria-label="View camera details"]',
    duplicateButton: '[data-testid="duplicate-camera"], [aria-label="Duplicate camera"]',

    // Camera form elements
    cameraForm: '[data-testid="camera-form"], form',
    nameInput: '[data-testid="camera-name-input"], input[name="name"]',
    locationInput: '[data-testid="camera-location-input"], input[name="location"]',
    modelSelect: '[data-testid="camera-model-select"], select[name="model"]',
    ipAddressInput: '[data-testid="camera-ip-input"], input[name="ipAddress"]',
    latitudeInput: '[data-testid="camera-latitude-input"], input[name="latitude"]',
    longitudeInput: '[data-testid="camera-longitude-input"], input[name="longitude"]',
    frameRateInput: '[data-testid="camera-framerate-input"], input[name="frameRate"]',
    resolutionSelect: '[data-testid="camera-resolution-select"], select[name="resolution"]',
    descriptionTextarea: '[data-testid="camera-description"], textarea[name="description"]',

    // Form buttons
    saveButton: '[data-testid="save-camera"], button[type="submit"]',
    cancelButton: '[data-testid="cancel-camera"], button:has-text("Cancel")',

    // Search and filters
    searchInput: '[data-testid="camera-search"], input[placeholder*="Search"]',
    statusFilter: '[data-testid="status-filter"], select[name="statusFilter"]',
    locationFilter: '[data-testid="location-filter"], select[name="locationFilter"]',
    modelFilter: '[data-testid="model-filter"], select[name="modelFilter"]',
    clearFiltersButton: '[data-testid="clear-filters"], button:has-text("Clear")',

    // Pagination
    pagination: '[data-testid="pagination"]',
    previousPageButton: '[data-testid="previous-page"], [aria-label="Previous page"]',
    nextPageButton: '[data-testid="next-page"], [aria-label="Next page"]',
    pageInfo: '[data-testid="page-info"]',

    // View controls
    gridViewButton: '[data-testid="grid-view"], [aria-label="Grid view"]',
    listViewButton: '[data-testid="list-view"], [aria-label="List view"]',
    sortSelect: '[data-testid="sort-cameras"], select[name="sort"]',

    // Bulk actions
    selectAllCheckbox: '[data-testid="select-all-cameras"]',
    selectedCountInfo: '[data-testid="selected-count"]',
    bulkDeleteButton: '[data-testid="bulk-delete"], button:has-text("Delete Selected")',
    bulkExportButton: '[data-testid="bulk-export"], button:has-text("Export")',

    // Modals and dialogs
    deleteConfirmModal: '[data-testid="delete-confirm-modal"], [role="dialog"]',
    confirmDeleteButton: '[data-testid="confirm-delete"], button:has-text("Delete")',
    cancelDeleteButton: '[data-testid="cancel-delete"], button:has-text("Cancel")',

    // Camera details view
    cameraDetails: '[data-testid="camera-details"]',
    detailsBackButton: '[data-testid="back-to-list"], button:has-text("Back")',

    // Live preview
    livePreview: '[data-testid="camera-preview"], video, canvas',
    previewControls: '[data-testid="preview-controls"]',
    playButton: '[data-testid="play-preview"]',
    pauseButton: '[data-testid="pause-preview"]',
    fullscreenButton: '[data-testid="fullscreen-preview"]',

    // Status indicators
    onlineIndicator: '[data-testid="status-online"], .status-online',
    offlineIndicator: '[data-testid="status-offline"], .status-offline',
    maintenanceIndicator: '[data-testid="status-maintenance"], .status-maintenance',

    // Loading and error states
    loadingSpinner: '[data-testid="loading"], .loading, .spinner',
    errorMessage: '[data-testid="error-message"], .error',
    emptyState: '[data-testid="empty-state"], .empty-state',

    // Map view (if available)
    mapView: '[data-testid="camera-map"], .camera-map',
    mapMarker: '[data-testid="camera-marker"]',

    // Export/Import
    exportButton: '[data-testid="export-cameras"]',
    importButton: '[data-testid="import-cameras"]',
    fileInput: 'input[type="file"]',
  } as const;

  constructor(page: Page) {
    super(page);
  }

  /**
   * Navigate to cameras page
   */
  async navigateToCameras(): Promise<void> {
    await this.navigateTo('/cameras');
    await this.waitForCameraPageToLoad();
  }

  /**
   * Wait for camera page to load
   */
  async waitForCameraPageToLoad(): Promise<void> {
    await this.waitForElement(this.selectors.cameraListPage);

    // Wait for either camera grid/list or empty state
    await Promise.race([
      this.waitForElement(this.selectors.cameraGrid),
      this.waitForElement(this.selectors.cameraList),
      this.waitForElement(this.selectors.emptyState)
    ]);
  }

  /**
   * Get list of cameras displayed
   */
  async getCameraList(): Promise<Array<{
    id: string;
    name: string;
    status: string;
    location: string;
    model?: string;
  }>> {
    const cameras: Array<{
      id: string;
      name: string;
      status: string;
      location: string;
      model?: string;
    }> = [];

    // Try to get cameras from grid view first, then list view
    const cameraElements = await this.page.$$(`${this.selectors.cameraCard}, ${this.selectors.cameraRow}`);

    for (const element of cameraElements) {
      try {
        const id = await element.evaluate(el => el.getAttribute('data-camera-id') || '');
        const name = await element.$eval(this.selectors.cameraName, el => el.textContent?.trim() || '');
        const status = await element.$eval(this.selectors.cameraStatus, el => el.textContent?.trim() || '');
        const location = await element.$eval(this.selectors.cameraLocation, el => el.textContent?.trim() || '');

        let model = '';
        try {
          model = await element.$eval(this.selectors.cameraModel, el => el.textContent?.trim() || '');
        } catch {
          // Model might not be displayed in all views
        }

        cameras.push({ id, name, status, location, model });
      } catch (error) {
        console.warn('Error extracting camera data:', error);
      }
    }

    return cameras;
  }

  /**
   * Search for cameras
   */
  async searchCameras(query: string): Promise<void> {
    await this.fillInput(this.selectors.searchInput, query);

    // Wait for search results to update
    await this.page.waitForTimeout(1000);
  }

  /**
   * Filter cameras by status
   */
  async filterByStatus(status: 'all' | 'online' | 'offline' | 'maintenance'): Promise<void> {
    if (await this.isElementVisible(this.selectors.statusFilter)) {
      await this.selectOption(this.selectors.statusFilter, status);
      await this.page.waitForTimeout(1000);
    }
  }

  /**
   * Clear all filters
   */
  async clearFilters(): Promise<void> {
    if (await this.isElementVisible(this.selectors.clearFiltersButton)) {
      await this.clickElement(this.selectors.clearFiltersButton);
      await this.page.waitForTimeout(1000);
    }
  }

  /**
   * Switch to grid view
   */
  async switchToGridView(): Promise<void> {
    if (await this.isElementVisible(this.selectors.gridViewButton)) {
      await this.clickElement(this.selectors.gridViewButton);
      await this.waitForElement(this.selectors.cameraGrid);
    }
  }

  /**
   * Switch to list view
   */
  async switchToListView(): Promise<void> {
    if (await this.isElementVisible(this.selectors.listViewButton)) {
      await this.clickElement(this.selectors.listViewButton);
      await this.waitForElement(this.selectors.cameraList);
    }
  }

  /**
   * Click add camera button
   */
  async clickAddCamera(): Promise<void> {
    await this.clickElement(this.selectors.addCameraButton);
    await this.waitForCameraForm();
  }

  /**
   * Wait for camera form to load
   */
  async waitForCameraForm(): Promise<void> {
    await this.waitForElement(this.selectors.cameraForm);
    await this.waitForElement(this.selectors.nameInput);
    await this.waitForElement(this.selectors.saveButton);
  }

  /**
   * Fill camera form
   */
  async fillCameraForm(camera: TestCamera): Promise<void> {
    await this.fillInput(this.selectors.nameInput, camera.name);
    await this.fillInput(this.selectors.locationInput, camera.location);
    await this.fillInput(this.selectors.ipAddressInput, camera.ipAddress);

    if (camera.latitude !== undefined) {
      await this.fillInput(this.selectors.latitudeInput, camera.latitude.toString());
    }

    if (camera.longitude !== undefined) {
      await this.fillInput(this.selectors.longitudeInput, camera.longitude.toString());
    }

    if (camera.frameRate !== undefined) {
      await this.fillInput(this.selectors.frameRateInput, camera.frameRate.toString());
    }

    if (camera.model && await this.isElementVisible(this.selectors.modelSelect)) {
      await this.selectOption(this.selectors.modelSelect, camera.model);
    }

    if (camera.description && await this.isElementVisible(this.selectors.descriptionTextarea)) {
      await this.fillInput(this.selectors.descriptionTextarea, camera.description);
    }
  }

  /**
   * Save camera form
   */
  async saveCameraForm(): Promise<void> {
    await this.clickElement(this.selectors.saveButton);

    // Wait for form to be submitted and page to redirect/update
    await Promise.race([
      this.page.waitForNavigation({ waitUntil: 'networkidle0', timeout: 10000 }),
      this.page.waitForTimeout(3000)
    ]);
  }

  /**
   * Cancel camera form
   */
  async cancelCameraForm(): Promise<void> {
    await this.clickElement(this.selectors.cancelButton);
  }

  /**
   * Create a new camera
   */
  async createCamera(camera: TestCamera): Promise<void> {
    await this.clickAddCamera();
    await this.fillCameraForm(camera);
    await this.saveCameraForm();
  }

  /**
   * Find camera by name
   */
  async findCameraByName(name: string): Promise<boolean> {
    const cameras = await this.getCameraList();
    return cameras.some(camera => camera.name === name);
  }

  /**
   * Edit camera by name
   */
  async editCameraByName(name: string): Promise<void> {
    // First search for the camera
    await this.searchCameras(name);

    // Find and click edit button for the camera
    const cameraElement = await this.page.$(`[data-testid="camera-card"]:has-text("${name}"), [data-testid="camera-row"]:has-text("${name}")`);

    if (cameraElement) {
      const editButton = await cameraElement.$(this.selectors.editButton);
      if (editButton) {
        await editButton.click();
        await this.waitForCameraForm();
      }
    }
  }

  /**
   * Update camera
   */
  async updateCamera(currentName: string, updatedCamera: Partial<TestCamera>): Promise<void> {
    await this.editCameraByName(currentName);

    // Fill only the fields that are provided
    if (updatedCamera.name) {
      await this.clearAndFillInput(this.selectors.nameInput, updatedCamera.name);
    }
    if (updatedCamera.location) {
      await this.clearAndFillInput(this.selectors.locationInput, updatedCamera.location);
    }
    if (updatedCamera.ipAddress) {
      await this.clearAndFillInput(this.selectors.ipAddressInput, updatedCamera.ipAddress);
    }
    if (updatedCamera.frameRate !== undefined) {
      await this.clearAndFillInput(this.selectors.frameRateInput, updatedCamera.frameRate.toString());
    }

    await this.saveCameraForm();
  }

  /**
   * Delete camera by name
   */
  async deleteCameraByName(name: string, confirm = true): Promise<void> {
    // Search for the camera
    await this.searchCameras(name);

    // Find and click delete button
    const cameraElement = await this.page.$(`[data-testid="camera-card"]:has-text("${name}"), [data-testid="camera-row"]:has-text("${name}")`);

    if (cameraElement) {
      const deleteButton = await cameraElement.$(this.selectors.deleteButton);
      if (deleteButton) {
        await deleteButton.click();

        // Wait for confirmation modal
        await this.waitForElement(this.selectors.deleteConfirmModal);

        if (confirm) {
          await this.clickElement(this.selectors.confirmDeleteButton);
        } else {
          await this.clickElement(this.selectors.cancelDeleteButton);
        }

        // Wait for modal to close
        await this.page.waitForTimeout(1000);
      }
    }
  }

  /**
   * View camera details
   */
  async viewCameraDetails(name: string): Promise<void> {
    await this.searchCameras(name);

    const cameraElement = await this.page.$(`[data-testid="camera-card"]:has-text("${name}"), [data-testid="camera-row"]:has-text("${name}")`);

    if (cameraElement) {
      // Try clicking the view details button, or the camera card itself
      try {
        const viewButton = await cameraElement.$(this.selectors.viewDetailsButton);
        if (viewButton) {
          await viewButton.click();
        } else {
          // Click the camera card itself
          await cameraElement.click();
        }

        await this.waitForElement(this.selectors.cameraDetails);
      } catch (error) {
        console.warn('Could not view camera details:', error);
      }
    }
  }

  /**
   * Go back to camera list from details
   */
  async backToList(): Promise<void> {
    await this.clickElement(this.selectors.detailsBackButton);
    await this.waitForCameraPageToLoad();
  }

  /**
   * Select cameras for bulk operations
   */
  async selectCameras(cameraNames: string[]): Promise<void> {
    for (const name of cameraNames) {
      const cameraElement = await this.page.$(`[data-testid="camera-card"]:has-text("${name}"), [data-testid="camera-row"]:has-text("${name}")`);

      if (cameraElement) {
        const checkbox = await cameraElement.$('input[type="checkbox"]');
        if (checkbox) {
          await checkbox.click();
        }
      }
    }
  }

  /**
   * Select all cameras
   */
  async selectAllCameras(): Promise<void> {
    if (await this.isElementVisible(this.selectors.selectAllCheckbox)) {
      await this.clickElement(this.selectors.selectAllCheckbox);
    }
  }

  /**
   * Bulk delete selected cameras
   */
  async bulkDeleteCameras(confirm = true): Promise<void> {
    await this.clickElement(this.selectors.bulkDeleteButton);
    await this.waitForElement(this.selectors.deleteConfirmModal);

    if (confirm) {
      await this.clickElement(this.selectors.confirmDeleteButton);
    } else {
      await this.clickElement(this.selectors.cancelDeleteButton);
    }
  }

  /**
   * Export cameras
   */
  async exportCameras(): Promise<void> {
    if (await this.isElementVisible(this.selectors.exportButton)) {
      await this.clickElement(this.selectors.exportButton);

      // Wait for download to start
      await this.page.waitForTimeout(2000);
    }
  }

  /**
   * Get camera count
   */
  async getCameraCount(): Promise<number> {
    const cameras = await this.getCameraList();
    return cameras.length;
  }

  /**
   * Check if camera list is empty
   */
  async isCameraListEmpty(): Promise<boolean> {
    return await this.isElementVisible(this.selectors.emptyState);
  }

  /**
   * Check if loading
   */
  async isLoading(): Promise<boolean> {
    return await this.isElementVisible(this.selectors.loadingSpinner);
  }

  /**
   * Get error message
   */
  async getErrorMessage(): Promise<string | null> {
    if (await this.isElementVisible(this.selectors.errorMessage)) {
      return await this.getElementText(this.selectors.errorMessage);
    }
    return null;
  }

  /**
   * Wait for cameras to load
   */
  async waitForCamerasToLoad(): Promise<void> {
    // Wait for loading to finish
    await this.page.waitForFunction(
      (loadingSelector) => !document.querySelector(loadingSelector),
      { timeout: 10000 },
      this.selectors.loadingSpinner
    );

    // Wait for content to appear
    await Promise.race([
      this.waitForElement(this.selectors.cameraGrid),
      this.waitForElement(this.selectors.cameraList),
      this.waitForElement(this.selectors.emptyState)
    ]);
  }

  /**
   * Verify camera status
   */
  async verifyCameraStatus(name: string, expectedStatus: 'online' | 'offline' | 'maintenance'): Promise<boolean> {
    await this.searchCameras(name);

    const cameraElement = await this.page.$(`[data-testid="camera-card"]:has-text("${name}"), [data-testid="camera-row"]:has-text("${name}")`);

    if (cameraElement) {
      const statusElement = await cameraElement.$(this.selectors.cameraStatus);
      if (statusElement) {
        const statusText = await statusElement.evaluate(el => el.textContent?.toLowerCase().trim());
        return statusText?.includes(expectedStatus) || false;
      }
    }

    return false;
  }

  /**
   * Test camera live preview
   */
  async testLivePreview(cameraName: string): Promise<boolean> {
    await this.viewCameraDetails(cameraName);

    if (await this.isElementVisible(this.selectors.livePreview)) {
      // Try to play the preview
      if (await this.isElementVisible(this.selectors.playButton)) {
        await this.clickElement(this.selectors.playButton);
        await this.page.waitForTimeout(2000);
      }

      return true;
    }

    return false;
  }

  /**
   * Helper method to clear input and fill new value
   */
  private async clearAndFillInput(selector: string, value: string): Promise<void> {
    await this.page.focus(selector);
    await this.page.keyboard.down('Control');
    await this.page.keyboard.press('KeyA');
    await this.page.keyboard.up('Control');
    await this.page.keyboard.type(value);
  }
}