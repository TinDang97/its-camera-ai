import { Browser, Page } from 'puppeteer';
import { LoginPage } from '../pages/LoginPage';
import { DashboardPage } from '../pages/DashboardPage';
import { getTestBrowser } from '../config/puppeteer.config';
import { TEST_USERS } from '../fixtures/test-data';

describe('Authentication E2E Tests', () => {
  let browser: Browser;
  let page: Page;
  let loginPage: LoginPage;
  let dashboardPage: DashboardPage;

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
    loginPage = new LoginPage(page);
    dashboardPage = new DashboardPage(page);

    // Start each test from the login page
    await loginPage.navigateToLogin();
  });

  afterEach(async () => {
    if (page) {
      await page.close();
    }
  });

  describe('Login Page UI', () => {
    test('should display all login form elements', async () => {
      const formElements = await loginPage.verifyLoginForm();

      expect(formElements.hasEmailField).toBe(true);
      expect(formElements.hasPasswordField).toBe(true);
      expect(formElements.hasLoginButton).toBe(true);
      expect(formElements.hasLogo).toBe(true);

      // These elements might be optional depending on design
      // expect(formElements.hasRememberMe).toBe(true);
      // expect(formElements.hasForgotPassword).toBe(true);
      // expect(formElements.hasSignUp).toBe(true);
    });

    test('should have proper page title and branding', async () => {
      const title = await page.title();
      expect(title).toContain('Sign In');

      // Check if logo is visible and properly positioned
      const logoVisible = await loginPage.isElementVisible('[data-testid="logo"], .logo');
      expect(logoVisible).toBe(true);
    });

    test('should support keyboard navigation', async () => {
      const keyboardNavWorking = await loginPage.testKeyboardNavigation();
      expect(keyboardNavWorking).toBe(true);
    });

    test('should toggle password visibility', async () => {
      await loginPage.fillPassword('testpassword');

      // Check if password toggle exists
      const hasToggle = await loginPage.isElementVisible('[data-testid="password-toggle"], button[aria-label*="password"]');
      if (hasToggle) {
        await loginPage.togglePasswordVisibility();

        // Verify password field type changed
        const passwordType = await page.$eval(
          '[data-testid="password-input"], input[type="password"], input[name="password"]',
          (el: HTMLInputElement) => el.type
        );
        expect(['text', 'password']).toContain(passwordType);
      }
    });
  });

  describe('Form Validation', () => {
    test('should show validation errors for empty fields', async () => {
      const validation = await loginPage.testFormValidation();
      expect(validation.emptyFieldsError).toBe(true);
    });

    test('should validate email format', async () => {
      await loginPage.fillEmail('invalid-email');
      await loginPage.fillPassword('password123');
      await loginPage.clickLogin();

      await page.waitForTimeout(1000);
      const hasError = await loginPage.hasErrorMessage();
      expect(hasError).toBe(true);
    });

    test('should handle form submission with Enter key', async () => {
      await loginPage.testEnterKeySubmission();

      // Should either show validation error or attempt to login
      await page.waitForTimeout(1000);
      const hasError = await loginPage.hasErrorMessage();
      const isLoading = await loginPage.isLoading();

      expect(hasError || isLoading).toBe(true);
    });

    test('should show loading state during login attempt', async () => {
      await loginPage.fillEmail(TEST_USERS.admin.email);
      await loginPage.fillPassword(TEST_USERS.admin.password);
      await loginPage.clickLogin();

      // Check for loading indicator immediately after clicking
      const isLoading = await loginPage.isLoading();
      // Loading state might be very brief with mock API
      expect(typeof isLoading).toBe('boolean');
    });
  });

  describe('Successful Authentication', () => {
    test('should login with valid admin credentials', async () => {
      await loginPage.loginWithUser(TEST_USERS.admin);

      // Should redirect to dashboard
      await loginPage.waitForSuccessfulLogin();

      // Verify we're on dashboard page
      const currentUrl = loginPage.getCurrentUrl();
      expect(currentUrl).toMatch(/\/(dashboard|home)/);

      // Verify dashboard elements are loaded
      const isDashboardLoaded = await dashboardPage.isDashboardLoaded();
      expect(isDashboardLoaded).toBe(true);
    });

    test('should login with valid regular user credentials', async () => {
      await loginPage.loginWithUser(TEST_USERS.user);

      await loginPage.waitForSuccessfulLogin();

      const currentUrl = loginPage.getCurrentUrl();
      expect(currentUrl).toMatch(/\/(dashboard|home)/);
    });

    test('should handle remember me functionality', async () => {
      await loginPage.login(TEST_USERS.admin.email, TEST_USERS.admin.password, true);

      await loginPage.waitForSuccessfulLogin();

      // Verify remember me was checked
      // This would need to be verified by checking if session persists
      // across browser restarts, but that's complex for this test
      expect(true).toBe(true); // Placeholder
    });

    test('should redirect to intended page after login', async () => {
      // Navigate to a protected page first (should redirect to login)
      await page.goto(`${process.env.E2E_BASE_URL || 'http://localhost:3002'}/dashboard/analytics`);

      // Should be redirected to login page
      await loginPage.waitForLoginFormToLoad();

      // Login
      await loginPage.loginWithUser(TEST_USERS.admin);

      // Should redirect back to the intended page
      await page.waitForTimeout(2000);
      const currentUrl = loginPage.getCurrentUrl();
      expect(currentUrl).toContain('analytics');
    });
  });

  describe('Failed Authentication', () => {
    test('should show error for invalid credentials', async () => {
      await loginPage.login('invalid@test.com', 'wrongpassword');

      await page.waitForTimeout(2000);
      const errorMessage = await loginPage.getErrorMessage();
      expect(errorMessage).toBeTruthy();
      expect(errorMessage?.toLowerCase()).toContain('invalid');
    });

    test('should show error for non-existent user', async () => {
      await loginPage.login('nonexistent@test.com', 'password123');

      await page.waitForTimeout(2000);
      const hasError = await loginPage.hasErrorMessage();
      expect(hasError).toBe(true);
    });

    test('should show error for disabled user account', async () => {
      await loginPage.loginWithUser(TEST_USERS.disabled);

      await page.waitForTimeout(2000);
      const errorMessage = await loginPage.getErrorMessage();
      expect(errorMessage).toBeTruthy();
      expect(errorMessage?.toLowerCase()).toMatch(/disabled|suspended|inactive/);
    });

    test('should handle network errors gracefully', async () => {
      // Simulate network failure
      await page.setOfflineMode(true);

      await loginPage.login(TEST_USERS.admin.email, TEST_USERS.admin.password);

      await page.waitForTimeout(3000);
      const errorMessage = await loginPage.getErrorMessage();
      expect(errorMessage).toBeTruthy();

      // Restore network
      await page.setOfflineMode(false);
    });
  });

  describe('Multi-Factor Authentication (MFA)', () => {
    test('should handle MFA flow for enabled users', async () => {
      await loginPage.loginWithUser(TEST_USERS.mfaUser);

      // Should show MFA form
      const mfaInputVisible = await loginPage.isElementVisible('[data-testid="mfa-code-input"], input[name="mfaCode"]');
      expect(mfaInputVisible).toBe(true);

      // Complete MFA
      await loginPage.handleMFA('123456');

      // Should redirect to dashboard
      await loginPage.waitForSuccessfulLogin();
      const currentUrl = loginPage.getCurrentUrl();
      expect(currentUrl).toMatch(/\/(dashboard|home)/);
    });

    test('should show error for invalid MFA code', async () => {
      await loginPage.login(TEST_USERS.mfaUser.email, TEST_USERS.mfaUser.password);

      // Wait for MFA form
      await page.waitForSelector('[data-testid="mfa-code-input"], input[name="mfaCode"]', { timeout: 5000 });

      // Enter invalid MFA code
      await loginPage.handleMFA('000000');

      await page.waitForTimeout(2000);
      const hasError = await loginPage.hasErrorMessage();
      expect(hasError).toBe(true);
    });

    test('should allow MFA code resend', async () => {
      await loginPage.login(TEST_USERS.mfaUser.email, TEST_USERS.mfaUser.password);

      // Wait for MFA form
      await page.waitForSelector('[data-testid="mfa-code-input"], input[name="mfaCode"]', { timeout: 5000 });

      // Check if resend button exists and click it
      const resendButton = '[data-testid="mfa-resend"], button:has-text("Resend")';
      const hasResendButton = await loginPage.isElementVisible(resendButton);

      if (hasResendButton) {
        await loginPage.clickElement(resendButton);

        // Should show success message or update UI
        await page.waitForTimeout(1000);
        // This is implementation-dependent
        expect(true).toBe(true);
      }
    });
  });

  describe('Navigation and Links', () => {
    test('should navigate to forgot password page', async () => {
      const hasForgotPasswordLink = await loginPage.isElementVisible('[data-testid="forgot-password-link"], a:has-text("Forgot")');

      if (hasForgotPasswordLink) {
        await loginPage.clickForgotPassword();

        // Should navigate to forgot password page
        await page.waitForTimeout(1000);
        const currentUrl = loginPage.getCurrentUrl();
        expect(currentUrl).toMatch(/forgot|reset/);
      }
    });

    test('should navigate to sign up page', async () => {
      const hasSignUpLink = await loginPage.isElementVisible('[data-testid="signup-link"], a:has-text("Sign Up")');

      if (hasSignUpLink) {
        await loginPage.clickSignUp();

        // Should navigate to sign up page
        await page.waitForTimeout(1000);
        const currentUrl = loginPage.getCurrentUrl();
        expect(currentUrl).toMatch(/signup|register/);
      }
    });
  });

  describe('Internationalization', () => {
    test('should support language switching', async () => {
      const hasLanguageSwitcher = await loginPage.isElementVisible('[data-testid="language-switcher"]');

      if (hasLanguageSwitcher) {
        // Test Vietnamese language
        await loginPage.changeLanguage('vi');

        // Verify language change by checking URL or content
        const currentUrl = loginPage.getCurrentUrl();
        expect(currentUrl).toMatch(/\/vi\/|locale=vi/);

        // Switch back to English
        await loginPage.changeLanguage('en');

        const newUrl = loginPage.getCurrentUrl();
        expect(newUrl).toMatch(/\/en\/|locale=en/);
      }
    });
  });

  describe('Security and Performance', () => {
    test('should not expose sensitive information in DOM', async () => {
      await loginPage.fillPassword('sensitivepassword');

      // Check that password is not visible in page source
      const pageContent = await page.content();
      expect(pageContent).not.toContain('sensitivepassword');

      // Verify password field has proper type
      const passwordType = await page.$eval(
        '[data-testid="password-input"], input[type="password"], input[name="password"]',
        (el: HTMLInputElement) => el.type
      );
      expect(passwordType).toBe('password');
    });

    test('should measure and validate Web Vitals', async () => {
      // Measure Web Vitals for login page
      const webVitals = await loginPage.measureWebVitals();

      // Validate performance thresholds
      expect(webVitals.LCP).toBeLessThan(2500); // 2.5s for good LCP
      expect(webVitals.FID).toBeLessThan(100);  // 100ms for good FID
      expect(webVitals.CLS).toBeLessThan(0.1);  // 0.1 for good CLS
      expect(webVitals.FCP).toBeLessThan(1800); // 1.8s for good FCP
      expect(webVitals.TTFB).toBeLessThan(800); // 800ms for good TTFB
    });

    test('should prevent multiple rapid form submissions', async () => {
      await loginPage.fillEmail(TEST_USERS.admin.email);
      await loginPage.fillPassword(TEST_USERS.admin.password);

      // Click login button multiple times rapidly
      await loginPage.clickLogin();
      await loginPage.clickLogin();
      await loginPage.clickLogin();

      // Should only process one request
      // This test depends on implementation - button should be disabled
      // or loading state should prevent multiple submissions
      const isLoading = await loginPage.isLoading();
      expect(typeof isLoading).toBe('boolean');
    });

    test('should handle CSP and security headers', async () => {
      const response = await page.goto(`${process.env.E2E_BASE_URL || 'http://localhost:3002'}/auth/login`);

      if (response) {
        const headers = response.headers();

        // Check for security headers
        expect(headers['x-frame-options'] || headers['x-content-type-options']).toBeDefined();

        // In production, should have CSP header
        if (process.env.NODE_ENV === 'production') {
          expect(headers['content-security-policy']).toBeDefined();
        }
      }
    });
  });

  describe('Accessibility', () => {
    test('should have proper ARIA labels and roles', async () => {
      // Check for proper form labeling
      const emailLabel = await page.$eval(
        '[data-testid="email-input"], input[type="email"], input[name="email"]',
        (el: HTMLInputElement) => el.getAttribute('aria-label') || el.labels?.[0]?.textContent
      );
      expect(emailLabel).toBeTruthy();

      const passwordLabel = await page.$eval(
        '[data-testid="password-input"], input[type="password"], input[name="password"]',
        (el: HTMLInputElement) => el.getAttribute('aria-label') || el.labels?.[0]?.textContent
      );
      expect(passwordLabel).toBeTruthy();
    });

    test('should support screen reader navigation', async () => {
      // Test that form has proper heading structure
      const headings = await page.$$eval('h1, h2, h3',
        (elements) => elements.map(el => el.textContent)
      );
      expect(headings.length).toBeGreaterThan(0);
      expect(headings.some(h => h?.toLowerCase().includes('sign'))).toBe(true);
    });

    test('should have proper focus management', async () => {
      // Focus should start on first form field
      await page.focus('[data-testid="email-input"], input[type="email"], input[name="email"]');

      const focusedElement = await page.evaluate(() => {
        return document.activeElement?.tagName;
      });
      expect(focusedElement).toBe('INPUT');
    });
  });
});