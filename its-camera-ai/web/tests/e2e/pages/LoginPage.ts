import { Page } from 'puppeteer';
import { BasePage } from './BasePage';
import { TestUser } from '../fixtures/test-data';

export class LoginPage extends BasePage {
  private selectors = {
    // Form elements
    emailInput: '[data-testid="email-input"], input[type="email"], input[name="email"]',
    passwordInput: '[data-testid="password-input"], input[type="password"], input[name="password"]',
    loginButton: '[data-testid="login-button"], button[type="submit"], button:has-text("Sign In")',

    // MFA elements
    mfaCodeInput: '[data-testid="mfa-code-input"], input[name="mfaCode"]',
    mfaSubmitButton: '[data-testid="mfa-submit-button"], button[type="submit"]:has-text("Verify")',

    // Navigation and status
    loginForm: '[data-testid="login-form"], form',
    errorMessage: '[data-testid="error-message"], .error, .alert-error',
    successMessage: '[data-testid="success-message"], .success, .alert-success',
    loadingIndicator: '[data-testid="loading"], .loading, .spinner',

    // Links
    forgotPasswordLink: '[data-testid="forgot-password-link"], a:has-text("Forgot")',
    signUpLink: '[data-testid="signup-link"], a:has-text("Sign Up")',

    // Remember me
    rememberMeCheckbox: '[data-testid="remember-me"], input[name="rememberMe"]',

    // Language switcher
    languageSwitcher: '[data-testid="language-switcher"]',

    // Logo and branding
    logo: '[data-testid="logo"], .logo',

    // Password visibility toggle
    passwordToggle: '[data-testid="password-toggle"], button[aria-label*="password"]',
  } as const;

  constructor(page: Page) {
    super(page);
  }

  /**
   * Navigate to login page
   */
  async navigateToLogin(): Promise<void> {
    await this.navigateTo('/auth/login');
    await this.waitForLoginFormToLoad();
  }

  /**
   * Wait for login form to load
   */
  async waitForLoginFormToLoad(): Promise<void> {
    await this.waitForElement(this.selectors.loginForm);
    await this.waitForElement(this.selectors.emailInput);
    await this.waitForElement(this.selectors.passwordInput);
    await this.waitForElement(this.selectors.loginButton);
  }

  /**
   * Fill email field
   */
  async fillEmail(email: string): Promise<void> {
    await this.fillInput(this.selectors.emailInput, email);
  }

  /**
   * Fill password field
   */
  async fillPassword(password: string): Promise<void> {
    await this.fillInput(this.selectors.passwordInput, password);
  }

  /**
   * Toggle password visibility
   */
  async togglePasswordVisibility(): Promise<void> {
    if (await this.isElementVisible(this.selectors.passwordToggle)) {
      await this.clickElement(this.selectors.passwordToggle);
    }
  }

  /**
   * Check remember me checkbox
   */
  async checkRememberMe(): Promise<void> {
    if (await this.isElementVisible(this.selectors.rememberMeCheckbox)) {
      const isChecked = await this.page.$eval(
        this.selectors.rememberMeCheckbox,
        (el: HTMLInputElement) => el.checked
      );

      if (!isChecked) {
        await this.clickElement(this.selectors.rememberMeCheckbox);
      }
    }
  }

  /**
   * Click login button
   */
  async clickLogin(): Promise<void> {
    await this.clickElement(this.selectors.loginButton);
  }

  /**
   * Login with credentials
   */
  async login(email: string, password: string, rememberMe = false): Promise<void> {
    await this.fillEmail(email);
    await this.fillPassword(password);

    if (rememberMe) {
      await this.checkRememberMe();
    }

    await this.clickLogin();
  }

  /**
   * Login with test user
   */
  async loginWithUser(user: TestUser): Promise<void> {
    await this.login(user.email, user.password);

    // Handle MFA if enabled
    if (user.mfaEnabled) {
      await this.handleMFA();
    }

    // Wait for redirect after successful login
    await this.waitForSuccessfulLogin();
  }

  /**
   * Handle MFA flow
   */
  async handleMFA(code = '123456'): Promise<void> {
    // Wait for MFA form to appear
    await this.waitForElement(this.selectors.mfaCodeInput);

    // Fill MFA code
    await this.fillInput(this.selectors.mfaCodeInput, code);

    // Submit MFA
    await this.clickElement(this.selectors.mfaSubmitButton);
  }

  /**
   * Wait for successful login (redirect to dashboard)
   */
  async waitForSuccessfulLogin(): Promise<void> {
    // Wait for URL change to dashboard or home page
    await this.page.waitForFunction(
      () => window.location.pathname.includes('/dashboard') ||
            window.location.pathname.includes('/home'),
      { timeout: 10000 }
    );

    // Additional check for dashboard elements
    try {
      await this.page.waitForSelector('[data-testid="dashboard-header"], [data-testid="main-navigation"]', {
        timeout: 5000
      });
    } catch {
      // Dashboard elements might not be loaded yet, that's ok
    }
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
   * Check if error message is displayed
   */
  async hasErrorMessage(): Promise<boolean> {
    return await this.isElementVisible(this.selectors.errorMessage);
  }

  /**
   * Check if loading indicator is visible
   */
  async isLoading(): Promise<boolean> {
    return await this.isElementVisible(this.selectors.loadingIndicator);
  }

  /**
   * Click forgot password link
   */
  async clickForgotPassword(): Promise<void> {
    await this.clickElement(this.selectors.forgotPasswordLink, true);
  }

  /**
   * Click sign up link
   */
  async clickSignUp(): Promise<void> {
    await this.clickElement(this.selectors.signUpLink, true);
  }

  /**
   * Change language
   */
  async changeLanguage(language: 'en' | 'vi'): Promise<void> {
    if (await this.isElementVisible(this.selectors.languageSwitcher)) {
      await this.clickElement(this.selectors.languageSwitcher);

      // Wait for language options and select
      const languageOption = `[data-testid="language-${language}"]`;
      await this.waitForElement(languageOption);
      await this.clickElement(languageOption);

      // Wait for page to reload with new language
      await this.page.waitForTimeout(1000);
    }
  }

  /**
   * Verify login form elements are present
   */
  async verifyLoginForm(): Promise<{
    hasEmailField: boolean;
    hasPasswordField: boolean;
    hasLoginButton: boolean;
    hasRememberMe: boolean;
    hasForgotPassword: boolean;
    hasSignUp: boolean;
    hasLogo: boolean;
  }> {
    return {
      hasEmailField: await this.isElementVisible(this.selectors.emailInput),
      hasPasswordField: await this.isElementVisible(this.selectors.passwordInput),
      hasLoginButton: await this.isElementVisible(this.selectors.loginButton),
      hasRememberMe: await this.isElementVisible(this.selectors.rememberMeCheckbox),
      hasForgotPassword: await this.isElementVisible(this.selectors.forgotPasswordLink),
      hasSignUp: await this.isElementVisible(this.selectors.signUpLink),
      hasLogo: await this.isElementVisible(this.selectors.logo),
    };
  }

  /**
   * Test login form validation
   */
  async testFormValidation(): Promise<{
    emptyFieldsError: boolean;
    invalidEmailError: boolean;
  }> {
    const results = {
      emptyFieldsError: false,
      invalidEmailError: false,
    };

    // Test empty fields
    await this.clickLogin();
    await this.page.waitForTimeout(1000);
    results.emptyFieldsError = await this.hasErrorMessage();

    // Clear any errors
    await this.page.reload();
    await this.waitForLoginFormToLoad();

    // Test invalid email format
    await this.fillEmail('invalid-email');
    await this.fillPassword('password123');
    await this.clickLogin();
    await this.page.waitForTimeout(1000);
    results.invalidEmailError = await this.hasErrorMessage();

    return results;
  }

  /**
   * Test keyboard navigation
   */
  async testKeyboardNavigation(): Promise<boolean> {
    // Focus on email field and tab through form
    await this.page.focus(this.selectors.emailInput);

    // Tab to password field
    await this.page.keyboard.press('Tab');
    const passwordFocused = await this.page.evaluate((selector) => {
      return document.activeElement === document.querySelector(selector);
    }, this.selectors.passwordInput);

    // Tab to login button (or remember me if present)
    await this.page.keyboard.press('Tab');

    // Tab to login button if remember me was focused
    if (await this.isElementVisible(this.selectors.rememberMeCheckbox)) {
      await this.page.keyboard.press('Tab');
    }

    const loginButtonFocused = await this.page.evaluate((selector) => {
      return document.activeElement === document.querySelector(selector);
    }, this.selectors.loginButton);

    return passwordFocused && loginButtonFocused;
  }

  /**
   * Test form submission with Enter key
   */
  async testEnterKeySubmission(): Promise<void> {
    await this.fillEmail('test@example.com');
    await this.fillPassword('password123');

    // Press Enter while in password field
    await this.page.focus(this.selectors.passwordInput);
    await this.page.keyboard.press('Enter');
  }

  /**
   * Get current URL
   */
  getCurrentUrl(): string {
    return this.page.url();
  }

  /**
   * Check if we're on login page
   */
  isOnLoginPage(): boolean {
    return this.getCurrentUrl().includes('/auth/login') ||
           this.getCurrentUrl().includes('/login');
  }

  /**
   * Check if we're redirected away from login
   */
  async isRedirectedFromLogin(): Promise<boolean> {
    await this.page.waitForTimeout(2000);
    return !this.isOnLoginPage();
  }
}