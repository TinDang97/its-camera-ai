#!/usr/bin/env node

/**
 * Accessibility Validation Script for React Components
 *
 * This script validates WCAG 2.1 AA compliance for React components:
 * - Semantic HTML usage
 * - ARIA attributes and labels
 * - Keyboard navigation support
 * - Color contrast and visual accessibility
 * - Screen reader compatibility
 * - Focus management
 */

const fs = require('fs');
const path = require('path');

// ANSI color codes
const colors = {
  reset: '\x1b[0m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  cyan: '\x1b[36m',
  magenta: '\x1b[35m',
};

function colorize(text, color) {
  return `${colors[color]}${text}${colors.reset}`;
}

function logError(message) {
  console.error(colorize(`❌ ${message}`, 'red'));
}

function logWarning(message) {
  console.warn(colorize(`⚠️  ${message}`, 'yellow'));
}

function logSuccess(message) {
  console.log(colorize(`✅ ${message}`, 'green'));
}

function logInfo(message) {
  console.log(colorize(`ℹ️  ${message}`, 'blue'));
}

/**
 * Validate semantic HTML usage
 */
function validateSemanticHTML(content, filePath) {
  const errors = [];
  const warnings = [];

  // Check for proper heading hierarchy
  const headings = content.match(/<h[1-6][^>]*>/g) || [];
  if (headings.length > 0) {
    const headingLevels = headings.map(h => parseInt(h.match(/h([1-6])/)[1]));

    // Check for skipped heading levels
    for (let i = 1; i < headingLevels.length; i++) {
      if (headingLevels[i] - headingLevels[i-1] > 1) {
        warnings.push(`${filePath}: Heading level skipped (h${headingLevels[i-1]} to h${headingLevels[i]})`);
      }
    }

    // Check for multiple h1 elements
    const h1Count = headingLevels.filter(level => level === 1).length;
    if (h1Count > 1) {
      warnings.push(`${filePath}: Multiple h1 elements found - consider using only one per page`);
    }
  }

  // Check for proper list usage
  const listItems = content.match(/<li[^>]*>/g) || [];
  const lists = content.match(/<(ul|ol)[^>]*>/g) || [];

  if (listItems.length > 0 && lists.length === 0) {
    errors.push(`${filePath}: List items found without proper ul/ol wrapper`);
  }

  // Check for proper form labels
  const inputs = content.match(/<input[^>]*>/g) || [];
  const labels = content.match(/<label[^>]*>/g) || [];

  inputs.forEach((input, index) => {
    if (!input.includes('aria-label') && !input.includes('aria-labelledby')) {
      if (labels.length === 0) {
        warnings.push(`${filePath}: Input element #${index + 1} should have associated label`);
      }
    }
  });

  // Check for proper button vs link usage
  const buttons = content.match(/<button[^>]*>/g) || [];
  const links = content.match(/<a[^>]*>/g) || [];

  buttons.forEach((button, index) => {
    if (button.includes('href')) {
      warnings.push(`${filePath}: Button #${index + 1} should not have href attribute - use <a> instead`);
    }
  });

  links.forEach((link, index) => {
    if (link.includes('onClick') && !link.includes('href')) {
      warnings.push(`${filePath}: Link #${index + 1} with onClick should have href or be a button`);
    }
  });

  return { errors, warnings };
}

/**
 * Validate ARIA attributes and labels
 */
function validateARIA(content, filePath) {
  const errors = [];
  const warnings = [];

  // Check for required alt text on images
  const images = content.match(/<img[^>]*>/g) || [];
  images.forEach((img, index) => {
    if (!img.includes('alt=')) {
      errors.push(`${filePath}: Image #${index + 1} missing alt attribute`);
    } else if (img.includes('alt=""') && !img.includes('aria-hidden="true"')) {
      warnings.push(`${filePath}: Image #${index + 1} has empty alt - consider aria-hidden if decorative`);
    }
  });

  // Check for proper ARIA labels on interactive elements
  const interactiveElements = [
    ...content.match(/<button[^>]*>/g) || [],
    ...content.match(/<input[^>]*type="button"[^>]*>/g) || [],
    ...content.match(/<input[^>]*type="submit"[^>]*>/g) || [],
    ...content.match(/<a[^>]*onClick[^>]*>/g) || [],
  ];

  interactiveElements.forEach((element, index) => {
    if (!element.includes('aria-label') &&
        !element.includes('aria-labelledby') &&
        !element.includes('aria-describedby') &&
        !element.includes('>') // Skip if we can't see the content
    ) {
      warnings.push(`${filePath}: Interactive element #${index + 1} may need aria-label`);
    }
  });

  // Check for proper role attributes
  const rolePattern = /role="([^"]*)"/g;
  let roleMatch;
  const validRoles = [
    'button', 'checkbox', 'dialog', 'grid', 'gridcell', 'listbox', 'menubar',
    'menuitem', 'option', 'progressbar', 'radio', 'scrollbar', 'slider',
    'spinbutton', 'tab', 'tabpanel', 'textbox', 'tooltip', 'tree', 'treeitem',
    'alert', 'alertdialog', 'application', 'article', 'banner', 'complementary',
    'contentinfo', 'form', 'main', 'navigation', 'region', 'search'
  ];

  while ((roleMatch = rolePattern.exec(content)) !== null) {
    const role = roleMatch[1];
    if (!validRoles.includes(role)) {
      warnings.push(`${filePath}: Invalid ARIA role "${role}"`);
    }
  }

  // Check for ARIA properties consistency
  if (content.includes('aria-expanded')) {
    if (!content.includes('aria-controls') && !content.includes('aria-owns')) {
      warnings.push(`${filePath}: aria-expanded should be used with aria-controls or aria-owns`);
    }
  }

  if (content.includes('aria-describedby') || content.includes('aria-labelledby')) {
    // Extract referenced IDs
    const idRefs = content.match(/aria-(?:describedby|labelledby)="([^"]*)"/g) || [];
    idRefs.forEach(ref => {
      const id = ref.match(/"([^"]*)"/)[1];
      if (!content.includes(`id="${id}"`)) {
        errors.push(`${filePath}: Referenced ID "${id}" not found in component`);
      }
    });
  }

  return { errors, warnings };
}

/**
 * Validate keyboard navigation support
 */
function validateKeyboardNavigation(content, filePath) {
  const warnings = [];

  // Check for click handlers without keyboard support
  const clickableElements = content.match(/<div[^>]*onClick[^>]*>/g) || [];
  clickableElements.forEach((element, index) => {
    if (!element.includes('onKeyDown') &&
        !element.includes('onKeyPress') &&
        !element.includes('onKeyUp') &&
        !element.includes('tabIndex')) {
      warnings.push(`${filePath}: Clickable div #${index + 1} should support keyboard interaction`);
    }
  });

  // Check for custom components that might need tabIndex
  const customInteractive = content.match(/<\w+[^>]*onClick[^>]*>/g) || [];
  customInteractive.forEach((element, index) => {
    const tagName = element.match(/<(\w+)/)[1];
    if (!['button', 'a', 'input', 'select', 'textarea'].includes(tagName.toLowerCase())) {
      if (!element.includes('tabIndex') && !element.includes('role=')) {
        warnings.push(`${filePath}: Custom interactive element "${tagName}" #${index + 1} may need tabIndex or role`);
      }
    }
  });

  // Check for focus management in modals/dialogs
  if (content.includes('dialog') || content.includes('modal') || content.includes('Modal')) {
    if (!content.includes('focus') && !content.includes('autoFocus')) {
      warnings.push(`${filePath}: Modal/dialog should manage focus for accessibility`);
    }
  }

  return { warnings };
}

/**
 * Validate color and visual accessibility
 */
function validateVisualAccessibility(content, filePath) {
  const warnings = [];

  // Check for color-only information
  if (content.includes('color:') || content.includes('backgroundColor:')) {
    if (!content.includes('aria-label') && !content.includes('title') && !content.includes('alt')) {
      warnings.push(`${filePath}: Color styling should be accompanied by text or ARIA labels`);
    }
  }

  // Check for proper text sizing
  if (content.includes('fontSize') || content.includes('font-size')) {
    const sizeMatches = content.match(/(?:fontSize|font-size):\s*["']?(\d+)px/g);
    if (sizeMatches) {
      sizeMatches.forEach(match => {
        const size = parseInt(match.match(/(\d+)px/)[1]);
        if (size < 16) {
          warnings.push(`${filePath}: Font size ${size}px may be too small (minimum 16px recommended)`);
        }
      });
    }
  }

  // Check for proper contrast indicators
  if (content.includes('disabled') || content.includes(':disabled')) {
    if (!content.includes('aria-disabled') && !content.includes('aria-label')) {
      warnings.push(`${filePath}: Disabled elements should include aria-disabled or descriptive labels`);
    }
  }

  return { warnings };
}

/**
 * Validate form accessibility
 */
function validateFormAccessibility(content, filePath) {
  const errors = [];
  const warnings = [];

  // Check for form validation
  if (content.includes('<form') || content.includes('onSubmit')) {
    if (!content.includes('required') && !content.includes('aria-required')) {
      warnings.push(`${filePath}: Form should indicate required fields`);
    }

    if (!content.includes('error') && !content.includes('invalid')) {
      warnings.push(`${filePath}: Form should provide error feedback mechanisms`);
    }
  }

  // Check for fieldset and legend usage
  const fieldsets = content.match(/<fieldset[^>]*>/g) || [];
  const legends = content.match(/<legend[^>]*>/g) || [];

  if (fieldsets.length > legends.length) {
    errors.push(`${filePath}: Fieldsets should include legend elements`);
  }

  // Check for input types
  const inputs = content.match(/<input[^>]*>/g) || [];
  inputs.forEach((input, index) => {
    if (!input.includes('type=')) {
      warnings.push(`${filePath}: Input #${index + 1} should specify type attribute`);
    }

    if (input.includes('type="text"') && input.includes('email')) {
      warnings.push(`${filePath}: Input #${index + 1} should use type="email" for email fields`);
    }
  });

  return { errors, warnings };
}

/**
 * Validate screen reader compatibility
 */
function validateScreenReader(content, filePath) {
  const warnings = [];

  // Check for skip links
  if (content.includes('<nav') || content.includes('navigation')) {
    if (!content.includes('skip') && !content.includes('#main')) {
      warnings.push(`${filePath}: Consider adding skip navigation links`);
    }
  }

  // Check for proper page structure
  if (content.includes('<main') || content.includes('role="main"')) {
    if (!content.includes('aria-label') && !content.includes('<h1')) {
      warnings.push(`${filePath}: Main content area should have clear heading or label`);
    }
  }

  // Check for live regions
  if (content.includes('loading') || content.includes('error') || content.includes('success')) {
    if (!content.includes('aria-live') && !content.includes('role="status"') && !content.includes('role="alert"')) {
      warnings.push(`${filePath}: Dynamic content changes should use aria-live regions`);
    }
  }

  // Check for table accessibility
  const tables = content.match(/<table[^>]*>/g) || [];
  if (tables.length > 0) {
    if (!content.includes('<caption') && !content.includes('aria-label')) {
      warnings.push(`${filePath}: Tables should include caption or aria-label`);
    }

    if (!content.includes('<th') && !content.includes('scope=')) {
      warnings.push(`${filePath}: Tables should use th elements with scope attributes`);
    }
  }

  return { warnings };
}

/**
 * Validate a single file for accessibility
 */
function validateAccessibility(filePath) {
  const errors = [];
  const warnings = [];

  if (!fs.existsSync(filePath)) {
    errors.push(`File not found: ${filePath}`);
    return { errors, warnings };
  }

  let content;
  try {
    content = fs.readFileSync(filePath, 'utf8');
  } catch (error) {
    errors.push(`Error reading file ${filePath}: ${error.message}`);
    return { errors, warnings };
  }

  // Skip non-React/HTML files
  if (!content.includes('<') || (!content.includes('tsx') && !content.includes('jsx'))) {
    return { errors, warnings };
  }

  // Run all validations
  const validations = [
    validateSemanticHTML(content, filePath),
    validateARIA(content, filePath),
    validateKeyboardNavigation(content, filePath),
    validateVisualAccessibility(content, filePath),
    validateFormAccessibility(content, filePath),
    validateScreenReader(content, filePath),
  ];

  // Collect all results
  validations.forEach(result => {
    if (result.errors) errors.push(...result.errors);
    if (result.warnings) warnings.push(...result.warnings);
  });

  return { errors, warnings };
}

/**
 * Main validation function
 */
function validateAccessibilityInFiles(filePaths) {
  logInfo('♿ Validating accessibility compliance...');

  let totalErrors = 0;
  let totalWarnings = 0;
  const validatedFiles = [];

  // If no file paths provided, get them from command line arguments
  if (!filePaths || filePaths.length === 0) {
    filePaths = process.argv.slice(2);
  }

  // If still no paths, scan component files
  if (filePaths.length === 0) {
    const componentDirs = [
      path.join(__dirname, '..', 'components'),
      path.join(__dirname, '..', 'app'),
    ];

    filePaths = [];

    function scanDirectory(dir) {
      try {
        const files = fs.readdirSync(dir);
        files.forEach(file => {
          const fullPath = path.join(dir, file);
          const stat = fs.statSync(fullPath);

          if (stat.isDirectory() && !file.startsWith('.') && file !== 'node_modules') {
            scanDirectory(fullPath);
          } else if (file.match(/\.(tsx|jsx)$/)) {
            filePaths.push(fullPath);
          }
        });
      } catch (error) {
        // Ignore directory read errors
      }
    }

    componentDirs.forEach(dir => {
      if (fs.existsSync(dir)) {
        scanDirectory(dir);
      }
    });
  }

  if (filePaths.length === 0) {
    logWarning('No component files found to validate');
    return;
  }

  logInfo(`Validating ${filePaths.length} component files for accessibility...`);

  // Validate each file
  filePaths.forEach(filePath => {
    const { errors, warnings } = validateAccessibility(filePath);

    if (errors.length > 0 || warnings.length > 0) {
      validatedFiles.push({
        filePath,
        errors,
        warnings,
      });
    }

    totalErrors += errors.length;
    totalWarnings += warnings.length;
  });

  // Display results
  if (validatedFiles.length > 0) {
    console.log('\n' + '='.repeat(80));
    logInfo('Accessibility Validation Results:');
    console.log('='.repeat(80));

    validatedFiles.forEach(({ filePath, errors, warnings }) => {
      const relativePath = path.relative(process.cwd(), filePath);
      console.log(`\n${colorize(relativePath, 'cyan')}:`);

      errors.forEach(error => logError(error));
      warnings.forEach(warning => logWarning(warning));
    });
  }

  // Summary
  console.log('\n' + '='.repeat(80));

  if (totalErrors === 0) {
    logSuccess('Accessibility validation passed!');
    if (totalWarnings > 0) {
      logInfo(`Found ${totalWarnings} accessibility improvement(s) to consider`);
    }

    console.log('\n♿ Accessibility Best Practices:');
    console.log('• Use semantic HTML elements (header, nav, main, section, article)');
    console.log('• Include alt text for all images');
    console.log('• Ensure proper heading hierarchy (h1-h6)');
    console.log('• Add ARIA labels for interactive elements');
    console.log('• Support keyboard navigation for all interactive elements');
    console.log('• Use sufficient color contrast (4.5:1 for normal text)');
    console.log('• Provide focus indicators for keyboard users');
    console.log('• Use live regions for dynamic content updates');
  } else {
    logError(`Accessibility validation failed with ${totalErrors} error(s) and ${totalWarnings} warning(s)`);
    console.log('\nCritical accessibility issues must be fixed for WCAG 2.1 AA compliance.');
    process.exit(1);
  }
}

// Handle command line execution
if (require.main === module) {
  validateAccessibilityInFiles();
}

module.exports = {
  validateAccessibilityInFiles,
  validateAccessibility,
  validateSemanticHTML,
  validateARIA,
  validateKeyboardNavigation,
  validateVisualAccessibility,
  validateFormAccessibility,
  validateScreenReader,
};
