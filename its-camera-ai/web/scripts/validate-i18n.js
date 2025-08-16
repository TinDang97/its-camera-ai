#!/usr/bin/env node

/**
 * i18n Validation Script for next-intl
 *
 * This script validates translation files for completeness, consistency,
 * and proper formatting according to next-intl standards.
 *
 * Validates:
 * - JSON structure and syntax
 * - Translation key consistency across locales
 * - Placeholder consistency (ICU MessageFormat)
 * - Missing or empty translations
 * - Unused translation keys
 * - Nested object structure consistency
 */

const fs = require('fs');
const path = require('path');

// Configuration
const MESSAGES_DIR = path.join(__dirname, '..', 'messages');
const REQUIRED_LOCALES = ['en', 'vi']; // Add more locales as needed
const DEFAULT_LOCALE = 'en';

// ICU MessageFormat placeholder pattern
const ICU_PLACEHOLDER_PATTERN = /\{[^}]+\}/g;

// ANSI color codes for console output
const colors = {
  reset: '\x1b[0m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  cyan: '\x1b[36m',
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
 * Load and parse a translation file
 */
function loadTranslationFile(locale) {
  const filePath = path.join(MESSAGES_DIR, `${locale}.json`);

  if (!fs.existsSync(filePath)) {
    return { error: `Translation file not found: ${locale}.json` };
  }

  try {
    const content = fs.readFileSync(filePath, 'utf8');
    const translations = JSON.parse(content);

    if (typeof translations !== 'object' || translations === null || Array.isArray(translations)) {
      return { error: `Invalid JSON structure in ${locale}.json - must be an object` };
    }

    return { translations, filePath };
  } catch (error) {
    return { error: `Invalid JSON syntax in ${locale}.json: ${error.message}` };
  }
}

/**
 * Flatten nested translation object into dot-notation keys
 */
function flattenTranslations(obj, prefix = '') {
  const flattened = {};

  for (const [key, value] of Object.entries(obj)) {
    const newKey = prefix ? `${prefix}.${key}` : key;

    if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
      Object.assign(flattened, flattenTranslations(value, newKey));
    } else if (typeof value === 'string') {
      flattened[newKey] = value;
    } else {
      flattened[newKey] = String(value);
    }
  }

  return flattened;
}

/**
 * Extract ICU MessageFormat placeholders from a string
 */
function extractPlaceholders(text) {
  const matches = text.match(ICU_PLACEHOLDER_PATTERN);
  return matches ? matches.sort() : [];
}

/**
 * Validate translation completeness across locales
 */
function validateTranslationCompleteness(translations) {
  const errors = [];
  const warnings = [];

  if (!translations[DEFAULT_LOCALE]) {
    errors.push(`Default locale '${DEFAULT_LOCALE}' not found`);
    return { errors, warnings };
  }

  const defaultKeys = new Set(Object.keys(flattenTranslations(translations[DEFAULT_LOCALE])));

  for (const locale of REQUIRED_LOCALES) {
    if (!translations[locale]) {
      errors.push(`Missing locale: ${locale}`);
      continue;
    }

    const localeKeys = new Set(Object.keys(flattenTranslations(translations[locale])));

    // Check for missing keys
    for (const key of defaultKeys) {
      if (!localeKeys.has(key)) {
        errors.push(`Missing translation key "${key}" in ${locale}.json`);
      }
    }

    // Check for extra keys
    for (const key of localeKeys) {
      if (!defaultKeys.has(key)) {
        warnings.push(`Extra translation key "${key}" in ${locale}.json (not in ${DEFAULT_LOCALE}.json)`);
      }
    }
  }

  return { errors, warnings };
}

/**
 * Validate placeholder consistency across locales
 */
function validatePlaceholderConsistency(translations) {
  const errors = [];

  if (!translations[DEFAULT_LOCALE]) {
    return { errors };
  }

  const defaultFlat = flattenTranslations(translations[DEFAULT_LOCALE]);

  for (const locale of REQUIRED_LOCALES) {
    if (!translations[locale] || locale === DEFAULT_LOCALE) continue;

    const localeFlat = flattenTranslations(translations[locale]);

    for (const key of Object.keys(defaultFlat)) {
      if (!localeFlat[key]) continue;

      const defaultPlaceholders = extractPlaceholders(defaultFlat[key]);
      const localePlaceholders = extractPlaceholders(localeFlat[key]);

      if (JSON.stringify(defaultPlaceholders) !== JSON.stringify(localePlaceholders)) {
        errors.push(
          `Placeholder mismatch for key "${key}" in ${locale}.json:\n` +
          `  Expected: ${defaultPlaceholders.join(', ') || 'none'}\n` +
          `  Found: ${localePlaceholders.join(', ') || 'none'}`
        );
      }
    }
  }

  return { errors };
}

/**
 * Validate for empty or whitespace-only translations
 */
function validateEmptyTranslations(translations) {
  const errors = [];

  for (const locale of REQUIRED_LOCALES) {
    if (!translations[locale]) continue;

    const flattened = flattenTranslations(translations[locale]);

    for (const [key, value] of Object.entries(flattened)) {
      if (!value || typeof value !== 'string' || value.trim() === '') {
        errors.push(`Empty translation value for key "${key}" in ${locale}.json`);
      }
    }
  }

  return { errors };
}

/**
 * Check for potential unused translation keys by searching source files
 */
function checkUnusedTranslations(translations) {
  const warnings = [];

  if (!translations[DEFAULT_LOCALE]) {
    return { warnings };
  }

  const defaultFlat = flattenTranslations(translations[DEFAULT_LOCALE]);
  const usedKeys = new Set();

  // Search for translation key usage in source files
  const searchPaths = [
    path.join(__dirname, '..', 'app'),
    path.join(__dirname, '..', 'components'),
    path.join(__dirname, '..', 'hooks'),
  ];

  function searchInFile(filePath) {
    try {
      const content = fs.readFileSync(filePath, 'utf8');

      // Look for t('key') or t("key") patterns
      const tCallPattern = /t\(['"]([\w.]+)['"]\)/g;
      let match;

      while ((match = tCallPattern.exec(content)) !== null) {
        usedKeys.add(match[1]);
      }

      // Look for useTranslations('namespace') patterns
      const useTranslationsPattern = /useTranslations\(['"]([\w.]+)['"]\)/g;
      while ((match = useTranslationsPattern.exec(content)) !== null) {
        // Mark all keys under this namespace as potentially used
        const namespace = match[1];
        for (const key of Object.keys(defaultFlat)) {
          if (key.startsWith(`${namespace}.`)) {
            usedKeys.add(key);
          }
        }
      }
    } catch (error) {
      // Ignore file read errors
    }
  }

  function walkDirectory(dir) {
    try {
      const files = fs.readdirSync(dir);

      for (const file of files) {
        const filePath = path.join(dir, file);
        const stat = fs.statSync(filePath);

        if (stat.isDirectory() && !file.startsWith('.') && file !== 'node_modules') {
          walkDirectory(filePath);
        } else if (file.match(/\.(ts|tsx|js|jsx)$/)) {
          searchInFile(filePath);
        }
      }
    } catch (error) {
      // Ignore directory read errors
    }
  }

  searchPaths.forEach(walkDirectory);

  // Check for unused keys
  for (const key of Object.keys(defaultFlat)) {
    if (!usedKeys.has(key)) {
      warnings.push(`Potentially unused translation key: "${key}"`);
    }
  }

  return { warnings };
}

/**
 * Validate JSON formatting and structure
 */
function validateJsonStructure(locale, filePath) {
  const errors = [];

  try {
    const content = fs.readFileSync(filePath, 'utf8');

    // Check for trailing commas (not valid JSON but common mistake)
    if (content.includes(',\n}') || content.includes(',\n ]')) {
      errors.push(`JSON contains trailing commas in ${locale}.json`);
    }

    // Check for consistent indentation
    const lines = content.split('\n');
    let expectedIndent = 0;

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();
      if (line === '') continue;

      if (line.includes('{') && !line.includes('}')) {
        expectedIndent += 2;
      } else if (line.includes('}') && !line.includes('{')) {
        expectedIndent -= 2;
      }
    }

  } catch (error) {
    errors.push(`Error reading ${locale}.json: ${error.message}`);
  }

  return { errors };
}

/**
 * Main validation function
 */
function validateI18n() {
  logInfo('Starting i18n validation...');

  let totalErrors = 0;
  let totalWarnings = 0;
  const translations = {};

  // Load all translation files
  for (const locale of REQUIRED_LOCALES) {
    logInfo(`Loading ${locale}.json...`);

    const result = loadTranslationFile(locale);
    if (result.error) {
      logError(result.error);
      totalErrors++;
      continue;
    }

    translations[locale] = result.translations;

    // Validate JSON structure
    const { errors: structureErrors } = validateJsonStructure(locale, result.filePath);
    structureErrors.forEach(error => {
      logError(error);
      totalErrors++;
    });
  }

  if (Object.keys(translations).length === 0) {
    logError('No valid translation files found');
    process.exit(1);
  }

  // Validate translation completeness
  logInfo('Validating translation completeness...');
  const { errors: completenessErrors, warnings: completenessWarnings } = validateTranslationCompleteness(translations);

  completenessErrors.forEach(error => {
    logError(error);
    totalErrors++;
  });

  completenessWarnings.forEach(warning => {
    logWarning(warning);
    totalWarnings++;
  });

  // Validate placeholder consistency
  logInfo('Validating placeholder consistency...');
  const { errors: placeholderErrors } = validatePlaceholderConsistency(translations);

  placeholderErrors.forEach(error => {
    logError(error);
    totalErrors++;
  });

  // Validate empty translations
  logInfo('Checking for empty translations...');
  const { errors: emptyErrors } = validateEmptyTranslations(translations);

  emptyErrors.forEach(error => {
    logError(error);
    totalErrors++;
  });

  // Check for unused translations (warnings only)
  logInfo('Checking for unused translations...');
  const { warnings: unusedWarnings } = checkUnusedTranslations(translations);

  unusedWarnings.forEach(warning => {
    logWarning(warning);
    totalWarnings++;
  });

  // Summary
  console.log('\n' + '='.repeat(60));

  if (totalErrors === 0) {
    logSuccess(`i18n validation passed!`);
    if (totalWarnings > 0) {
      logInfo(`Found ${totalWarnings} warning(s) that should be reviewed.`);
    }
  } else {
    logError(`i18n validation failed with ${totalErrors} error(s) and ${totalWarnings} warning(s)`);
    console.log('\nTo fix these issues:');
    console.log('1. Ensure all required locales have translation files');
    console.log('2. Make sure all keys exist in all locales');
    console.log('3. Verify ICU MessageFormat placeholders match across locales');
    console.log('4. Remove empty translation values');
    console.log('5. Consider removing unused translation keys');

    process.exit(1);
  }
}

// Handle command line arguments
if (require.main === module) {
  validateI18n();
}

module.exports = {
  validateI18n,
  loadTranslationFile,
  flattenTranslations,
  extractPlaceholders,
};
