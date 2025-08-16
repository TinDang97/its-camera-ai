#!/usr/bin/env node

/**
 * Translation Completeness Checker for next-intl
 *
 * This script checks for translation completeness across all supported locales:
 * - Compares translation keys between locales
 * - Identifies missing translations
 * - Calculates completion percentages
 * - Suggests priority translations based on usage
 */

const fs = require('fs');
const path = require('path');

// Configuration
const MESSAGES_DIR = path.join(__dirname, '..', 'messages');
const DEFAULT_LOCALE = 'en';
const PRIORITY_NAMESPACES = ['common', 'dashboard', 'camera', 'auth']; // High-priority translation namespaces

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
 * Load all available translation files
 */
function loadTranslationFiles() {
  const translations = {};
  const supportedLocales = [];

  if (!fs.existsSync(MESSAGES_DIR)) {
    logError(`Messages directory not found: ${MESSAGES_DIR}`);
    process.exit(1);
  }

  const files = fs.readdirSync(MESSAGES_DIR);

  files.forEach(file => {
    if (path.extname(file) === '.json') {
      const locale = path.basename(file, '.json');
      const filePath = path.join(MESSAGES_DIR, file);

      try {
        const content = fs.readFileSync(filePath, 'utf8');
        translations[locale] = JSON.parse(content);
        supportedLocales.push(locale);
      } catch (error) {
        logError(`Error loading ${file}: ${error.message}`);
      }
    }
  });

  return { translations, supportedLocales };
}

/**
 * Flatten nested translation object
 */
function flattenTranslations(obj, prefix = '') {
  const flattened = {};

  for (const [key, value] of Object.entries(obj)) {
    const newKey = prefix ? `${prefix}.${key}` : key;

    if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
      Object.assign(flattened, flattenTranslations(value, newKey));
    } else {
      flattened[newKey] = value;
    }
  }

  return flattened;
}

/**
 * Calculate translation completeness for each locale
 */
function calculateCompleteness(translations, supportedLocales) {
  if (!translations[DEFAULT_LOCALE]) {
    logError(`Default locale '${DEFAULT_LOCALE}' not found`);
    process.exit(1);
  }

  const defaultFlat = flattenTranslations(translations[DEFAULT_LOCALE]);
  const totalKeys = Object.keys(defaultFlat).length;

  const completeness = {};

  supportedLocales.forEach(locale => {
    if (locale === DEFAULT_LOCALE) {
      completeness[locale] = {
        total: totalKeys,
        translated: totalKeys,
        missing: [],
        percentage: 100,
        empty: [],
      };
      return;
    }

    const localeFlat = flattenTranslations(translations[locale]);
    const translatedKeys = [];
    const missingKeys = [];
    const emptyKeys = [];

    Object.keys(defaultFlat).forEach(key => {
      if (localeFlat[key]) {
        if (typeof localeFlat[key] === 'string' && localeFlat[key].trim() !== '') {
          translatedKeys.push(key);
        } else {
          emptyKeys.push(key);
        }
      } else {
        missingKeys.push(key);
      }
    });

    completeness[locale] = {
      total: totalKeys,
      translated: translatedKeys.length,
      missing: missingKeys,
      empty: emptyKeys,
      percentage: ((translatedKeys.length / totalKeys) * 100).toFixed(1),
    };
  });

  return completeness;
}

/**
 * Identify priority translations based on namespace
 */
function identifyPriorityTranslations(missingKeys) {
  const priority = [];
  const regular = [];

  missingKeys.forEach(key => {
    const namespace = key.split('.')[0];
    if (PRIORITY_NAMESPACES.includes(namespace)) {
      priority.push(key);
    } else {
      regular.push(key);
    }
  });

  return { priority, regular };
}

/**
 * Generate completion report
 */
function generateCompletionReport(completeness, supportedLocales) {
  console.log('\n' + '='.repeat(80));
  logInfo('Translation Completeness Report');
  console.log('='.repeat(80));

  // Overall summary
  console.log(`\n📊 ${colorize('Overall Summary', 'cyan')}:`);
  console.log(`Total translation keys: ${colorize(completeness[DEFAULT_LOCALE].total, 'cyan')}`);
  console.log(`Supported locales: ${colorize(supportedLocales.length, 'cyan')}`);

  // Per-locale breakdown
  console.log(`\n🌐 ${colorize('Locale Breakdown', 'cyan')}:`);

  supportedLocales.forEach(locale => {
    const data = completeness[locale];
    const percentage = parseFloat(data.percentage);

    let statusColor = 'red';
    let statusIcon = '❌';

    if (percentage === 100) {
      statusColor = 'green';
      statusIcon = '✅';
    } else if (percentage >= 90) {
      statusColor = 'yellow';
      statusIcon = '⚠️';
    }

    console.log(`\n  ${statusIcon} ${colorize(locale.toUpperCase(), 'cyan')}:`);
    console.log(`    Completion: ${colorize(`${data.percentage}%`, statusColor)} (${data.translated}/${data.total})`);

    if (data.missing.length > 0) {
      console.log(`    Missing: ${colorize(data.missing.length, 'red')} keys`);
    }

    if (data.empty.length > 0) {
      console.log(`    Empty: ${colorize(data.empty.length, 'yellow')} keys`);
    }
  });

  // Detailed missing translations
  supportedLocales.forEach(locale => {
    const data = completeness[locale];

    if (data.missing.length > 0 || data.empty.length > 0) {
      console.log(`\n📋 ${colorize(`Missing Translations for ${locale.toUpperCase()}`, 'cyan')}:`);

      if (data.missing.length > 0) {
        const { priority, regular } = identifyPriorityTranslations(data.missing);

        if (priority.length > 0) {
          console.log(`\n  ${colorize('🔥 Priority (Core Features):', 'red')}`);
          priority.slice(0, 10).forEach(key => {
            console.log(`    • ${key}`);
          });

          if (priority.length > 10) {
            console.log(`    ... and ${priority.length - 10} more priority keys`);
          }
        }

        if (regular.length > 0) {
          console.log(`\n  ${colorize('📝 Regular:', 'yellow')}`);
          regular.slice(0, 5).forEach(key => {
            console.log(`    • ${key}`);
          });

          if (regular.length > 5) {
            console.log(`    ... and ${regular.length - 5} more keys`);
          }
        }
      }

      if (data.empty.length > 0) {
        console.log(`\n  ${colorize('⚪ Empty Values:', 'yellow')}`);
        data.empty.slice(0, 5).forEach(key => {
          console.log(`    • ${key}`);
        });

        if (data.empty.length > 5) {
          console.log(`    ... and ${data.empty.length - 5} more empty keys`);
        }
      }
    }
  });

  return completeness;
}

/**
 * Generate actionable recommendations
 */
function generateRecommendations(completeness, supportedLocales) {
  console.log(`\n💡 ${colorize('Recommendations', 'cyan')}:`);

  const incompleteLocales = supportedLocales.filter(locale =>
    locale !== DEFAULT_LOCALE && parseFloat(completeness[locale].percentage) < 100
  );

  if (incompleteLocales.length === 0) {
    logSuccess('All locales are 100% complete!');
    return;
  }

  // Priority order by completion percentage
  const sortedLocales = incompleteLocales.sort((a, b) =>
    parseFloat(completeness[b].percentage) - parseFloat(completeness[a].percentage)
  );

  console.log('\n1. 🎯 Focus on these locales in order:');
  sortedLocales.forEach((locale, index) => {
    const data = completeness[locale];
    const priorityMissing = identifyPriorityTranslations(data.missing).priority.length;

    console.log(`   ${index + 1}. ${locale}: ${data.percentage}% complete`);
    if (priorityMissing > 0) {
      console.log(`      - ${priorityMissing} priority translations needed`);
    }
  });

  // Most critical keys across all locales
  const allMissingKeys = {};
  incompleteLocales.forEach(locale => {
    completeness[locale].missing.forEach(key => {
      allMissingKeys[key] = (allMissingKeys[key] || 0) + 1;
    });
  });

  const criticalKeys = Object.entries(allMissingKeys)
    .filter(([key, count]) => count >= Math.ceil(incompleteLocales.length / 2))
    .sort(([,a], [,b]) => b - a)
    .slice(0, 10);

  if (criticalKeys.length > 0) {
    console.log('\n2. 🔑 Most critical missing keys (affecting multiple locales):');
    criticalKeys.forEach(([key, count]) => {
      console.log(`   • ${key} (missing in ${count}/${incompleteLocales.length} locales)`);
    });
  }

  // Translation workflow suggestions
  console.log('\n3. 📋 Next Steps:');
  console.log('   • Start with priority keys in core features');
  console.log('   • Use translation management tools for batch updates');
  console.log('   • Review context for complex translations');
  console.log('   • Test translations in UI for proper fit');
  console.log('   • Consider placeholder values for missing translations');
}

/**
 * Check for potential issues
 */
function checkTranslationIssues(translations, supportedLocales) {
  console.log(`\n🔍 ${colorize('Translation Quality Check', 'cyan')}:`);

  let issuesFound = false;

  supportedLocales.forEach(locale => {
    if (locale === DEFAULT_LOCALE) return;

    const defaultFlat = flattenTranslations(translations[DEFAULT_LOCALE]);
    const localeFlat = flattenTranslations(translations[locale]);

    // Check for copy-paste errors (same as default locale)
    const sameAsDefault = [];
    Object.keys(defaultFlat).forEach(key => {
      if (localeFlat[key] && localeFlat[key] === defaultFlat[key] &&
          typeof localeFlat[key] === 'string' && localeFlat[key].length > 3) {
        sameAsDefault.push(key);
      }
    });

    if (sameAsDefault.length > 0) {
      issuesFound = true;
      logWarning(`${locale}: ${sameAsDefault.length} translations identical to ${DEFAULT_LOCALE}`);
      if (sameAsDefault.length <= 5) {
        sameAsDefault.forEach(key => console.log(`     • ${key}`));
      } else {
        sameAsDefault.slice(0, 3).forEach(key => console.log(`     • ${key}`));
        console.log(`     ... and ${sameAsDefault.length - 3} more`);
      }
    }

    // Check for very short translations that might be incomplete
    const suspiciouslyShort = [];
    Object.keys(localeFlat).forEach(key => {
      if (defaultFlat[key] && defaultFlat[key].length > 10 &&
          localeFlat[key] && localeFlat[key].length < 3) {
        suspiciouslyShort.push(key);
      }
    });

    if (suspiciouslyShort.length > 0) {
      issuesFound = true;
      logWarning(`${locale}: ${suspiciouslyShort.length} suspiciously short translations`);
      suspiciouslyShort.slice(0, 3).forEach(key => {
        console.log(`     • ${key}: "${localeFlat[key]}" (original: "${defaultFlat[key].substring(0, 30)}...")`);
      });
    }
  });

  if (!issuesFound) {
    logSuccess('No translation quality issues detected');
  }
}

/**
 * Main function
 */
function checkTranslationCompleteness() {
  logInfo('🌍 Checking translation completeness...');

  const { translations, supportedLocales } = loadTranslationFiles();

  if (supportedLocales.length === 0) {
    logError('No translation files found');
    process.exit(1);
  }

  if (!supportedLocales.includes(DEFAULT_LOCALE)) {
    logError(`Default locale '${DEFAULT_LOCALE}' not found in available locales: ${supportedLocales.join(', ')}`);
    process.exit(1);
  }

  logInfo(`Found locales: ${supportedLocales.join(', ')}`);

  const completeness = calculateCompleteness(translations, supportedLocales);
  generateCompletionReport(completeness, supportedLocales);
  checkTranslationIssues(translations, supportedLocales);
  generateRecommendations(completeness, supportedLocales);

  // Check if any locale is critically incomplete
  const criticalThreshold = 80; // 80% completion threshold
  const criticallyIncomplete = supportedLocales.filter(locale =>
    locale !== DEFAULT_LOCALE && parseFloat(completeness[locale].percentage) < criticalThreshold
  );

  console.log('\n' + '='.repeat(80));

  if (criticallyIncomplete.length > 0) {
    logWarning(`${criticallyIncomplete.length} locale(s) below ${criticalThreshold}% completion threshold`);
    console.log('\nConsider:\n• Adding placeholder translations for missing keys\n• Prioritizing critical user-facing translations\n• Setting up automated translation workflows');
  } else {
    logSuccess('All locales meet minimum completion requirements');
  }

  // Output summary for CI/CD
  const avgCompletion = supportedLocales
    .filter(locale => locale !== DEFAULT_LOCALE)
    .reduce((sum, locale) => sum + parseFloat(completeness[locale].percentage), 0) /
    (supportedLocales.length - 1);

  console.log(`\nAverage completion: ${avgCompletion.toFixed(1)}%`);
}

// Handle command line execution
if (require.main === module) {
  checkTranslationCompleteness();
}

module.exports = {
  checkTranslationCompleteness,
  loadTranslationFiles,
  flattenTranslations,
  calculateCompleteness,
  identifyPriorityTranslations,
};
