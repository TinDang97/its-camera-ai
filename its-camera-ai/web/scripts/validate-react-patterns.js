#!/usr/bin/env node

/**
 * React 19 Patterns Validation Script
 *
 * This script validates React 19+ specific patterns and concurrent features:
 * - useOptimistic usage for immediate UI feedback
 * - useDeferredValue for expensive operations
 * - useTransition for non-blocking updates
 * - Proper 'use client' directive usage
 * - Suspense boundaries and error handling
 * - Modern hook patterns and best practices
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

function logSuggestion(message) {
  console.log(colorize(`💡 ${message}`, 'magenta'));
}

/**
 * Check for useOptimistic opportunities
 */
function validateOptimisticUpdates(content, filePath) {
  const suggestions = [];
  const warnings = [];

  // Look for loading states that could benefit from optimistic updates
  if (content.includes('useState') && (content.includes('loading') || content.includes('pending') || content.includes('submitting'))) {
    if (!content.includes('useOptimistic')) {
      suggestions.push(`${filePath}: Consider using useOptimistic for immediate UI feedback on user actions`);
    }
  }

  // Check for form submissions without optimistic updates
  if (content.includes('onSubmit') || content.includes('fetch') || content.includes('POST')) {
    if (content.includes('useState') && !content.includes('useOptimistic')) {
      suggestions.push(`${filePath}: Form submissions could benefit from useOptimistic for better UX`);
    }
  }

  // Validate proper useOptimistic usage
  if (content.includes('useOptimistic')) {
    // Check if reducer function is properly defined
    if (!content.includes('function') && !content.includes('=>')) {
      warnings.push(`${filePath}: useOptimistic requires a reducer function as second argument`);
    }

    // Check if optimistic updates are actually being used
    if (!content.includes('startOptimisticUpdate') && !content.includes('optimistic')) {
      warnings.push(`${filePath}: useOptimistic is imported but not being used effectively`);
    }
  }

  return { suggestions, warnings };
}

/**
 * Check for useDeferredValue opportunities
 */
function validateDeferredValues(content, filePath) {
  const suggestions = [];
  const warnings = [];

  // Look for expensive operations that could be deferred
  const expensiveOperations = ['filter', 'map', 'sort', 'reduce', 'find', 'search'];
  const hasExpensiveOps = expensiveOperations.some(op => content.includes(`.${op}(`));

  if (hasExpensiveOps && content.includes('useState')) {
    if (!content.includes('useDeferredValue') && !content.includes('useMemo')) {
      suggestions.push(`${filePath}: Expensive array operations could benefit from useDeferredValue or useMemo`);
    }
  }

  // Look for search/filter input patterns
  if (content.includes('input') && content.includes('onChange') && hasExpensiveOps) {
    if (!content.includes('useDeferredValue')) {
      suggestions.push(`${filePath}: Search/filter inputs should use useDeferredValue for better performance`);
    }
  }

  // Validate proper useDeferredValue usage
  if (content.includes('useDeferredValue')) {
    // Check if the deferred value is actually used
    const deferredMatches = content.match(/const\s+(\w+)\s*=\s*useDeferredValue/g);
    if (deferredMatches) {
      deferredMatches.forEach(match => {
        const varName = match.match(/const\s+(\w+)/)[1];
        if (!content.includes(varName) || content.split(varName).length <= 2) {
          warnings.push(`${filePath}: useDeferredValue result '${varName}' doesn't appear to be used`);
        }
      });
    }
  }

  return { suggestions, warnings };
}

/**
 * Check for useTransition opportunities
 */
function validateTransitions(content, filePath) {
  const suggestions = [];
  const warnings = [];

  // Look for state updates that could be non-urgent
  if (content.includes('setState') || content.includes('set') && content.includes('useState')) {
    if (!content.includes('useTransition') && !content.includes('startTransition')) {
      // Check for specific patterns that benefit from transitions
      if (content.includes('navigate') || content.includes('router') || content.includes('search')) {
        suggestions.push(`${filePath}: Navigation/search state updates should use useTransition`);
      }

      if (content.includes('tab') || content.includes('filter') || content.includes('sort')) {
        suggestions.push(`${filePath}: UI state changes could use useTransition for better responsiveness`);
      }
    }
  }

  // Validate proper useTransition usage
  if (content.includes('useTransition')) {
    // Check if isPending is used to show loading state
    if (!content.includes('isPending')) {
      warnings.push(`${filePath}: useTransition's isPending should be used to show loading states`);
    }

    // Check if startTransition is actually called
    if (!content.includes('startTransition')) {
      warnings.push(`${filePath}: useTransition's startTransition function should be used`);
    }
  }

  return { suggestions, warnings };
}

/**
 * Validate 'use client' directive usage
 */
function validateClientDirective(content, filePath) {
  const errors = [];
  const warnings = [];

  const hasClientDirective = content.includes("'use client'") || content.includes('"use client"');

  // Interactive features that require 'use client'
  const clientFeatures = [
    'useState', 'useEffect', 'useReducer', 'useContext',
    'onClick', 'onChange', 'onSubmit', 'onFocus', 'onBlur',
    'addEventListener', 'document.', 'window.',
    'localStorage', 'sessionStorage'
  ];

  const hasClientFeatures = clientFeatures.some(feature => content.includes(feature));

  if (hasClientFeatures && !hasClientDirective) {
    errors.push(`${filePath}: Interactive component requires 'use client' directive`);
  }

  // Check for server-only code with client directive
  if (hasClientDirective) {
    const serverOnlyFeatures = ['cookies()', 'headers()', 'notFound()', 'redirect()'];
    const hasServerFeatures = serverOnlyFeatures.some(feature => content.includes(feature));

    if (hasServerFeatures) {
      errors.push(`${filePath}: Client component cannot use server-only features`);
    }
  }

  // Validate directive placement
  if (hasClientDirective) {
    const lines = content.split('\n');
    const directiveLine = lines.findIndex(line =>
      line.includes("'use client'") || line.includes('"use client"')
    );

    if (directiveLine > 3) { // Allow for some imports/comments at top
      warnings.push(`${filePath}: 'use client' directive should be at the top of the file`);
    }
  }

  return { errors, warnings };
}

/**
 * Validate Suspense boundaries and error handling
 */
function validateSuspensePatterns(content, filePath) {
  const suggestions = [];
  const warnings = [];

  // Check for async operations without Suspense
  const hasAsyncOps = content.includes('async') || content.includes('fetch') || content.includes('Promise');
  const hasSuspense = content.includes('Suspense') || content.includes('<Suspense');

  if (hasAsyncOps && !hasSuspense && !content.includes('use client')) {
    suggestions.push(`${filePath}: Server components with async operations should be wrapped in Suspense`);
  }

  // Check for proper fallback usage
  if (content.includes('<Suspense')) {
    if (!content.includes('fallback')) {
      warnings.push(`${filePath}: Suspense boundary should include fallback prop`);
    }
  }

  // Check for error boundaries with Suspense
  if (hasSuspense && !content.includes('ErrorBoundary') && !content.includes('try')) {
    suggestions.push(`${filePath}: Suspense boundaries should include error handling`);
  }

  return { suggestions, warnings };
}

/**
 * Validate modern hook patterns
 */
function validateModernHookPatterns(content, filePath) {
  const suggestions = [];
  const warnings = [];

  // Check for useCallback/useMemo opportunities
  if (content.includes('function') && content.includes('props')) {
    if (!content.includes('useCallback') && !content.includes('useMemo')) {
      if (content.includes('onClick') || content.includes('onSubmit') || content.includes('onChange')) {
        suggestions.push(`${filePath}: Event handlers should be wrapped in useCallback for performance`);
      }
    }
  }

  // Check for proper dependency arrays
  const hookDeps = ['useEffect', 'useCallback', 'useMemo'];
  hookDeps.forEach(hook => {
    if (content.includes(hook)) {
      // Look for missing dependency arrays
      const hookPattern = new RegExp(`${hook}\\([^)]*\\)(?!\\s*,\\s*\\[)`, 'g');
      if (hookPattern.test(content)) {
        warnings.push(`${filePath}: ${hook} may be missing dependency array`);
      }
    }
  });

  // Check for ref patterns
  if (content.includes('useRef')) {
    if (content.includes('current') && !content.includes('HTMLElement')) {
      suggestions.push(`${filePath}: Consider adding TypeScript types for useRef`);
    }
  }

  return { suggestions, warnings };
}

/**
 * Validate React 19 concurrent rendering patterns
 */
function validateConcurrentRendering(content, filePath) {
  const suggestions = [];
  const warnings = [];

  // Check for proper error boundary usage with concurrent features
  const concurrentFeatures = ['useOptimistic', 'useDeferredValue', 'useTransition'];
  const hasConcurrentFeatures = concurrentFeatures.some(feature => content.includes(feature));

  if (hasConcurrentFeatures && !content.includes('ErrorBoundary')) {
    suggestions.push(`${filePath}: Components using concurrent features should include error boundaries`);
  }

  // Check for proper loading states
  if (content.includes('isPending') || content.includes('isLoading')) {
    if (!content.includes('Skeleton') && !content.includes('Loading') && !content.includes('Spinner')) {
      suggestions.push(`${filePath}: Loading states should show meaningful UI feedback`);
    }
  }

  // Validate streaming patterns
  if (content.includes('Suspense') && content.includes('lazy')) {
    if (!content.includes('preload') && !content.includes('prefetch')) {
      suggestions.push(`${filePath}: Consider preloading lazy components for better UX`);
    }
  }

  return { suggestions, warnings };
}

/**
 * Validate a single React file
 */
function validateReactPatterns(filePath) {
  const errors = [];
  const warnings = [];
  const suggestions = [];

  if (!fs.existsSync(filePath)) {
    errors.push(`File not found: ${filePath}`);
    return { errors, warnings, suggestions };
  }

  let content;
  try {
    content = fs.readFileSync(filePath, 'utf8');
  } catch (error) {
    errors.push(`Error reading file ${filePath}: ${error.message}`);
    return { errors, warnings, suggestions };
  }

  // Skip non-React files
  if (!content.includes('react') && !content.includes('React') && !filePath.includes('.tsx')) {
    return { errors, warnings, suggestions };
  }

  // Run all validations
  const validations = [
    validateOptimisticUpdates(content, filePath),
    validateDeferredValues(content, filePath),
    validateTransitions(content, filePath),
    validateClientDirective(content, filePath),
    validateSuspensePatterns(content, filePath),
    validateModernHookPatterns(content, filePath),
    validateConcurrentRendering(content, filePath),
  ];

  // Collect all results
  validations.forEach(result => {
    if (result.errors) errors.push(...result.errors);
    if (result.warnings) warnings.push(...result.warnings);
    if (result.suggestions) suggestions.push(...result.suggestions);
  });

  return { errors, warnings, suggestions };
}

/**
 * Main validation function
 */
function validateReactPatternsInFiles(filePaths) {
  logInfo('🚀 Validating React 19 patterns...');

  let totalErrors = 0;
  let totalWarnings = 0;
  let totalSuggestions = 0;
  const validatedFiles = [];

  // If no file paths provided, get them from command line arguments
  if (!filePaths || filePaths.length === 0) {
    filePaths = process.argv.slice(2);
  }

  // If still no paths, scan React files
  if (filePaths.length === 0) {
    const reactDirs = [
      path.join(__dirname, '..', 'components'),
      path.join(__dirname, '..', 'app'),
      path.join(__dirname, '..', 'hooks'),
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
          } else if (file.match(/\.(tsx|ts|jsx|js)$/)) {
            filePaths.push(fullPath);
          }
        });
      } catch (error) {
        // Ignore directory read errors
      }
    }

    reactDirs.forEach(dir => {
      if (fs.existsSync(dir)) {
        scanDirectory(dir);
      }
    });
  }

  if (filePaths.length === 0) {
    logWarning('No React files found to validate');
    return;
  }

  logInfo(`Validating ${filePaths.length} React files...`);

  // Validate each file
  filePaths.forEach(filePath => {
    const { errors, warnings, suggestions } = validateReactPatterns(filePath);

    if (errors.length > 0 || warnings.length > 0 || suggestions.length > 0) {
      validatedFiles.push({
        filePath,
        errors,
        warnings,
        suggestions,
      });
    }

    totalErrors += errors.length;
    totalWarnings += warnings.length;
    totalSuggestions += suggestions.length;
  });

  // Display results
  if (validatedFiles.length > 0) {
    console.log('\n' + '='.repeat(80));
    logInfo('React 19 Patterns Validation Results:');
    console.log('='.repeat(80));

    validatedFiles.forEach(({ filePath, errors, warnings, suggestions }) => {
      const relativePath = path.relative(process.cwd(), filePath);
      console.log(`\n${colorize(relativePath, 'cyan')}:`);

      errors.forEach(error => logError(error));
      warnings.forEach(warning => logWarning(warning));
      suggestions.forEach(suggestion => logSuggestion(suggestion));
    });
  }

  // Summary
  console.log('\n' + '='.repeat(80));

  if (totalErrors === 0) {
    logSuccess('React patterns validation passed!');
    if (totalWarnings > 0 || totalSuggestions > 0) {
      logInfo(`Found ${totalWarnings} warning(s) and ${totalSuggestions} suggestion(s) for improvement`);
    }

    if (totalSuggestions > 0) {
      console.log('\n💡 React 19 Performance Tips:');
      console.log('• Use useOptimistic for immediate UI feedback');
      console.log('• Use useDeferredValue for expensive filtering/search operations');
      console.log('• Use useTransition for non-urgent state updates');
      console.log('• Wrap async server components in Suspense boundaries');
      console.log('• Include error boundaries for better error handling');
    }
  } else {
    logError(`React patterns validation failed with ${totalErrors} error(s)`);
    process.exit(1);
  }
}

// Handle command line execution
if (require.main === module) {
  validateReactPatternsInFiles();
}

module.exports = {
  validateReactPatternsInFiles,
  validateReactPatterns,
  validateOptimisticUpdates,
  validateDeferredValues,
  validateTransitions,
  validateClientDirective,
  validateSuspensePatterns,
  validateModernHookPatterns,
  validateConcurrentRendering,
};
