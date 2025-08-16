#!/usr/bin/env node

/**
 * Component Validation Script for Next.js React 19+ Application
 *
 * This script validates React components for:
 * - Proper export patterns
 * - TypeScript usage and imports
 * - Accessibility attributes
 * - React 19 concurrent features usage
 * - Error boundary patterns
 * - Performance best practices
 * - Design system compliance
 */

const fs = require('fs');
const path = require('path');

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
 * Check if file has proper export patterns
 */
function validateExports(content, filePath) {
  const errors = [];
  const warnings = [];

  // Check for export patterns
  const hasDefaultExport = content.includes('export default');
  const hasNamedExports = content.includes('export {') || /export (const|function|class|interface|type)/.test(content);

  if (!hasDefaultExport && !hasNamedExports && content.length > 100) {
    errors.push(`${filePath}: Missing export statement`);
  }

  // Check for proper component export patterns
  if (filePath.includes('/components/') && content.includes('function') && !hasDefaultExport && !hasNamedExports) {
    warnings.push(`${filePath}: Component should be exported`);
  }

  return { errors, warnings };
}

/**
 * Validate TypeScript usage and imports
 */
function validateTypeScript(content, filePath) {
  const errors = [];
  const warnings = [];

  if (!filePath.endsWith('.ts') && !filePath.endsWith('.tsx')) {
    return { errors, warnings };
  }

  // Check for missing imports in non-trivial files
  if (content.length > 100 && !content.includes('import') && !content.includes('require')) {
    warnings.push(`${filePath}: File may be missing imports`);
  }

  // Check for any usage (should use proper types)
  if (content.includes(': any') || content.includes('<any>') || content.includes('as any')) {
    warnings.push(`${filePath}: Contains 'any' type - consider using specific types`);
  }

  // Check for missing interface/type definitions for props
  if (content.includes('function') && content.includes('props') && !content.includes('interface') && !content.includes('type')) {
    warnings.push(`${filePath}: Consider defining prop types with interface or type`);
  }

  return { errors, warnings };
}

/**
 * Validate accessibility attributes
 */
function validateAccessibility(content, filePath) {
  const errors = [];
  const warnings = [];

  // Check buttons without aria-label or accessible text
  const buttonMatches = content.match(/<button[^>]*>/g);
  if (buttonMatches) {
    buttonMatches.forEach((button, index) => {
      if (!button.includes('aria-label') && !button.includes('aria-labelledby') && !button.includes('aria-describedby')) {
        warnings.push(`${filePath}: Button element #${index + 1} may need aria-label for accessibility`);
      }
    });
  }

  // Check img elements without alt attributes
  const imgMatches = content.match(/<img[^>]*>/g);
  if (imgMatches) {
    imgMatches.forEach((img, index) => {
      if (!img.includes('alt=')) {
        errors.push(`${filePath}: Image element #${index + 1} missing alt attribute`);
      }
    });
  }

  // Check input elements without labels
  const inputMatches = content.match(/<input[^>]*>/g);
  if (inputMatches) {
    inputMatches.forEach((input, index) => {
      if (!input.includes('aria-label') && !input.includes('aria-labelledby') && !content.includes('<label')) {
        warnings.push(`${filePath}: Input element #${index + 1} may need label or aria-label`);
      }
    });
  }

  // Check for click handlers without keyboard events
  if (content.includes('onClick') && !content.includes('onKeyDown') && !content.includes('onKeyPress')) {
    const divClickMatches = content.match(/<div[^>]*onClick/g);
    if (divClickMatches && divClickMatches.length > 0) {
      warnings.push(`${filePath}: Clickable div elements should include keyboard event handlers`);
    }
  }

  return { errors, warnings };
}

/**
 * Validate React 19 concurrent features usage
 */
function validateReact19Patterns(content, filePath) {
  const warnings = [];
  const suggestions = [];

  // Check for potential useOptimistic usage
  if (content.includes('useState') && (content.includes('loading') || content.includes('pending'))) {
    suggestions.push(`${filePath}: Consider using useOptimistic for better user experience with loading states`);
  }

  // Check for expensive operations that could use useDeferredValue
  if (content.includes('filter') || content.includes('map') || content.includes('sort')) {
    if (!content.includes('useDeferredValue') && !content.includes('useMemo')) {
      suggestions.push(`${filePath}: Consider using useDeferredValue or useMemo for expensive filtering/mapping operations`);
    }
  }

  // Check for state updates that could use useTransition
  if (content.includes('setState') && !content.includes('useTransition')) {
    suggestions.push(`${filePath}: Consider using useTransition for non-urgent state updates`);
  }

  // Check for proper use client directive
  if (content.includes('useState') || content.includes('useEffect') || content.includes('onClick')) {
    if (!content.includes("'use client'") && !content.includes('"use client"')) {
      warnings.push(`${filePath}: Interactive component may need 'use client' directive`);
    }
  }

  return { warnings, suggestions };
}

/**
 * Validate error boundary patterns
 */
function validateErrorBoundaries(content, filePath) {
  const warnings = [];

  // Check for async operations without error handling
  if (content.includes('async') || content.includes('fetch') || content.includes('Promise')) {
    if (!content.includes('try') && !content.includes('catch') && !content.includes('ErrorBoundary')) {
      warnings.push(`${filePath}: Async operations should include error handling or be wrapped in ErrorBoundary`);
    }
  }

  // Check for stateful client components without error boundaries
  if (content.includes("'use client'") && content.includes('useState')) {
    if (!content.includes('ErrorBoundary') && !content.includes('componentDidCatch')) {
      warnings.push(`${filePath}: Stateful client components should be wrapped with ErrorBoundary`);
    }
  }

  return { warnings };
}

/**
 * Validate performance best practices
 */
function validatePerformance(content, filePath) {
  const warnings = [];

  // Check for missing React.memo on components
  if (content.includes('export default function') || content.includes('export const')) {
    if (!content.includes('memo') && !content.includes('React.memo')) {
      warnings.push(`${filePath}: Consider wrapping components with React.memo for performance`);
    }
  }

  // Check for inline object/array creation in JSX
  const inlineObjectMatches = content.match(/\w+\s*=\s*\{[^}]*\}/g);
  if (inlineObjectMatches && inlineObjectMatches.some(match => match.includes(':'))) {
    warnings.push(`${filePath}: Avoid creating objects inline in JSX - use useMemo or constants`);
  }

  // Check for missing dependency arrays in useEffect
  const useEffectMatches = content.match(/useEffect\([^)]*\)/g);
  if (useEffectMatches) {
    useEffectMatches.forEach((effect, index) => {
      if (!effect.includes('[') || effect.includes('[]')) {
        // This is either missing deps or empty deps - both might need review
        warnings.push(`${filePath}: useEffect #${index + 1} dependency array should be reviewed`);
      }
    });
  }

  return { warnings };
}

/**
 * Validate design system compliance
 */
function validateDesignSystem(content, filePath) {
  const warnings = [];

  // Check for hardcoded colors instead of design system
  const colorMatches = content.match(/#[0-9a-fA-F]{3,6}/g);
  if (colorMatches) {
    warnings.push(`${filePath}: Found hardcoded colors - use design system variables instead`);
  }

  // Check for proper Tabler Icons usage
  if (content.includes('Icon') && !content.includes('@tabler/icons-react')) {
    warnings.push(`${filePath}: Use Tabler Icons from @tabler/icons-react for consistency`);
  }

  // Check for proper className usage with Tailwind
  if (content.includes('className') && content.includes('style=')) {
    warnings.push(`${filePath}: Avoid mixing className and inline styles - prefer Tailwind classes`);
  }

  return { warnings };
}

/**
 * Validate a single component file
 */
function validateComponent(filePath) {
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

  // Run all validations
  const validations = [
    validateExports(content, filePath),
    validateTypeScript(content, filePath),
    validateAccessibility(content, filePath),
    validateReact19Patterns(content, filePath),
    validateErrorBoundaries(content, filePath),
    validatePerformance(content, filePath),
    validateDesignSystem(content, filePath),
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
function validateComponents(filePaths) {
  logInfo('Starting component validation...');

  let totalErrors = 0;
  let totalWarnings = 0;
  let totalSuggestions = 0;
  const validatedFiles = [];

  // If no file paths provided, get them from command line arguments
  if (!filePaths || filePaths.length === 0) {
    filePaths = process.argv.slice(2);
  }

  // If still no paths, scan component directories
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
          } else if (file.match(/\.(tsx|ts|jsx|js)$/)) {
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

  logInfo(`Validating ${filePaths.length} component files...`);

  // Validate each file
  filePaths.forEach(filePath => {
    const { errors, warnings, suggestions } = validateComponent(filePath);

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
    logInfo('Component Validation Results:');
    console.log('='.repeat(80));

    validatedFiles.forEach(({ filePath, errors, warnings, suggestions }) => {
      const relativePath = path.relative(process.cwd(), filePath);
      console.log(`\n${colorize(relativePath, 'cyan')}:`);

      errors.forEach(error => logError(error));
      warnings.forEach(warning => logWarning(warning));
      suggestions.forEach(suggestion => logInfo(`💡 ${suggestion}`));
    });
  }

  // Summary
  console.log('\n' + '='.repeat(80));

  if (totalErrors === 0) {
    logSuccess('Component validation passed!');
    if (totalWarnings > 0) {
      logInfo(`Found ${totalWarnings} warning(s) and ${totalSuggestions} suggestion(s)`);
    }
  } else {
    logError(`Component validation failed with ${totalErrors} error(s), ${totalWarnings} warning(s), and ${totalSuggestions} suggestion(s)`);

    console.log('\nCommon fixes:');
    console.log('1. Add proper export statements to components');
    console.log('2. Include alt attributes for all images');
    console.log('3. Add aria-labels for interactive elements');
    console.log('4. Use TypeScript interfaces for component props');
    console.log('5. Wrap async operations in error boundaries');
    console.log('6. Use React 19 concurrent features for better UX');
    console.log('7. Follow design system guidelines');

    process.exit(1);
  }
}

// Handle command line execution
if (require.main === module) {
  validateComponents();
}

module.exports = {
  validateComponents,
  validateComponent,
  validateExports,
  validateTypeScript,
  validateAccessibility,
  validateReact19Patterns,
  validateErrorBoundaries,
  validatePerformance,
  validateDesignSystem,
};
