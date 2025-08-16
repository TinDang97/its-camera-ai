# Frontend Pre-commit & Deployment Setup

This document provides a comprehensive guide for the Next.js frontend pre-commit hooks and deployment infrastructure that has been implemented for the ITS Camera AI project.

## 🎯 Overview

The frontend now includes a production-grade pre-commit configuration that ensures:
- **Code Quality**: Automated formatting, linting, and type checking
- **Performance**: Bundle size monitoring and optimization checks
- **Accessibility**: WCAG 2.1 AA compliance validation
- **Security**: Dependency vulnerability scanning
- **Internationalization**: Translation completeness validation
- **React 19 Patterns**: Modern concurrent features validation

## 🚀 Quick Start

### 1. Initial Setup

Run the automated setup script from the `web/` directory:

```bash
cd web
./scripts/setup-pre-commit.sh
```

This script will:
- Install all necessary npm dependencies
- Set up pre-commit hooks
- Validate configuration files
- Test all validation scripts

### 2. Manual Setup (Alternative)

If you prefer manual setup:

```bash
# Install dependencies
cd web
npm install

# Install pre-commit (Python required)
pip install pre-commit

# Install hooks
cd ..
pre-commit install
pre-commit install --hook-type commit-msg
cd web

# Set up Husky
npx husky install
```

## 📋 Configuration Files

### Pre-commit Configuration
- **Root**: `/.pre-commit-config.yaml` - Main configuration with backend + frontend hooks
- **Frontend**: `/web/.pre-commit-config.yaml` - Frontend-specific detailed configuration
- **Lint-staged**: `/web/.lintstagedrc.json` - Incremental file checks

### Code Quality
- **ESLint**: `/web/eslint.config.mjs` - Next.js 15+ and React 19 rules
- **Prettier**: `/web/.prettierrc.json` - Formatting with Tailwind CSS plugin
- **TypeScript**: `/web/tsconfig.json` - Strict type checking configuration

### Validation Scripts
- `/web/scripts/validate-i18n.js` - Translation validation
- `/web/scripts/validate-components.js` - Component structure checks
- `/web/scripts/validate-react-patterns.js` - React 19 patterns validation
- `/web/scripts/validate-accessibility.js` - WCAG compliance checks
- `/web/scripts/check-bundle-size.js` - Bundle size analysis
- `/web/scripts/check-translation-completeness.js` - i18n completeness

## 🔧 Available Commands

### Code Quality
```bash
npm run lint              # Run ESLint
npm run lint:fix          # Fix ESLint issues
npm run format            # Format with Prettier
npm run format:check      # Check Prettier formatting
npm run type-check        # TypeScript validation
```

### Validation
```bash
npm run validate:i18n              # Check translations
npm run validate:components        # Validate component structure
npm run validate:react-patterns    # Check React 19 usage
npm run validate:accessibility     # WCAG compliance
npm run validate:translations      # Translation completeness
npm run validate:all              # Run all validations
```

### Build & Analysis
```bash
npm run build             # Production build
npm run build:check       # Build without linting
npm run analyze:size      # Bundle size analysis
```

## 🎭 Pre-commit Hook Stages

### Pre-commit (Fast Checks)
- **Prettier**: Code formatting
- **ESLint**: Linting with auto-fix
- **TypeScript**: Type checking
- **i18n**: Translation validation
- **Component**: Structure validation

### Pre-push (Comprehensive Checks)
- **Security**: npm audit for vulnerabilities
- **Accessibility**: WCAG compliance
- **React Patterns**: React 19 best practices
- **Bundle Size**: Performance analysis
- **Build**: Production build validation

### Manual (On-demand)
- **Lighthouse**: Performance auditing
- **Bundle Analysis**: Detailed size breakdown

## 🛠️ Development Workflow

### Daily Development
1. **Write Code**: The pre-commit hooks will automatically run
2. **Commit Changes**: Hooks validate and format your code
3. **Push Changes**: Additional checks ensure quality

### Before Pull Requests
```bash
# Run comprehensive validation
npm run validate:all

# Check bundle size
npm run analyze:size

# Test production build
npm run build
```

### Fixing Common Issues

#### ESLint Errors
```bash
npm run lint:fix          # Auto-fix most issues
npm run lint              # Check remaining issues
```

#### TypeScript Errors
```bash
npm run type-check        # See all type errors
# Fix types manually or update tsconfig.json
```

#### Translation Issues
```bash
npm run validate:i18n     # See missing translations
# Add missing keys to messages/*.json files
```

#### Accessibility Issues
```bash
npm run validate:accessibility  # See accessibility violations
# Add aria-labels, alt text, semantic HTML
```

## 📊 Quality Standards

### Code Quality Gates
- **ESLint**: Zero warnings/errors
- **TypeScript**: Strict mode compliance
- **Prettier**: Consistent formatting
- **Bundle Size**: <500KB initial load
- **Accessibility**: WCAG 2.1 AA compliance

### Performance Metrics
- **Core Web Vitals**: LCP <2.5s, FID <100ms, CLS <0.1
- **Bundle Growth**: <5% increase per feature
- **Type Coverage**: >95%
- **Translation Coverage**: >90% for all locales

## 🔍 Troubleshooting

### Common Issues

#### "pre-commit not found"
```bash
pip install pre-commit
# or
brew install pre-commit
```

#### "npm audit fails"
```bash
npm audit fix           # Fix automatically
npm audit --audit-level=moderate  # Check manually
```

#### "TypeScript errors in strict mode"
- Update type annotations
- Add proper null checks
- Use type guards for unions

#### "Bundle size exceeded"
- Use dynamic imports for code splitting
- Optimize images and assets
- Remove unused dependencies

### Getting Help

#### Check Configuration
```bash
# Validate ESLint config
npx eslint --print-config .

# Check Prettier config
npx prettier --check .

# Verify TypeScript config
npx tsc --noEmit
```

#### Run Individual Validations
```bash
node scripts/validate-components.js components/ui/button.tsx
node scripts/validate-i18n.js
node scripts/check-bundle-size.js
```

#### Debug Pre-commit Hooks
```bash
# Run specific hook
pre-commit run frontend-type-check

# Run all hooks manually
pre-commit run --all-files

# Skip hooks (emergency only)
git commit --no-verify
```

## 🎨 Customization

### Adding New Validation Rules

1. **Create Script**: Add to `/web/scripts/`
2. **Update Package.json**: Add npm script
3. **Configure Pre-commit**: Add hook to `.pre-commit-config.yaml`
4. **Test**: Run and validate

### Modifying Quality Standards

#### ESLint Rules
Edit `/web/eslint.config.mjs`:
```javascript
rules: {
  "your-custom-rule": "error",
  // Disable rule
  "existing-rule": "off"
}
```

#### Bundle Size Limits
Edit `/web/scripts/check-bundle-size.js`:
```javascript
const MAX_BUNDLE_SIZE = 600 * 1024; // Increase to 600KB
```

#### TypeScript Strictness
Edit `/web/tsconfig.json`:
```json
{
  "compilerOptions": {
    "exactOptionalPropertyTypes": false  // Reduce strictness
  }
}
```

## 📈 Deployment Integration

The pre-commit configuration integrates with:

### CI/CD Pipeline
- **GitHub Actions**: Runs same checks in CI
- **Quality Gates**: Prevents merging failing code
- **Performance Monitoring**: Tracks bundle size over time

### Development Environment
- **VS Code Integration**: Real-time ESLint/Prettier
- **Hot Reload**: Fast development cycle
- **Error Reporting**: Clear feedback on issues

### Production Deployment
- **Build Validation**: Ensures deployable code
- **Security Scanning**: No vulnerable dependencies
- **Performance Monitoring**: Bundle size tracking

## 🏆 Best Practices

### Component Development
```tsx
// ✅ Good: Proper TypeScript, accessibility, React 19
'use client';

import { useOptimistic } from 'react';
import { Button } from '@/components/ui/button';

interface Props {
  onSubmit: (data: FormData) => Promise<void>;
}

export function FormComponent({ onSubmit }: Props) {
  const [optimisticState, updateOptimistic] = useOptimistic(
    { submitted: false },
    (state, newState) => ({ ...state, ...newState })
  );

  return (
    <form onSubmit={handleSubmit}>
      <Button 
        type="submit"
        aria-label="Submit form"
        disabled={optimisticState.submitted}
      >
        Submit
      </Button>
    </form>
  );
}
```

### Translation Keys
```json
// messages/en.json
{
  "common": {
    "submit": "Submit",
    "cancel": "Cancel"
  },
  "dashboard": {
    "title": "Traffic Dashboard",
    "metrics": {
      "totalCameras": "Total Cameras: {count}"
    }
  }
}
```

### Bundle Optimization
```typescript
// ✅ Good: Dynamic imports
const HeavyComponent = lazy(() => import('./HeavyComponent'));

// ✅ Good: Tree shaking
import { Button } from '@/components/ui/button';

// ❌ Bad: Import entire library
import * as lodash from 'lodash';
```

## 📝 Summary

This pre-commit setup provides:
- **Automated Quality**: Code formatting, linting, type checking
- **Performance Monitoring**: Bundle size and Core Web Vitals tracking
- **Accessibility Compliance**: WCAG 2.1 AA validation
- **Security Scanning**: Dependency vulnerability checks
- **Developer Experience**: Fast feedback and easy fixes
- **Team Consistency**: Shared standards and practices

The configuration is designed to be comprehensive yet performant, ensuring high code quality without slowing down development velocity.