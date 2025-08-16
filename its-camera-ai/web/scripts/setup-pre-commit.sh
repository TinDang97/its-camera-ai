#!/bin/bash

# Pre-commit Setup and Validation Script for Next.js Frontend
# This script sets up and validates the pre-commit configuration

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

log_header() {
    echo -e "\n${CYAN}======================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}======================================${NC}\n"
}

# Check if we're in the correct directory
if [ ! -f "package.json" ] || [ ! -d "../" ] || [ ! -f "../.pre-commit-config.yaml" ]; then
    log_error "Please run this script from the web/ directory"
    exit 1
fi

log_header "🚀 Setting up Pre-commit for Next.js Frontend"

# Step 1: Install npm dependencies
log_info "Installing npm dependencies..."
if command -v npm &> /dev/null; then
    npm install
    log_success "npm dependencies installed"
else
    log_error "npm not found. Please install Node.js and npm"
    exit 1
fi

# Step 2: Install pre-commit (Python)
log_info "Checking pre-commit installation..."
if command -v pre-commit &> /dev/null; then
    log_success "pre-commit is already installed"
else
    log_info "Installing pre-commit..."
    if command -v pip &> /dev/null; then
        pip install pre-commit
        log_success "pre-commit installed via pip"
    elif command -v pip3 &> /dev/null; then
        pip3 install pre-commit
        log_success "pre-commit installed via pip3"
    elif command -v brew &> /dev/null; then
        brew install pre-commit
        log_success "pre-commit installed via brew"
    else
        log_error "Could not install pre-commit. Please install Python/pip or Homebrew"
        exit 1
    fi
fi

# Step 3: Install husky
log_info "Setting up Husky..."
npx husky install
log_success "Husky installed"

# Step 4: Install pre-commit hooks
log_info "Installing pre-commit hooks..."
cd ..
pre-commit install
pre-commit install --hook-type commit-msg
cd web
log_success "Pre-commit hooks installed"

# Step 5: Validate configuration files
log_header "🔍 Validating Configuration Files"

# Check ESLint config
log_info "Validating ESLint configuration..."
if npx eslint --print-config . > /dev/null 2>&1; then
    log_success "ESLint configuration is valid"
else
    log_error "ESLint configuration has issues"
fi

# Check Prettier config
log_info "Validating Prettier configuration..."
if npx prettier --check --config .prettierrc.json . --log-level=error > /dev/null 2>&1; then
    log_success "Prettier configuration is valid"
else
    log_warning "Prettier may need to format some files"
fi

# Check TypeScript config
log_info "Validating TypeScript configuration..."
if npx tsc --noEmit > /dev/null 2>&1; then
    log_success "TypeScript configuration is valid"
else
    log_warning "TypeScript has some issues that may need attention"
fi

# Step 6: Test validation scripts
log_header "🧪 Testing Validation Scripts"

# Test i18n validation
log_info "Testing i18n validation script..."
if node scripts/validate-i18n.js > /dev/null 2>&1; then
    log_success "i18n validation script works"
else
    log_warning "i18n validation found issues (this may be expected)"
fi

# Test component validation
log_info "Testing component validation script..."
if node scripts/validate-components.js components/ui/button.tsx > /dev/null 2>&1; then
    log_success "Component validation script works"
else
    log_warning "Component validation found issues (this may be expected)"
fi

# Test React patterns validation
log_info "Testing React patterns validation script..."
if node scripts/validate-react-patterns.js app/ > /dev/null 2>&1; then
    log_success "React patterns validation script works"
else
    log_warning "React patterns validation found issues (this may be expected)"
fi

# Test accessibility validation
log_info "Testing accessibility validation script..."
if node scripts/validate-accessibility.js components/ > /dev/null 2>&1; then
    log_success "Accessibility validation script works"
else
    log_warning "Accessibility validation found issues (this may be expected)"
fi

# Test translation completeness
log_info "Testing translation completeness script..."
if node scripts/check-translation-completeness.js > /dev/null 2>&1; then
    log_success "Translation completeness script works"
else
    log_warning "Translation completeness found issues (this may be expected)"
fi

# Step 7: Test pre-commit hook execution
log_header "🔧 Testing Pre-commit Hook Execution"

log_info "Testing frontend lint-staged hook..."
cd ..
if pre-commit run frontend-lint-staged --files web/package.json > /dev/null 2>&1; then
    log_success "Frontend lint-staged hook works"
else
    log_warning "Frontend lint-staged hook needs attention"
fi

log_info "Testing frontend type-check hook..."
if pre-commit run frontend-type-check --files web/tsconfig.json > /dev/null 2>&1; then
    log_success "Frontend type-check hook works"
else
    log_warning "Frontend type-check hook needs attention"
fi

log_info "Testing frontend i18n validation hook..."
if pre-commit run frontend-i18n-validation --files web/messages/en.json > /dev/null 2>&1; then
    log_success "Frontend i18n validation hook works"
else
    log_warning "Frontend i18n validation hook needs attention"
fi

cd web

# Step 8: Performance and size validation
log_header "📊 Performance and Size Validation"

log_info "Checking bundle size analysis script..."
if [ -d ".next" ]; then
    if node scripts/check-bundle-size.js > /dev/null 2>&1; then
        log_success "Bundle size analysis script works"
    else
        log_warning "Bundle size analysis needs a production build"
    fi
else
    log_warning "No build found - run 'npm run build' to test bundle analysis"
fi

# Step 9: Generate summary report
log_header "📋 Setup Summary Report"

echo "Pre-commit setup completed! Here's what was configured:"
echo ""
echo "🔧 Core Configuration:"
echo "  ✓ ESLint with Next.js 15+ and React 19 rules"
echo "  ✓ Prettier with Tailwind CSS plugin"
echo "  ✓ TypeScript strict mode configuration"
echo "  ✓ Lint-staged for incremental checks"
echo ""
echo "🧪 Validation Scripts:"
echo "  ✓ i18n translation validation"
echo "  ✓ Component structure validation"
echo "  ✓ React 19 patterns validation"
echo "  ✓ Accessibility compliance checking"
echo "  ✓ Bundle size analysis"
echo ""
echo "🎯 Pre-commit Hooks:"
echo "  ✓ Code formatting and linting"
echo "  ✓ TypeScript type checking"
echo "  ✓ Translation validation"
echo "  ✓ Security scanning (pre-push)"
echo "  ✓ Accessibility checks (pre-push)"
echo ""
echo "📚 Available Commands:"
echo "  npm run lint          - Run ESLint"
echo "  npm run format         - Format code with Prettier"
echo "  npm run type-check     - Run TypeScript checks"
echo "  npm run validate:all   - Run all validation scripts"
echo "  npm run analyze:size   - Analyze bundle size"
echo ""
echo "🔄 Git Workflow:"
echo "  - Pre-commit: Formatting, linting, type checking"
echo "  - Pre-push: Full validation, security, accessibility"
echo ""

log_success "Pre-commit setup is complete!"
log_info "Try making a commit to see the hooks in action"

exit 0
