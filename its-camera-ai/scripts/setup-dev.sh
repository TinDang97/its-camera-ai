#!/bin/bash
# Development Environment Setup Script for ITS Camera AI
# This script helps developers quickly set up their local development environment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check system requirements
check_requirements() {
    print_info "Checking system requirements..."
    
    # Check Python version
    if command_exists python3.12; then
        print_success "Python 3.12 found"
    else
        print_error "Python 3.12 is required but not found. Please install Python 3.12+"
        exit 1
    fi
    
    # Check Docker
    if command_exists docker; then
        print_success "Docker found"
    else
        print_error "Docker is required but not found. Please install Docker"
        exit 1
    fi
    
    # Check Docker Compose
    if command_exists "docker compose" || command_exists docker-compose; then
        print_success "Docker Compose found"
    else
        print_error "Docker Compose is required but not found. Please install Docker Compose"
        exit 1
    fi
    
    # Check uv
    if command_exists uv; then
        print_success "uv package manager found"
    else
        print_warning "uv package manager not found. Installing..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        source $HOME/.cargo/env
        print_success "uv installed successfully"
    fi
}

# Setup environment file
setup_env_file() {
    print_info "Setting up environment file..."
    
    if [ ! -f .env ]; then
        cp .env.example .env
        print_success "Environment file created from template"
        print_warning "Please review and update .env file with your specific settings"
    else
        print_warning "Environment file already exists, skipping..."
    fi
}

# Install dependencies
install_dependencies() {
    print_info "Installing Python dependencies..."
    
    # Install development dependencies
    uv sync --group dev --group ml
    
    print_success "Dependencies installed successfully"
}

# Setup pre-commit hooks
setup_pre_commit() {
    print_info "Setting up pre-commit hooks..."
    
    uv run pre-commit install
    
    print_success "Pre-commit hooks installed"
}

# Create necessary directories
create_directories() {
    print_info "Creating necessary directories..."
    
    mkdir -p data logs models temp backups
    
    print_success "Directories created"
}

# Setup database
setup_database() {
    print_info "Setting up database..."
    
    # Start PostgreSQL container
    docker compose --profile dev up -d postgres
    
    # Wait for PostgreSQL to be ready
    print_info "Waiting for PostgreSQL to be ready..."
    sleep 10
    
    # Run database migrations (if they exist)
    if [ -f alembic.ini ]; then
        uv run alembic upgrade head
        print_success "Database migrations applied"
    else
        print_warning "No Alembic configuration found, skipping migrations"
    fi
}

# Main setup function
main() {
    echo -e "${BLUE}"
    echo "╔═══════════════════════════════════════╗"
    echo "║        ITS Camera AI Setup           ║"
    echo "║   Development Environment Setup      ║"
    echo "╚═══════════════════════════════════════╝"
    echo -e "${NC}"
    
    check_requirements
    setup_env_file
    install_dependencies
    setup_pre_commit
    create_directories
    
    # Ask if user wants to setup database
    echo
    read -p "Do you want to setup the database now? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        setup_database
    fi
    
    # Success message
    echo
    print_success "Development environment setup complete!"
    echo
    print_info "Next steps:"
    echo "  1. Review and update the .env file with your settings"
    echo "  2. Run 'make dev' to start the development environment"
    echo "  3. Run 'make test' to run the tests"
    echo "  4. Visit http://localhost:8000/docs for API documentation"
    echo
    print_info "Useful commands:"
    echo "  - make help          : Show all available commands"
    echo "  - make docker-up     : Start all services"
    echo "  - make test          : Run tests"
    echo "  - make format        : Format code"
    echo "  - make jupyter       : Start Jupyter Lab for ML development"
    echo
}

# Run main function
main "$@"