#!/bin/bash
# Cleanup Script for ITS Camera AI
# Platform Engineering: Complete environment cleanup and reset

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/logs/cleanup.log"

# Functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE" 2>/dev/null || true
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1" >> "$LOG_FILE" 2>/dev/null || true
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1" >> "$LOG_FILE" 2>/dev/null || true
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE" 2>/dev/null || true
}

# Create log directory
mkdir -p "$(dirname "$LOG_FILE")" 2>/dev/null || true

log "Starting ITS Camera AI Environment Cleanup..."

# Show current Docker status
show_current_status() {
    log "Current Docker environment status:"
    
    # Show running containers
    local running_containers
    running_containers=$(docker ps --filter "name=its-camera-ai" --format "table {{.Names}}\t{{.Status}}" 2>/dev/null || echo "none")
    if [[ "$running_containers" != "none" && -n "$running_containers" ]]; then
        info "Running ITS Camera AI containers:"
        echo "$running_containers"
    else
        info "No ITS Camera AI containers currently running"
    fi
    
    # Show Docker volumes
    local its_volumes
    its_volumes=$(docker volume ls --filter "name=its-camera-ai" --format "{{.Name}}" 2>/dev/null || echo "")
    if [[ -n "$its_volumes" ]]; then
        info "ITS Camera AI volumes found:"
        echo "$its_volumes" | sed 's/^/  /'
    else
        info "No ITS Camera AI volumes found"
    fi
    
    # Show Docker networks
    local its_networks
    its_networks=$(docker network ls --filter "name=its" --format "{{.Name}}" 2>/dev/null || echo "")
    if [[ -n "$its_networks" ]]; then
        info "ITS Camera AI networks found:"
        echo "$its_networks" | sed 's/^/  /'
    else
        info "No ITS Camera AI networks found"
    fi
    
    # Show disk usage
    local disk_usage
    disk_usage=$(du -sh "${SCRIPT_DIR}" 2>/dev/null || echo "unknown")
    info "Current project disk usage: $disk_usage"
}

# Stop all services
stop_all_services() {
    log "Stopping all ITS Camera AI services..."
    
    local compose_files=(
        "-f docker-compose.yml"
        "-f docker-compose.yml -f docker-compose.dev.yml"
        "-f docker-compose.yml -f docker-compose.prod.yml"
        "-f docker-compose.yml -f docker-compose.gpu.yml"
        "-f docker-compose.yml -f docker-compose.edge.yml"
    )
    
    for compose_file in "${compose_files[@]}"; do
        info "Stopping services with compose file: $compose_file"
        docker compose $compose_file down --remove-orphans 2>/dev/null || warn "Failed to stop services with $compose_file"
    done
    
    # Force stop any remaining ITS Camera AI containers
    local remaining_containers
    remaining_containers=$(docker ps -q --filter "name=its-camera-ai" 2>/dev/null || echo "")
    if [[ -n "$remaining_containers" ]]; then
        info "Force stopping remaining containers..."
        echo "$remaining_containers" | xargs docker stop 2>/dev/null || warn "Failed to force stop some containers"
        echo "$remaining_containers" | xargs docker rm 2>/dev/null || warn "Failed to remove some containers"
    fi
    
    log "Services stopped ✓"
}

# Remove containers
remove_containers() {
    if [[ "${REMOVE_CONTAINERS:-true}" == "true" ]]; then
        log "Removing ITS Camera AI containers..."
        
        # Remove all ITS Camera AI containers (running and stopped)
        local all_containers
        all_containers=$(docker ps -aq --filter "name=its-camera-ai" 2>/dev/null || echo "")
        if [[ -n "$all_containers" ]]; then
            info "Removing containers..."
            echo "$all_containers" | xargs docker rm -f 2>/dev/null || warn "Failed to remove some containers"
        else
            info "No containers to remove"
        fi
        
        log "Containers removed ✓"
    else
        info "Skipping container removal (--keep-containers specified)"
    fi
}

# Remove volumes
remove_volumes() {
    if [[ "${REMOVE_VOLUMES:-true}" == "true" ]]; then
        log "Removing ITS Camera AI volumes..."
        
        # Remove named volumes
        local volumes
        volumes=$(docker volume ls --filter "name=its-camera-ai" --format "{{.Name}}" 2>/dev/null || echo "")
        if [[ -n "$volumes" ]]; then
            info "Removing volumes..."
            echo "$volumes" | xargs docker volume rm -f 2>/dev/null || warn "Failed to remove some volumes"
        else
            info "No volumes to remove"
        fi
        
        # Remove anonymous volumes
        info "Removing anonymous volumes..."
        docker volume prune -f >/dev/null 2>&1 || warn "Failed to prune anonymous volumes"
        
        log "Volumes removed ✓"
    else
        info "Skipping volume removal (--keep-volumes specified)"
    fi
}

# Remove networks
remove_networks() {
    if [[ "${REMOVE_NETWORKS:-true}" == "true" ]]; then
        log "Removing ITS Camera AI networks..."
        
        local networks
        networks=$(docker network ls --filter "name=its" --format "{{.Name}}" 2>/dev/null | grep -v "bridge\|host\|none" || echo "")
        if [[ -n "$networks" ]]; then
            info "Removing networks..."
            echo "$networks" | xargs docker network rm 2>/dev/null || warn "Failed to remove some networks"
        else
            info "No networks to remove"
        fi
        
        log "Networks removed ✓"
    else
        info "Skipping network removal (--keep-networks specified)"
    fi
}

# Remove images
remove_images() {
    if [[ "${REMOVE_IMAGES:-false}" == "true" ]]; then
        log "Removing ITS Camera AI images..."
        
        # Remove built images
        local images
        images=$(docker images --filter "reference=*its-camera-ai*" --format "{{.ID}}" 2>/dev/null || echo "")
        if [[ -n "$images" ]]; then
            info "Removing built images..."
            echo "$images" | xargs docker rmi -f 2>/dev/null || warn "Failed to remove some images"
        else
            info "No built images to remove"
        fi
        
        # Remove dangling images
        info "Removing dangling images..."
        docker image prune -f >/dev/null 2>&1 || warn "Failed to prune dangling images"
        
        log "Images removed ✓"
    else
        info "Skipping image removal (use --remove-images to remove)"
    fi
}

# Clean data directories
clean_data_directories() {
    if [[ "${CLEAN_DATA:-false}" == "true" ]]; then
        log "Cleaning data directories..."
        
        # Create backup if requested
        if [[ "${BACKUP_BEFORE_CLEAN:-false}" == "true" ]]; then
            local backup_dir="${SCRIPT_DIR}/backups/cleanup_$(date +'%Y%m%d_%H%M%S')"
            info "Creating backup in: $backup_dir"
            mkdir -p "$backup_dir"
            
            for dir in data logs models temp cache; do
                if [[ -d "${SCRIPT_DIR}/$dir" ]]; then
                    cp -r "${SCRIPT_DIR}/$dir" "$backup_dir/" 2>/dev/null || warn "Failed to backup $dir"
                fi
            done
        fi
        
        # Clean directories
        local dirs_to_clean=(
            "data/dev"
            "data/prod"
            "data/gpu"
            "data/edge"
            "data/postgres"
            "data/redis"
            "data/minio"
            "logs"
            "temp"
            "cache"
        )
        
        for dir in "${dirs_to_clean[@]}"; do
            if [[ -d "${SCRIPT_DIR}/$dir" ]]; then
                info "Cleaning $dir..."
                rm -rf "${SCRIPT_DIR:?}/$dir"/* 2>/dev/null || warn "Failed to clean $dir"
            fi
        done
        
        log "Data directories cleaned ✓"
    else
        info "Skipping data directory cleanup (use --clean-data to clean)"
    fi
}

# System cleanup
system_cleanup() {
    if [[ "${SYSTEM_CLEANUP:-false}" == "true" ]]; then
        log "Performing system-wide Docker cleanup..."
        
        info "Removing unused containers..."
        docker container prune -f >/dev/null 2>&1 || warn "Failed to prune containers"
        
        info "Removing unused networks..."
        docker network prune -f >/dev/null 2>&1 || warn "Failed to prune networks"
        
        info "Removing unused volumes..."
        docker volume prune -f >/dev/null 2>&1 || warn "Failed to prune volumes"
        
        info "Removing unused images..."
        docker image prune -a -f >/dev/null 2>&1 || warn "Failed to prune images"
        
        info "Removing build cache..."
        docker builder prune -f >/dev/null 2>&1 || warn "Failed to prune build cache"
        
        log "System cleanup completed ✓"
    else
        info "Skipping system cleanup (use --system-cleanup for full cleanup)"
    fi
}

# Reset to initial state
reset_to_initial_state() {
    if [[ "${RESET:-false}" == "true" ]]; then
        log "Resetting to initial project state..."
        
        # Reset environment files
        if [[ -f "${SCRIPT_DIR}/.env.example" ]]; then
            info "Resetting .env file..."
            cp "${SCRIPT_DIR}/.env.example" "${SCRIPT_DIR}/.env" 2>/dev/null || warn "Failed to reset .env file"
        fi
        
        # Remove generated environment files
        for env_file in .env.dev .env.prod .env.gpu .env.edge; do
            if [[ -f "${SCRIPT_DIR}/$env_file" ]]; then
                info "Removing generated environment file: $env_file"
                rm -f "${SCRIPT_DIR}/$env_file"
            fi
        done
        
        # Reset log files
        if [[ -d "${SCRIPT_DIR}/logs" ]]; then
            info "Clearing log files..."
            find "${SCRIPT_DIR}/logs" -name "*.log" -delete 2>/dev/null || warn "Failed to clear some log files"
        fi
        
        log "Reset to initial state completed ✓"
    else
        info "Skipping reset (use --reset to reset to initial state)"
    fi
}

# Show cleanup summary
show_cleanup_summary() {
    log "Cleanup completed! 🧹"
    
    echo
    info "=== Cleanup Summary ==="
    
    # Show remaining containers
    local remaining_containers
    remaining_containers=$(docker ps -aq --filter "name=its-camera-ai" 2>/dev/null | wc -l || echo "0")
    echo "📦 Remaining containers: $remaining_containers"
    
    # Show remaining volumes
    local remaining_volumes
    remaining_volumes=$(docker volume ls --filter "name=its-camera-ai" --format "{{.Name}}" 2>/dev/null | wc -l || echo "0")
    echo "💾 Remaining volumes: $remaining_volumes"
    
    # Show remaining networks
    local remaining_networks
    remaining_networks=$(docker network ls --filter "name=its" --format "{{.Name}}" 2>/dev/null | grep -v "bridge\|host\|none" | wc -l || echo "0")
    echo "🌐 Remaining networks: $remaining_networks"
    
    # Show remaining images
    local remaining_images
    remaining_images=$(docker images --filter "reference=*its-camera-ai*" --format "{{.ID}}" 2>/dev/null | wc -l || echo "0")
    echo "🖼️  Remaining images: $remaining_images"
    
    # Show disk space freed
    local current_usage
    current_usage=$(du -sh "${SCRIPT_DIR}" 2>/dev/null || echo "unknown")
    echo "💽 Current project size: $current_usage"
    
    # Show Docker system disk usage
    info "Docker system disk usage:"
    docker system df 2>/dev/null || warn "Failed to get Docker disk usage"
    
    echo
    info "=== Next Steps ==="
    echo "🚀 To start fresh: ./start-dev.sh"
    echo "🏭 For production: ./start-prod.sh"
    echo "🔥 For GPU setup: ./start-gpu.sh"
    echo "📊 Check status: docker compose ps"
    echo
}

# Confirmation prompt
confirm_cleanup() {
    if [[ "${FORCE:-false}" != "true" ]]; then
        echo
        warn "This will clean up ITS Camera AI environment:"
        [[ "${REMOVE_CONTAINERS:-true}" == "true" ]] && echo "  ❌ Remove all containers"
        [[ "${REMOVE_VOLUMES:-true}" == "true" ]] && echo "  ❌ Remove all volumes (DATA LOSS)"
        [[ "${REMOVE_NETWORKS:-true}" == "true" ]] && echo "  ❌ Remove all networks"
        [[ "${REMOVE_IMAGES:-false}" == "true" ]] && echo "  ❌ Remove all images"
        [[ "${CLEAN_DATA:-false}" == "true" ]] && echo "  ❌ Clean data directories (DATA LOSS)"
        [[ "${SYSTEM_CLEANUP:-false}" == "true" ]] && echo "  ❌ Full Docker system cleanup"
        [[ "${RESET:-false}" == "true" ]] && echo "  ❌ Reset to initial project state"
        echo
        
        read -p "Are you sure you want to continue? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            info "Cleanup cancelled"
            exit 0
        fi
    fi
}

# Main execution
main() {
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --force)
                export FORCE=true
                shift
                ;;
            --keep-containers)
                export REMOVE_CONTAINERS=false
                shift
                ;;
            --keep-volumes)
                export REMOVE_VOLUMES=false
                shift
                ;;
            --keep-networks)
                export REMOVE_NETWORKS=false
                shift
                ;;
            --remove-images)
                export REMOVE_IMAGES=true
                shift
                ;;
            --clean-data)
                export CLEAN_DATA=true
                shift
                ;;
            --backup-before-clean)
                export BACKUP_BEFORE_CLEAN=true
                export CLEAN_DATA=true
                shift
                ;;
            --system-cleanup)
                export SYSTEM_CLEANUP=true
                shift
                ;;
            --reset)
                export RESET=true
                shift
                ;;
            --all)
                export REMOVE_IMAGES=true
                export CLEAN_DATA=true
                export SYSTEM_CLEANUP=true
                export RESET=true
                shift
                ;;
            --help)
                echo "Usage: $0 [options]"
                echo
                echo "Options:"
                echo "  --force                 Skip confirmation prompt"
                echo "  --keep-containers       Keep containers (don't remove)"
                echo "  --keep-volumes          Keep volumes (don't remove)"
                echo "  --keep-networks         Keep networks (don't remove)"
                echo "  --remove-images         Remove Docker images"
                echo "  --clean-data            Clean data directories (DATA LOSS)"
                echo "  --backup-before-clean   Backup data before cleaning"
                echo "  --system-cleanup        Full Docker system cleanup"
                echo "  --reset                 Reset to initial project state"
                echo "  --all                   Complete cleanup (equivalent to all options)"
                echo "  --help                  Show this help message"
                echo
                echo "Examples:"
                echo "  $0                      Basic cleanup (containers, volumes, networks)"
                echo "  $0 --all --force        Complete cleanup without confirmation"
                echo "  $0 --keep-volumes       Cleanup but preserve data volumes"
                echo "  $0 --clean-data --backup-before-clean  Clean data with backup"
                exit 0
                ;;
            *)
                warn "Unknown option: $1"
                shift
                ;;
        esac
    done
    
    # Show current status
    show_current_status
    
    # Confirm cleanup
    confirm_cleanup
    
    # Execute cleanup steps
    stop_all_services
    remove_containers
    remove_volumes
    remove_networks
    remove_images
    clean_data_directories
    reset_to_initial_state
    system_cleanup
    
    # Show summary
    show_cleanup_summary
    
    log "Environment cleanup completed successfully!"
}

# Run main function
main "$@"