#!/bin/bash

# ITS Camera AI - Worker Management Script
# Starts and manages Celery workers for background processing

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WORKER_LOG_DIR="/var/log/its-camera-ai"
WORKER_PID_DIR="/var/run/its-camera-ai"

# Default configuration
WORKER_CONCURRENCY=${WORKER_CONCURRENCY:-2}
WORKER_LOGLEVEL=${WORKER_LOGLEVEL:-INFO}
REDIS_URL=${REDIS_URL:-redis://localhost:6379/0}
FLOWER_PORT=${FLOWER_PORT:-5555}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Create required directories
create_directories() {
    log "Creating required directories..."
    
    sudo mkdir -p "$WORKER_LOG_DIR"
    sudo mkdir -p "$WORKER_PID_DIR"
    
    # Set permissions
    sudo chown -R $(whoami):$(whoami) "$WORKER_LOG_DIR" "$WORKER_PID_DIR" 2>/dev/null || true
    
    success "Directories created successfully"
}

# Check dependencies
check_dependencies() {
    log "Checking dependencies..."
    
    # Check if Redis is running
    if ! redis-cli ping >/dev/null 2>&1; then
        error "Redis is not running. Please start Redis server first."
        exit 1
    fi
    
    # Check if PostgreSQL is running
    if ! pg_isready >/dev/null 2>&1; then
        warning "PostgreSQL may not be running. Some features might not work."
    fi
    
    # Check Python environment
    if ! python -c "import celery" >/dev/null 2>&1; then
        error "Celery is not installed. Please run: uv sync --group dev"
        exit 1
    fi
    
    success "Dependencies check completed"
}

# Start main Celery worker
start_worker() {
    local queue=$1
    local concurrency=$2
    local log_file="$WORKER_LOG_DIR/worker-${queue}.log"
    local pid_file="$WORKER_PID_DIR/worker-${queue}.pid"
    
    log "Starting Celery worker for queue: $queue"
    
    # Kill existing worker if running
    if [ -f "$pid_file" ] && kill -0 $(cat "$pid_file") 2>/dev/null; then
        warning "Worker for queue $queue is already running (PID: $(cat "$pid_file"))"
        return 0
    fi
    
    # Start worker
    celery -A its_camera_ai.workers worker \
        --loglevel="$WORKER_LOGLEVEL" \
        --concurrency="$concurrency" \
        --queues="$queue" \
        --logfile="$log_file" \
        --pidfile="$pid_file" \
        --detach \
        --time-limit=7200 \
        --soft-time-limit=3600 \
        --max-tasks-per-child=1000 \
        --prefetch-multiplier=1
    
    if [ $? -eq 0 ]; then
        success "Worker for queue $queue started successfully"
    else
        error "Failed to start worker for queue $queue"
        return 1
    fi
}

# Start Celery Beat scheduler
start_beat() {
    log "Starting Celery Beat scheduler..."
    
    local log_file="$WORKER_LOG_DIR/beat.log"
    local pid_file="$WORKER_PID_DIR/beat.pid"
    
    # Kill existing beat if running
    if [ -f "$pid_file" ] && kill -0 $(cat "$pid_file") 2>/dev/null; then
        warning "Celery Beat is already running (PID: $(cat "$pid_file"))"
        return 0
    fi
    
    # Start beat scheduler
    celery -A its_camera_ai.workers beat \
        --loglevel="$WORKER_LOGLEVEL" \
        --logfile="$log_file" \
        --pidfile="$pid_file" \
        --detach \
        --schedule="$WORKER_PID_DIR/celerybeat-schedule"
    
    if [ $? -eq 0 ]; then
        success "Celery Beat started successfully"
    else
        error "Failed to start Celery Beat"
        return 1
    fi
}

# Start Flower monitoring
start_flower() {
    log "Starting Flower monitoring interface..."
    
    local log_file="$WORKER_LOG_DIR/flower.log"
    local pid_file="$WORKER_PID_DIR/flower.pid"
    
    # Kill existing flower if running
    if [ -f "$pid_file" ] && kill -0 $(cat "$pid_file") 2>/dev/null; then
        warning "Flower is already running (PID: $(cat "$pid_file"))"
        return 0
    fi
    
    # Start Flower
    celery -A its_camera_ai.workers flower \
        --port="$FLOWER_PORT" \
        --broker="$REDIS_URL" \
        --logging=info \
        --log-file-prefix="$log_file" \
        --pid="$pid_file" \
        --detach
    
    if [ $? -eq 0 ]; then
        success "Flower started successfully on port $FLOWER_PORT"
        log "Access Flower at: http://localhost:$FLOWER_PORT"
    else
        error "Failed to start Flower"
        return 1
    fi
}

# Stop workers
stop_workers() {
    log "Stopping all workers..."
    
    # Stop workers
    for pid_file in "$WORKER_PID_DIR"/worker-*.pid; do
        if [ -f "$pid_file" ]; then
            local pid=$(cat "$pid_file")
            local queue=$(basename "$pid_file" .pid | sed 's/worker-//')
            
            if kill -0 "$pid" 2>/dev/null; then
                log "Stopping worker for queue: $queue (PID: $pid)"
                kill -TERM "$pid"
                
                # Wait for graceful shutdown
                local count=0
                while kill -0 "$pid" 2>/dev/null && [ $count -lt 30 ]; do
                    sleep 1
                    count=$((count + 1))
                done
                
                # Force kill if still running
                if kill -0 "$pid" 2>/dev/null; then
                    warning "Force killing worker for queue: $queue"
                    kill -KILL "$pid"
                fi
                
                rm -f "$pid_file"
                success "Worker for queue $queue stopped"
            else
                rm -f "$pid_file"
            fi
        fi
    done
    
    # Stop beat
    local beat_pid_file="$WORKER_PID_DIR/beat.pid"
    if [ -f "$beat_pid_file" ]; then
        local pid=$(cat "$beat_pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            log "Stopping Celery Beat (PID: $pid)"
            kill -TERM "$pid"
            sleep 2
            rm -f "$beat_pid_file"
            success "Celery Beat stopped"
        else
            rm -f "$beat_pid_file"
        fi
    fi
    
    # Stop flower
    local flower_pid_file="$WORKER_PID_DIR/flower.pid"
    if [ -f "$flower_pid_file" ]; then
        local pid=$(cat "$flower_pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            log "Stopping Flower (PID: $pid)"
            kill -TERM "$pid"
            sleep 2
            rm -f "$flower_pid_file"
            success "Flower stopped"
        else
            rm -f "$flower_pid_file"
        fi
    fi
}

# Show worker status
show_status() {
    log "Worker Status:"
    echo
    
    # Check workers
    for pid_file in "$WORKER_PID_DIR"/worker-*.pid; do
        if [ -f "$pid_file" ]; then
            local pid=$(cat "$pid_file")
            local queue=$(basename "$pid_file" .pid | sed 's/worker-//')
            
            if kill -0 "$pid" 2>/dev/null; then
                echo -e "  ${GREEN}✓${NC} Worker ($queue): Running (PID: $pid)"
            else
                echo -e "  ${RED}✗${NC} Worker ($queue): Stopped"
                rm -f "$pid_file"
            fi
        fi
    done
    
    # Check beat
    local beat_pid_file="$WORKER_PID_DIR/beat.pid"
    if [ -f "$beat_pid_file" ] && kill -0 $(cat "$beat_pid_file") 2>/dev/null; then
        echo -e "  ${GREEN}✓${NC} Celery Beat: Running (PID: $(cat "$beat_pid_file"))"
    else
        echo -e "  ${RED}✗${NC} Celery Beat: Stopped"
    fi
    
    # Check flower
    local flower_pid_file="$WORKER_PID_DIR/flower.pid"
    if [ -f "$flower_pid_file" ] && kill -0 $(cat "$flower_pid_file") 2>/dev/null; then
        echo -e "  ${GREEN}✓${NC} Flower: Running (PID: $(cat "$flower_pid_file"))"
        echo -e "    URL: http://localhost:$FLOWER_PORT"
    else
        echo -e "  ${RED}✗${NC} Flower: Stopped"
    fi
    
    echo
    
    # Show Redis status
    if redis-cli ping >/dev/null 2>&1; then
        echo -e "  ${GREEN}✓${NC} Redis: Running"
    else
        echo -e "  ${RED}✗${NC} Redis: Not accessible"
    fi
    
    # Show queue stats
    echo
    log "Queue Statistics:"
    if redis-cli ping >/dev/null 2>&1; then
        local queues=("realtime" "analytics" "maintenance" "default")
        for queue in "${queues[@]}"; do
            local count=$(redis-cli llen "celery" 2>/dev/null || echo "0")
            echo "  $queue: $count tasks"
        done
    else
        echo "  Unable to connect to Redis"
    fi
}

# Show logs
show_logs() {
    local service=${1:-all}
    
    case $service in
        workers)
            log "Worker logs:"
            for log_file in "$WORKER_LOG_DIR"/worker-*.log; do
                if [ -f "$log_file" ]; then
                    echo
                    echo "=== $(basename "$log_file") ==="
                    tail -n 20 "$log_file"
                fi
            done
            ;;
        beat)
            log "Celery Beat logs:"
            if [ -f "$WORKER_LOG_DIR/beat.log" ]; then
                tail -n 20 "$WORKER_LOG_DIR/beat.log"
            else
                echo "No beat log file found"
            fi
            ;;
        flower)
            log "Flower logs:"
            if [ -f "$WORKER_LOG_DIR/flower.log" ]; then
                tail -n 20 "$WORKER_LOG_DIR/flower.log"
            else
                echo "No flower log file found"
            fi
            ;;
        all|*)
            show_logs workers
            echo
            show_logs beat
            echo
            show_logs flower
            ;;
    esac
}

# Main function
main() {
    case ${1:-start} in
        start)
            log "Starting ITS Camera AI workers..."
            create_directories
            check_dependencies
            
            # Start workers for different queues
            start_worker "realtime" 4    # High priority, more workers
            start_worker "analytics" 2   # Medium priority
            start_worker "maintenance" 1 # Low priority
            start_worker "default" 2     # Default queue
            
            # Start beat scheduler
            start_beat
            
            # Start flower monitoring
            start_flower
            
            echo
            success "All workers started successfully!"
            echo
            show_status
            ;;
            
        stop)
            stop_workers
            success "All workers stopped"
            ;;
            
        restart)
            log "Restarting workers..."
            stop_workers
            sleep 3
            main start
            ;;
            
        status)
            show_status
            ;;
            
        logs)
            show_logs ${2:-all}
            ;;
            
        purge)
            log "Purging task queues..."
            celery -A its_camera_ai.workers purge -f
            success "Task queues purged"
            ;;
            
        inspect)
            log "Inspecting workers..."
            celery -A its_camera_ai.workers inspect active
            ;;
            
        monitor)
            log "Starting worker monitoring (Ctrl+C to exit)..."
            celery -A its_camera_ai.workers events
            ;;
            
        flower)
            log "Opening Flower monitoring interface..."
            if command -v xdg-open >/dev/null; then
                xdg-open "http://localhost:$FLOWER_PORT"
            elif command -v open >/dev/null; then
                open "http://localhost:$FLOWER_PORT"
            else
                log "Open http://localhost:$FLOWER_PORT in your browser"
            fi
            ;;
            
        help|*)
            echo "ITS Camera AI Worker Management Script"
            echo
            echo "Usage: $0 [COMMAND]"
            echo
            echo "Commands:"
            echo "  start     Start all workers, beat, and flower"
            echo "  stop      Stop all workers"
            echo "  restart   Restart all workers"
            echo "  status    Show worker status"
            echo "  logs      Show worker logs [workers|beat|flower|all]"
            echo "  purge     Purge all task queues"
            echo "  inspect   Inspect active tasks"
            echo "  monitor   Start real-time monitoring"
            echo "  flower    Open Flower monitoring interface"
            echo "  help      Show this help message"
            echo
            echo "Environment Variables:"
            echo "  WORKER_CONCURRENCY  Number of worker processes (default: 2)"
            echo "  WORKER_LOGLEVEL     Log level (default: INFO)"
            echo "  REDIS_URL          Redis connection URL"
            echo "  FLOWER_PORT        Flower web interface port (default: 5555)"
            ;;
    esac
}

# Change to project directory
cd "$PROJECT_ROOT"

# Run main function with all arguments
main "$@"