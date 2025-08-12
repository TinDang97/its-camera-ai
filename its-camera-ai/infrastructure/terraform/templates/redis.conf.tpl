# Redis Configuration Template for ITS Camera AI Streaming
# Optimized for high-throughput queue processing with Redis Streams

# Network and Security
bind 0.0.0.0
port 6379
protected-mode yes
requirepass $${REDIS_PASSWORD}

# Memory Management
maxmemory ${redis_config.maxmemory}
maxmemory-policy ${redis_config.maxmemory_policy}

# Persistence Configuration
%{ for save_rule in split(" ", redis_config.save) ~}
save ${save_rule}
%{ endfor ~}
rdbcompression yes
rdbchecksum yes

# AOF Configuration
appendonly ${redis_config.appendonly}
appendfilename "appendonly.aof"
appendfsync ${redis_config.appendfsync}
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb

# Network Settings
tcp-backlog ${redis_config.tcp_backlog}
timeout ${redis_config.timeout}
tcp-keepalive ${redis_config.tcp_keepalive}

# Client Configuration
maxclients 10000

# Redis Streams Configuration
stream-node-max-bytes ${redis_config.stream_node_max_bytes}
stream-node-max-entries ${redis_config.stream_node_max_entries}

# Performance Optimizations
io-threads ${redis_config.io_threads}
io-threads-do-reads ${redis_config.io_threads_do_reads}

# Lazy Deletion
lazyfree-lazy-eviction yes
lazyfree-lazy-expire yes
lazyfree-lazy-server-del yes
replica-lazy-flush yes

# Data Structure Optimizations
hash-max-ziplist-entries 1024
hash-max-ziplist-value 16384
list-max-ziplist-size -2
list-compress-depth 0
set-max-intset-entries 512
zset-max-ziplist-entries 128
zset-max-ziplist-value 64

# Logging
loglevel notice
logfile ""

# Advanced Settings
tcp-nodelay yes
stop-writes-on-bgsave-error no
notify-keyspace-events Ex

# HyperLogLog
hll-sparse-max-bytes 3000