# ITS Camera AI - UX Guidelines & Implementation

## Overview

This document provides comprehensive user experience guidelines, accessibility standards, mobile design patterns, and implementation best practices for the ITS Camera AI web application. These guidelines ensure consistent, accessible, and user-friendly experiences across all personas and devices.

## Table of Contents

1. [User Experience Principles](#user-experience-principles)
2. [Accessibility Guidelines (WCAG 2.1 AA)](#accessibility-guidelines-wcag-21-aa)
3. [Mobile-First Design Patterns](#mobile-first-design-patterns)
4. [Performance Optimization](#performance-optimization)
5. [Error Handling & Feedback](#error-handling--feedback)
6. [Implementation Best Practices](#implementation-best-practices)
7. [Testing & Quality Assurance](#testing--quality-assurance)

---

## User Experience Principles

### Core UX Principles for Traffic Monitoring

#### 1. **Immediate Clarity**
- Critical information is visible within 3 seconds of page load
- Status indicators use universal color conventions (green=good, red=problem)
- Performance metrics are always visible and easily interpreted

#### 2. **Contextual Efficiency** 
- Related actions are grouped logically
- Most common tasks require minimal clicks/taps
- Information hierarchy matches user mental models

#### 3. **Proactive Communication**
- System proactively surfaces issues before they become critical
- Predictive insights help users make informed decisions
- Clear progress indicators for all system operations

#### 4. **Adaptive Complexity**
- Interface adapts to user expertise level
- Progressive disclosure reveals advanced features on demand
- Customizable dashboards for different personas

### Persona-Specific UX Considerations

#### Traffic Operations Manager
**Primary Goals**: Real-time monitoring, incident response, system status
- **Dashboard Priority**: Live camera feeds, active alerts, system health
- **Key Interactions**: Alert acknowledgment, camera control, incident management
- **Information Density**: High - can process multiple data streams simultaneously
- **Response Time Expectations**: Immediate (<2 seconds for critical actions)

#### Traffic Engineer
**Primary Goals**: Analysis, optimization, configuration, reporting
- **Dashboard Priority**: Analytics, historical trends, performance metrics
- **Key Interactions**: Data filtering, report generation, threshold configuration
- **Information Density**: Variable - needs both summary and detailed views
- **Response Time Expectations**: Moderate (5-10 seconds for complex queries acceptable)

#### City Planner/Director
**Primary Goals**: Executive overview, ROI analysis, strategic planning
- **Dashboard Priority**: High-level KPIs, trends, compliance status
- **Key Interactions**: Report viewing, export, time range selection
- **Information Density**: Low to moderate - focused on key insights
- **Response Time Expectations**: Flexible (willing to wait for comprehensive reports)

---

## Accessibility Guidelines (WCAG 2.1 AA)

### Color & Contrast Requirements

#### Minimum Contrast Ratios
- **Normal Text**: 4.5:1 contrast ratio minimum
- **Large Text** (18pt+ or 14pt+ bold): 3:1 contrast ratio minimum
- **Non-text Elements**: 3:1 contrast ratio for UI components and graphics

#### Color Usage
```css
/* High contrast color combinations for critical information */
.high-contrast-critical {
  background-color: #dc2626; /* Red 600 */
  color: #ffffff; /* White */
  /* Contrast ratio: 5.9:1 - Exceeds AA standard */
}

.high-contrast-warning {
  background-color: #d97706; /* Amber 600 */
  color: #ffffff; /* White */
  /* Contrast ratio: 4.7:1 - Meets AA standard */
}

.high-contrast-success {
  background-color: #059669; /* Green 600 */
  color: #ffffff; /* White */
  /* Contrast ratio: 4.5:1 - Meets AA standard */
}

/* Alternative color indicators (never rely on color alone) */
.status-indicator::before {
  content: '';
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  margin-right: 8px;
}

.status-critical::before { 
  background: #dc2626;
  animation: pulse 2s infinite;
}

.status-warning::before { 
  background: #d97706;
  border: 2px solid #ffffff;
}

.status-success::before { 
  background: #059669;
}
```

### Keyboard Navigation

#### Navigation Patterns
```tsx
// Example: Comprehensive keyboard navigation for AlertPanel
const AlertPanel = ({ alerts, onAlertAction }) => {
  const [focusedAlertIndex, setFocusedAlertIndex] = useState(0);
  
  const handleKeyDown = useCallback((event: KeyboardEvent) => {
    switch (event.key) {
      case 'ArrowDown':
        event.preventDefault();
        setFocusedAlertIndex(prev => 
          Math.min(prev + 1, alerts.length - 1)
        );
        break;
      case 'ArrowUp':
        event.preventDefault();
        setFocusedAlertIndex(prev => Math.max(prev - 1, 0));
        break;
      case 'Enter':
      case ' ':
        event.preventDefault();
        onAlertAction(alerts[focusedAlertIndex].id, 'acknowledge');
        break;
      case 'Escape':
        setFocusedAlertIndex(0);
        break;
    }
  }, [alerts, focusedAlertIndex, onAlertAction]);

  return (
    <div 
      className="alert-panel"
      role="log"
      aria-live="polite"
      aria-label="Traffic system alerts"
      onKeyDown={handleKeyDown}
      tabIndex={0}
    >
      {alerts.map((alert, index) => (
        <AlertItem
          key={alert.id}
          alert={alert}
          focused={index === focusedAlertIndex}
          onAction={onAlertAction}
        />
      ))}
    </div>
  );
};
```

#### Focus Management
- **Skip Links**: Provide skip-to-main-content links
- **Focus Trapping**: Modal dialogs trap focus within the modal
- **Focus Restoration**: Return focus to trigger element after modal closes
- **Visible Focus Indicators**: 2px outline with sufficient contrast

```css
/* Focus indicators with high contrast */
.focus-visible {
  outline: 2px solid #2563eb;
  outline-offset: 2px;
  border-radius: 4px;
}

/* Skip links for keyboard users */
.skip-link {
  position: absolute;
  top: -40px;
  left: 6px;
  background: #2563eb;
  color: white;
  padding: 8px;
  border-radius: 4px;
  text-decoration: none;
  z-index: 1000;
  transition: top 0.3s;
}

.skip-link:focus {
  top: 6px;
}
```

### Screen Reader Support

#### ARIA Implementation
```tsx
// Example: Accessible performance indicator
const PerformanceIndicator = ({ 
  latency, 
  accuracy, 
  confidence 
}) => {
  const getLatencyStatus = (ms: number) => {
    if (ms < 50) return 'excellent';
    if (ms < 100) return 'good';
    if (ms < 200) return 'warning';
    return 'poor';
  };

  const latencyStatus = getLatencyStatus(latency);

  return (
    <div 
      className="performance-indicator"
      role="region"
      aria-label="System performance metrics"
    >
      <div 
        className={`metric-item metric-${latencyStatus}`}
        role="img"
        aria-label={`Inference latency: ${latency} milliseconds, status: ${latencyStatus}`}
      >
        <span className="metric-label" id="latency-label">
          Latency
        </span>
        <span 
          className="metric-value"
          aria-describedby="latency-label"
        >
          {latency}ms
        </span>
        <span className="sr-only">
          {latencyStatus === 'excellent' && 'Performance is excellent'}
          {latencyStatus === 'good' && 'Performance is good'}
          {latencyStatus === 'warning' && 'Performance needs attention'}
          {latencyStatus === 'poor' && 'Performance is poor, action required'}
        </span>
      </div>
      
      {/* Similar structure for accuracy and confidence */}
    </div>
  );
};
```

#### Live Regions for Dynamic Content
```tsx
// Real-time updates with appropriate live regions
const TrafficDashboard = () => {
  return (
    <div>
      {/* Polite updates for general metrics */}
      <div 
        id="metrics-live-region"
        aria-live="polite"
        aria-atomic="false"
        className="sr-only"
      />
      
      {/* Assertive updates for critical alerts */}
      <div 
        id="alerts-live-region"
        aria-live="assertive"
        aria-atomic="true"
        className="sr-only"
      />
      
      {/* Status updates */}
      <div
        id="status-live-region"
        aria-live="polite"
        role="status"
        className="sr-only"
      />
    </div>
  );
};
```

### Alternative Content Formats

#### Data Tables with Accessibility
```tsx
const AccessibleDataTable = ({ data, columns }) => {
  return (
    <table 
      className="data-table"
      role="table"
      aria-label="Camera performance data"
    >
      <caption className="sr-only">
        Performance data for {data.length} cameras, 
        sortable by column headers
      </caption>
      <thead>
        <tr role="row">
          {columns.map((column) => (
            <th 
              key={column.id}
              role="columnheader"
              aria-sort={
                column.sorted ? 
                (column.sortDirection === 'asc' ? 'ascending' : 'descending') : 
                'none'
              }
              tabIndex={0}
              onClick={() => handleSort(column.id)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  handleSort(column.id);
                }
              }}
            >
              {column.title}
              {column.sorted && (
                <span aria-hidden="true">
                  {column.sortDirection === 'asc' ? ' ↑' : ' ↓'}
                </span>
              )}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {data.map((row, index) => (
          <tr key={row.id} role="row">
            {columns.map((column) => (
              <td 
                key={column.id}
                role="gridcell"
                aria-describedby={`${column.id}-description`}
              >
                {formatCellValue(row[column.field], column.type)}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
};
```

---

## Mobile-First Design Patterns

### Responsive Breakpoints

```css
/* Mobile-first breakpoint system */
:root {
  --breakpoint-xs: 475px;   /* Large phones */
  --breakpoint-sm: 640px;   /* Small tablets */
  --breakpoint-md: 768px;   /* Tablets */
  --breakpoint-lg: 1024px;  /* Small laptops */
  --breakpoint-xl: 1280px;  /* Desktops */
  --breakpoint-2xl: 1536px; /* Large screens */
}

/* Mobile-first media queries */
@media (min-width: 640px) { /* sm and up */ }
@media (min-width: 768px) { /* md and up */ }
@media (min-width: 1024px) { /* lg and up */ }
@media (min-width: 1280px) { /* xl and up */ }
```

### Touch-Friendly Interface Design

#### Touch Target Specifications
```css
/* Minimum touch target sizes */
.touch-target {
  min-height: 44px;
  min-width: 44px;
  padding: 12px;
  margin: 4px;
}

/* Button spacing for mobile */
.button-group {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}

.button-group .button {
  min-height: 48px;
  padding: 12px 16px;
  border-radius: 8px;
}

/* Mobile-specific layouts */
@media (max-width: 768px) {
  .desktop-grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: 16px;
  }
  
  .card-grid {
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  }
}
```

#### Gesture Support
```tsx
// Example: Swipe gestures for camera navigation
const CameraCarousel = ({ cameras }) => {
  const [currentIndex, setCurrentIndex] = useState(0);
  
  const swipeHandlers = useSwipeable({
    onSwipedLeft: () => setCurrentIndex(prev => 
      Math.min(prev + 1, cameras.length - 1)
    ),
    onSwipedRight: () => setCurrentIndex(prev => 
      Math.max(prev - 1, 0)
    ),
    preventDefaultTouchmoveEvent: true,
    trackMouse: true
  });

  return (
    <div 
      {...swipeHandlers}
      className="camera-carousel"
      role="tablist"
      aria-label="Camera feeds"
    >
      <div 
        className="carousel-track"
        style={{ 
          transform: `translateX(-${currentIndex * 100}%)` 
        }}
      >
        {cameras.map((camera, index) => (
          <CameraCard
            key={camera.id}
            camera={camera}
            active={index === currentIndex}
            role="tabpanel"
            aria-hidden={index !== currentIndex}
          />
        ))}
      </div>
      
      {/* Accessible navigation controls */}
      <div className="carousel-controls">
        <button
          className="carousel-button carousel-prev"
          onClick={() => setCurrentIndex(prev => Math.max(prev - 1, 0))}
          disabled={currentIndex === 0}
          aria-label="Previous camera"
        >
          ←
        </button>
        <div className="carousel-indicators">
          {cameras.map((_, index) => (
            <button
              key={index}
              className={`indicator ${index === currentIndex ? 'active' : ''}`}
              onClick={() => setCurrentIndex(index)}
              aria-label={`Camera ${index + 1} of ${cameras.length}`}
            />
          ))}
        </div>
        <button
          className="carousel-button carousel-next"
          onClick={() => setCurrentIndex(prev => 
            Math.min(prev + 1, cameras.length - 1)
          )}
          disabled={currentIndex === cameras.length - 1}
          aria-label="Next camera"
        >
          →
        </button>
      </div>
    </div>
  );
};
```

### Progressive Web App (PWA) Features

#### Service Worker Implementation
```typescript
// service-worker.ts
const CACHE_NAME = 'its-camera-ai-v1.0.0';
const STATIC_CACHE_URLS = [
  '/',
  '/dashboard',
  '/cameras',
  '/analytics',
  '/offline.html'
];

// Install event - cache static assets
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(STATIC_CACHE_URLS))
  );
});

// Fetch event - serve from cache when offline
self.addEventListener('fetch', (event) => {
  if (event.request.url.includes('/api/')) {
    // Network-first strategy for API calls
    event.respondWith(
      fetch(event.request)
        .then(response => {
          // Cache successful API responses
          if (response.status === 200) {
            const responseClone = response.clone();
            caches.open(CACHE_NAME)
              .then(cache => cache.put(event.request, responseClone));
          }
          return response;
        })
        .catch(() => {
          // Fallback to cached response when offline
          return caches.match(event.request);
        })
    );
  } else {
    // Cache-first strategy for static assets
    event.respondWith(
      caches.match(event.request)
        .then(response => response || fetch(event.request))
    );
  }
});
```

#### Offline Support
```tsx
// Offline status component
const OfflineIndicator = () => {
  const [isOnline, setIsOnline] = useState(navigator.onLine);
  const [lastSync, setLastSync] = useState<Date | null>(null);

  useEffect(() => {
    const handleOnline = () => {
      setIsOnline(true);
      setLastSync(new Date());
      // Sync pending changes when back online
      syncPendingChanges();
    };

    const handleOffline = () => setIsOnline(false);

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  if (isOnline) {
    return (
      <div className="connection-status online">
        <span className="status-dot bg-green-500" />
        Online
        {lastSync && (
          <span className="text-xs text-gray-500">
            Synced {formatDistanceToNow(lastSync)} ago
          </span>
        )}
      </div>
    );
  }

  return (
    <div 
      className="connection-status offline"
      role="alert"
      aria-live="polite"
    >
      <span className="status-dot bg-red-500" />
      Offline - Some features limited
      <button 
        className="retry-button"
        onClick={() => window.location.reload()}
      >
        Retry Connection
      </button>
    </div>
  );
};
```

---

## Performance Optimization

### Loading Strategies

#### Lazy Loading Implementation
```tsx
// Lazy load components with suspense
const LazyAnalyticsPage = lazy(() => import('./pages/AnalyticsPage'));
const LazySettingsPage = lazy(() => import('./pages/SettingsPage'));

const App = () => {
  return (
    <Router>
      <Suspense 
        fallback={
          <div className="loading-container">
            <LoadingSkeleton />
          </div>
        }
      >
        <Routes>
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/analytics" element={<LazyAnalyticsPage />} />
          <Route path="/settings" element={<LazySettingsPage />} />
        </Routes>
      </Suspense>
    </Router>
  );
};
```

#### Virtual Scrolling for Large Lists
```tsx
// Virtual scrolling for camera list
const VirtualizedCameraList = ({ cameras }) => {
  const listRef = useRef<HTMLDivElement>(null);
  const itemHeight = 120; // Fixed height per camera card
  const containerHeight = 600; // Visible container height
  const visibleCount = Math.ceil(containerHeight / itemHeight);
  
  const [scrollTop, setScrollTop] = useState(0);
  const [startIndex, setStartIndex] = useState(0);
  
  const handleScroll = useCallback((e: React.UIEvent<HTMLDivElement>) => {
    const newScrollTop = e.currentTarget.scrollTop;
    setScrollTop(newScrollTop);
    setStartIndex(Math.floor(newScrollTop / itemHeight));
  }, [itemHeight]);
  
  const endIndex = Math.min(startIndex + visibleCount + 1, cameras.length);
  const visibleCameras = cameras.slice(startIndex, endIndex);
  
  return (
    <div
      ref={listRef}
      className="virtualized-list"
      style={{ height: containerHeight, overflow: 'auto' }}
      onScroll={handleScroll}
    >
      <div style={{ height: cameras.length * itemHeight, position: 'relative' }}>
        {visibleCameras.map((camera, index) => (
          <CameraCard
            key={camera.id}
            camera={camera}
            style={{
              position: 'absolute',
              top: (startIndex + index) * itemHeight,
              width: '100%',
              height: itemHeight
            }}
          />
        ))}
      </div>
    </div>
  );
};
```

#### Image Optimization
```tsx
// Optimized image loading with placeholder
const OptimizedCameraStream = ({ 
  streamUrl, 
  placeholder, 
  alt 
}: {
  streamUrl: string;
  placeholder: string;
  alt: string;
}) => {
  const [imageLoaded, setImageLoaded] = useState(false);
  const [imageError, setImageError] = useState(false);
  const imgRef = useRef<HTMLImageElement>(null);
  
  useEffect(() => {
    // Preload image
    const img = new Image();
    img.onload = () => setImageLoaded(true);
    img.onerror = () => setImageError(true);
    img.src = streamUrl;
  }, [streamUrl]);

  if (imageError) {
    return (
      <div className="image-error">
        <CameraOffIcon size={48} />
        <span>Stream unavailable</span>
      </div>
    );
  }

  return (
    <div className="image-container">
      {!imageLoaded && (
        <img
          src={placeholder}
          alt=""
          className="placeholder-image"
          aria-hidden="true"
        />
      )}
      <img
        ref={imgRef}
        src={streamUrl}
        alt={alt}
        className={`stream-image ${imageLoaded ? 'loaded' : 'loading'}`}
        onLoad={() => setImageLoaded(true)}
        onError={() => setImageError(true)}
        loading="lazy"
      />
      {!imageLoaded && !imageError && (
        <div className="loading-spinner" aria-label="Loading stream..." />
      )}
    </div>
  );
};
```

### Caching Strategies

#### Service Worker Caching
```typescript
// Advanced caching strategies
const CACHE_STRATEGIES = {
  STATIC: 'cache-first',
  API_DATA: 'network-first', 
  USER_CONTENT: 'cache-first',
  REAL_TIME: 'network-only'
};

class CacheManager {
  private static instance: CacheManager;
  private caches: Map<string, Cache> = new Map();

  static getInstance(): CacheManager {
    if (!CacheManager.instance) {
      CacheManager.instance = new CacheManager();
    }
    return CacheManager.instance;
  }

  async cacheResponse(
    request: Request, 
    response: Response, 
    strategy: string,
    ttl?: number
  ) {
    if (strategy === CACHE_STRATEGIES.NETWORK_ONLY) return;
    
    const cache = await caches.open('its-camera-ai');
    const responseToCache = response.clone();
    
    if (ttl) {
      // Add expiration metadata
      const headers = new Headers(responseToCache.headers);
      headers.set('sw-cache-expires', 
        (Date.now() + ttl * 1000).toString()
      );
      
      const modifiedResponse = new Response(responseToCache.body, {
        status: responseToCache.status,
        statusText: responseToCache.statusText,
        headers
      });
      
      await cache.put(request, modifiedResponse);
    } else {
      await cache.put(request, responseToCache);
    }
  }

  async getCachedResponse(request: Request): Promise<Response | null> {
    const cache = await caches.open('its-camera-ai');
    const response = await cache.match(request);
    
    if (!response) return null;
    
    // Check expiration
    const expires = response.headers.get('sw-cache-expires');
    if (expires && parseInt(expires) < Date.now()) {
      // Cache expired, remove it
      await cache.delete(request);
      return null;
    }
    
    return response;
  }
}
```

---

## Error Handling & Feedback

### Error Boundaries

```tsx
// Comprehensive error boundary with reporting
class ErrorBoundary extends Component<
  { children: React.ReactNode; fallback?: React.ComponentType<any> },
  { hasError: boolean; error: Error | null; errorInfo: ErrorInfo | null }
> {
  constructor(props: any) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    this.setState({
      error,
      errorInfo
    });
    
    // Log error to monitoring service
    logErrorToService({
      error: error.message,
      stack: error.stack,
      componentStack: errorInfo.componentStack,
      timestamp: new Date().toISOString(),
      userAgent: navigator.userAgent,
      url: window.location.href
    });
  }

  render() {
    if (this.state.hasError) {
      const FallbackComponent = this.props.fallback || DefaultErrorFallback;
      return (
        <FallbackComponent 
          error={this.state.error}
          errorInfo={this.state.errorInfo}
          onRetry={() => this.setState({ hasError: false, error: null, errorInfo: null })}
        />
      );
    }

    return this.props.children;
  }
}

// Default error fallback component
const DefaultErrorFallback = ({ 
  error, 
  errorInfo, 
  onRetry 
}: {
  error: Error | null;
  errorInfo: ErrorInfo | null;
  onRetry: () => void;
}) => {
  return (
    <div 
      className="error-boundary"
      role="alert"
      aria-live="assertive"
    >
      <h2>Something went wrong</h2>
      <p>
        We're sorry, but something unexpected happened. 
        Our team has been notified.
      </p>
      <div className="error-actions">
        <button 
          onClick={onRetry}
          className="button-primary"
        >
          Try Again
        </button>
        <button 
          onClick={() => window.location.reload()}
          className="button-secondary"
        >
          Refresh Page
        </button>
      </div>
      
      {process.env.NODE_ENV === 'development' && (
        <details className="error-details">
          <summary>Error Details (Development)</summary>
          <pre>{error?.stack}</pre>
          <pre>{errorInfo?.componentStack}</pre>
        </details>
      )}
    </div>
  );
};
```

### Loading States & Skeletons

```tsx
// Skeleton components for different content types
const DashboardSkeleton = () => {
  return (
    <div className="dashboard-skeleton">
      <div className="skeleton-header">
        <div className="skeleton-text h-8 w-64 mb-2" />
        <div className="skeleton-text h-4 w-96" />
      </div>
      
      <div className="skeleton-metrics grid grid-cols-4 gap-4 my-6">
        {[...Array(4)].map((_, i) => (
          <div key={i} className="skeleton-card">
            <div className="skeleton-text h-4 w-24 mb-2" />
            <div className="skeleton-text h-8 w-16" />
          </div>
        ))}
      </div>
      
      <div className="skeleton-content">
        <div className="skeleton-card h-96" />
      </div>
    </div>
  );
};

// CSS for skeleton animations
const skeletonStyles = `
.skeleton-text, .skeleton-card {
  background: linear-gradient(
    90deg,
    #f0f0f0 25%,
    #e0e0e0 37%,
    #f0f0f0 63%
  );
  background-size: 400% 100%;
  animation: skeleton-loading 1.4s ease-in-out infinite;
  border-radius: 4px;
}

@keyframes skeleton-loading {
  0% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}

@media (prefers-reduced-motion: reduce) {
  .skeleton-text, .skeleton-card {
    animation: none;
    background: #f0f0f0;
  }
}
`;
```

### Toast Notification System

```tsx
// Toast notification context and provider
interface ToastContextType {
  showToast: (toast: ToastOptions) => void;
  hideToast: (id: string) => void;
  hideAllToasts: () => void;
}

interface ToastOptions {
  type: 'success' | 'error' | 'warning' | 'info';
  title: string;
  message?: string;
  duration?: number;
  persistent?: boolean;
  actions?: ToastAction[];
}

const ToastProvider = ({ children }: { children: React.ReactNode }) => {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const showToast = useCallback((options: ToastOptions) => {
    const id = generateId();
    const toast: Toast = {
      id,
      ...options,
      timestamp: new Date()
    };

    setToasts(prev => [...prev, toast]);

    // Auto-dismiss non-persistent toasts
    if (!options.persistent) {
      setTimeout(() => {
        hideToast(id);
      }, options.duration || 5000);
    }
  }, []);

  const hideToast = useCallback((id: string) => {
    setToasts(prev => prev.filter(toast => toast.id !== id));
  }, []);

  return (
    <ToastContext.Provider value={{ showToast, hideToast, hideAllToasts: () => setToasts([]) }}>
      {children}
      <ToastContainer toasts={toasts} onDismiss={hideToast} />
    </ToastContext.Provider>
  );
};

// Toast container component
const ToastContainer = ({ 
  toasts, 
  onDismiss 
}: {
  toasts: Toast[];
  onDismiss: (id: string) => void;
}) => {
  return (
    <div 
      className="toast-container fixed top-4 right-4 z-50 space-y-2"
      role="region"
      aria-label="Notifications"
    >
      <AnimatePresence>
        {toasts.map(toast => (
          <ToastItem
            key={toast.id}
            toast={toast}
            onDismiss={onDismiss}
          />
        ))}
      </AnimatePresence>
    </div>
  );
};
```

---

## Implementation Best Practices

### Code Organization

#### Component Structure
```
src/
├── components/
│   ├── ui/                 # Base shadcn/ui components
│   ├── common/             # Shared components
│   ├── features/           # Feature-specific components
│   │   ├── dashboard/
│   │   ├── cameras/
│   │   ├── analytics/
│   │   └── security/
│   └── layout/             # Layout components
├── hooks/                  # Custom React hooks
├── utils/                  # Utility functions
├── types/                  # TypeScript type definitions
├── constants/              # Application constants
└── styles/                 # Global styles and themes
```

#### Component Development Guidelines

```tsx
// Example: Well-structured component with TypeScript
interface CameraCardProps {
  camera: CameraInfo;
  variant?: 'compact' | 'detailed';
  onAction?: (action: string, cameraId: string) => void;
  className?: string;
}

const CameraCard = memo(({
  camera,
  variant = 'compact',
  onAction,
  className
}: CameraCardProps) => {
  // Hooks at the top
  const [isLoading, setIsLoading] = useState(false);
  const { user } = useAuth();
  const { showToast } = useToast();

  // Event handlers
  const handleStatusClick = useCallback(async () => {
    if (!onAction) return;
    
    setIsLoading(true);
    try {
      await onAction('refresh', camera.id);
      showToast({
        type: 'success',
        title: 'Camera refreshed',
        message: `${camera.name} status updated`
      });
    } catch (error) {
      showToast({
        type: 'error',
        title: 'Refresh failed',
        message: 'Unable to refresh camera status'
      });
    } finally {
      setIsLoading(false);
    }
  }, [camera.id, camera.name, onAction, showToast]);

  // Computed values
  const statusColor = getStatusColor(camera.status);
  const canControl = user?.permissions.includes('camera:control');

  // Early returns for error states
  if (!camera) {
    return <div className="camera-card-error">Camera data unavailable</div>;
  }

  return (
    <Card className={cn('camera-card', `camera-card--${variant}`, className)}>
      <CardHeader>
        <div className="flex justify-between items-center">
          <CardTitle className="text-sm font-medium">
            {camera.name}
          </CardTitle>
          <Badge 
            variant={statusColor}
            className="cursor-pointer"
            onClick={handleStatusClick}
            disabled={isLoading}
          >
            {isLoading ? 'Updating...' : camera.status}
          </Badge>
        </div>
        {variant === 'detailed' && (
          <CardDescription>{camera.location}</CardDescription>
        )}
      </CardHeader>
      
      <CardContent>
        {/* Component content */}
      </CardContent>
    </Card>
  );
});

// Display name for debugging
CameraCard.displayName = 'CameraCard';

export default CameraCard;
```

#### Custom Hook Patterns

```tsx
// Example: Custom hook for real-time data
const useRealTimeData = <T>(
  url: string,
  options: {
    refreshInterval?: number;
    enabled?: boolean;
    onError?: (error: Error) => void;
  } = {}
) => {
  const {
    refreshInterval = 5000,
    enabled = true,
    onError
  } = options;

  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const fetchData = useCallback(async () => {
    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const result = await response.json();
      setData(result);
      setError(null);
    } catch (err) {
      const error = err instanceof Error ? err : new Error('Unknown error');
      setError(error);
      onError?.(error);
    } finally {
      setLoading(false);
    }
  }, [url, onError]);

  useEffect(() => {
    if (!enabled) return;

    fetchData();
    const interval = setInterval(fetchData, refreshInterval);

    return () => clearInterval(interval);
  }, [fetchData, refreshInterval, enabled]);

  return {
    data,
    loading,
    error,
    refetch: fetchData
  };
};
```

### Performance Optimization

#### Memoization Strategies
```tsx
// Memo for expensive calculations
const ExpensiveChart = memo(({ data, options }) => {
  const processedData = useMemo(() => {
    return data.map(item => ({
      ...item,
      processed: expensiveCalculation(item)
    }));
  }, [data]);

  const chartConfig = useMemo(() => ({
    ...options,
    theme: getChartTheme()
  }), [options]);

  return <Chart data={processedData} config={chartConfig} />;
});

// Callback memoization to prevent unnecessary re-renders
const CameraGrid = ({ cameras, onCameraSelect }) => {
  const handleCameraSelect = useCallback((cameraId: string) => {
    onCameraSelect?.(cameraId);
  }, [onCameraSelect]);

  return (
    <div className="camera-grid">
      {cameras.map(camera => (
        <CameraCard
          key={camera.id}
          camera={camera}
          onSelect={handleCameraSelect}
        />
      ))}
    </div>
  );
};
```

---

## Testing & Quality Assurance

### Accessibility Testing

#### Automated Testing
```typescript
// Example: Jest + Testing Library accessibility tests
import { render, screen } from '@testing-library/react';
import { axe, toHaveNoViolations } from 'jest-axe';
import userEvent from '@testing-library/user-event';

expect.extend(toHaveNoViolations);

describe('CameraCard Accessibility', () => {
  test('should not have any accessibility violations', async () => {
    const { container } = render(
      <CameraCard camera={mockCamera} />
    );
    
    const results = await axe(container);
    expect(results).toHaveNoViolations();
  });

  test('should be keyboard navigable', async () => {
    const user = userEvent.setup();
    const onAction = jest.fn();
    
    render(
      <CameraCard camera={mockCamera} onAction={onAction} />
    );

    // Test keyboard navigation
    const statusBadge = screen.getByRole('button', { name: /online/i });
    await user.tab();
    expect(statusBadge).toHaveFocus();
    
    await user.keyboard('{Enter}');
    expect(onAction).toHaveBeenCalledWith('refresh', mockCamera.id);
  });

  test('should have proper ARIA labels', () => {
    render(<CameraCard camera={mockCamera} />);
    
    expect(screen.getByLabelText(/camera status/i)).toBeInTheDocument();
    expect(screen.getByRole('button')).toHaveAttribute('aria-describedby');
  });
});
```

#### Manual Testing Checklist
```markdown
## Accessibility Testing Checklist

### Keyboard Navigation
- [ ] All interactive elements are keyboard accessible
- [ ] Tab order is logical and follows visual layout
- [ ] Focus indicators are clearly visible
- [ ] Escape key closes modals/dropdowns
- [ ] Arrow keys navigate within components

### Screen Reader Testing
- [ ] Content is read in logical order
- [ ] All images have appropriate alt text
- [ ] Form fields have associated labels
- [ ] Error messages are announced
- [ ] Live regions announce updates appropriately

### Visual Testing
- [ ] Text has sufficient color contrast (4.5:1 minimum)
- [ ] Interface is usable at 200% zoom
- [ ] Focus indicators are visible in high contrast mode
- [ ] No information is conveyed by color alone

### Motor Accessibility
- [ ] Touch targets are minimum 44px
- [ ] Hover states don't require precise mouse control
- [ ] Drag and drop has keyboard alternatives
- [ ] Time limits can be extended or disabled
```

### Performance Testing

#### Core Web Vitals Monitoring
```typescript
// Web Vitals monitoring
import { getCLS, getFID, getFCP, getLCP, getTTFB } from 'web-vitals';

const sendToAnalytics = (metric: any) => {
  // Send to your analytics service
  console.log(metric);
};

// Monitor Core Web Vitals
getCLS(sendToAnalytics);
getFID(sendToAnalytics);
getFCP(sendToAnalytics);
getLCP(sendToAnalytics);
getTTFB(sendToAnalytics);

// Performance budget thresholds
const PERFORMANCE_BUDGETS = {
  LCP: 2500,    // Largest Contentful Paint
  FID: 100,     // First Input Delay
  CLS: 0.1,     // Cumulative Layout Shift
  FCP: 1800,    // First Contentful Paint
  TTFB: 800     // Time to First Byte
};
```

This comprehensive UX guidelines document provides the foundation for creating accessible, performant, and user-friendly traffic monitoring interfaces that serve all personas effectively while maintaining high quality standards.