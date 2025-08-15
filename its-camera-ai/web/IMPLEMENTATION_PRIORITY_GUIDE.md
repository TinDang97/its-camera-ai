# ITS Camera AI Dashboard - Priority Implementation Guide

## Quick Start for Developers

This guide provides a priority-ordered, step-by-step implementation path for building the ITS Camera AI dashboard. Follow these tasks in order to ensure proper dependencies and optimal development flow.

---

## Week 1-3: Foundation Phase (Critical Path)

### Day 1-2: Design System Setup
```bash
# Priority: P0 - Blocks all UI development
Task ITS-001: Design System Foundation
```

**Implementation Steps:**
1. Update `/web/tailwind.config.ts` with semantic colors
2. Create `/web/lib/design-tokens.ts` for design system exports
3. Update `/web/app/globals.css` with CSS custom properties
4. Test color contrast ratios with axe-core

**Validation:** Run `npm run test:colors` to verify WCAG compliance

---

### Day 3-4: Typography & Layout System
```bash
# Priority: P0 - Required for all components
Task ITS-002: Typography System
Task ITS-003: Design Token System
```

**Implementation Steps:**
1. Install fonts: `npm install @next/font`
2. Configure font loading in `/web/app/layout.tsx`
3. Create responsive typography scale
4. Implement layout grid system

**Quick Test:** Check font loading performance in Network tab (<100ms)

---

### Day 5-7: Core Performance Components
```bash
# Priority: P0 - Used throughout dashboard
Task ITS-004: PerformanceIndicator Component
```

**File:** `/web/components/ui/performance-indicator.tsx`

```typescript
// Minimal working example to start with
export function PerformanceIndicator({ 
  latency, 
  accuracy, 
  confidence 
}: Props) {
  const getLatencyColor = (ms: number) => {
    if (ms < 50) return 'text-green-500';
    if (ms < 75) return 'text-lime-500';
    if (ms < 100) return 'text-amber-500';
    return 'text-red-500';
  };
  
  return (
    <div className="flex items-center gap-2">
      <span className={getLatencyColor(latency)}>
        {latency}ms
      </span>
      {/* Add other metrics */}
    </div>
  );
}
```

**Test Command:** `npm run test components/ui/performance-indicator`

---

### Day 8-10: System Health Monitoring
```bash
# Priority: P0 - Critical for operations dashboard
Task ITS-005: SystemHealthWidget
```

**File:** `/web/components/ui/system-health-widget.tsx`

**Dependencies to install:**
```bash
npm install date-fns clsx
```

**Key Features to Implement:**
1. Service status list with health indicators
2. Auto-refresh with configurable interval
3. Expandable details for each service
4. Overall system health summary

---

### Day 11-14: Layout & Navigation
```bash
# Priority: P0 - Framework for all pages
Task ITS-008: DashboardLayout Enhancement
```

**Files to Update:**
- `/web/components/layout/header.tsx` - Add performance indicators
- `/web/components/layout/sidebar.tsx` - Enhance navigation
- `/web/app/layout.tsx` - Integrate responsive layout

**Mobile Considerations:**
- Implement hamburger menu for mobile
- Add swipe gestures for navigation
- Ensure touch targets are 44x44px minimum

---

## Week 4-6: Real-time Integration Phase

### Day 15-17: WebSocket Foundation
```bash
# Priority: P0 - Enables all real-time features
Task ITS-009: Analytics WebSocket Client
```

**File:** `/web/lib/websocket-client.ts`

```typescript
// Start with basic connection management
class AnalyticsWebSocketClient {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  
  connect(url: string, token: string) {
    this.ws = new WebSocket(`${url}?token=${token}`);
    this.setupEventHandlers();
  }
  
  private setupEventHandlers() {
    this.ws.onopen = () => console.log('Connected');
    this.ws.onmessage = (event) => this.handleMessage(event);
    this.ws.onerror = (error) => this.handleError(error);
    this.ws.onclose = () => this.handleDisconnect();
  }
  
  // Add reconnection logic here
}
```

**Integration:** Update `/web/components/providers/websocket-provider.tsx`

---

### Day 18-20: Real-time Charts
```bash
# Priority: P1 - Core visualization
Task ITS-010: RealTimeChart Component
```

**File:** `/web/components/charts/real-time-chart.tsx`

**D3.js Setup:**
```bash
npm install d3 @types/d3
```

**Performance Tips:**
- Use `requestAnimationFrame` for smooth updates
- Implement data windowing (show last 100 points)
- Use canvas rendering for >1000 points

---

### Day 21-23: Traffic Heatmap
```bash
# Priority: P1 - Key traffic visualization
Task ITS-011: TrafficHeatmap Component
```

**File:** `/web/components/analytics/traffic-heatmap.tsx`

**Implementation Approach:**
1. Start with static heatmap using D3.js
2. Add real-time updates via WebSocket
3. Implement zoom/pan interactions
4. Add time range selector

---

## Week 7-9: Camera Management Phase

### Day 24-26: Camera Card Component
```bash
# Priority: P1 - Essential for camera monitoring
Task ITS-014: CameraCard with Live Preview
```

**File:** `/web/components/features/camera/CameraCard.tsx`

**HLS.js Integration:**
```bash
npm install hls.js
```

```typescript
// Basic HLS preview implementation
useEffect(() => {
  if (Hls.isSupported() && videoRef.current) {
    const hls = new Hls();
    hls.loadSource(streamUrl);
    hls.attachMedia(videoRef.current);
    return () => hls.destroy();
  }
}, [streamUrl]);
```

---

### Day 27-30: Live Stream Player
```bash
# Priority: P0 - Core functionality
Task ITS-016: LiveStreamPlayer
```

**File:** `/web/components/features/camera/LiveStreamPlayer.tsx`

**Multi-protocol Support:**
```typescript
const getStreamPlayer = (protocol: string) => {
  switch(protocol) {
    case 'hls': return HLSPlayer;
    case 'webrtc': return WebRTCPlayer;
    case 'rtsp': return RTSPProxyPlayer;
    default: return FallbackPlayer;
  }
};
```

---

## Week 10-12: Alert System Phase

### Day 31-33: Alert Panel
```bash
# Priority: P1 - Critical for incident response
Task ITS-017: AlertPanel Component
```

**File:** `/web/components/features/alerts/AlertPanel.tsx`

**State Management with Zustand:**
```typescript
// /web/stores/alert-store.ts
export const useAlertStore = create((set) => ({
  alerts: [],
  addAlert: (alert) => set((state) => ({
    alerts: [alert, ...state.alerts].slice(0, 100)
  })),
  dismissAlert: (id) => set((state) => ({
    alerts: state.alerts.filter(a => a.id !== id)
  }))
}));
```

---

## Week 13-15: Analytics Dashboard Phase

### Day 34-40: Complete Analytics Page
```bash
# Priority: P0 - Main dashboard functionality
Task ITS-019: Analytics Dashboard Implementation
```

**File Structure:**
```
/web/app/analytics/
  ├── page.tsx           # Main analytics page
  ├── layout.tsx         # Analytics layout
  ├── overview/
  │   └── page.tsx       # Overview tab
  ├── traffic-flow/
  │   └── page.tsx       # Traffic flow tab
  ├── incidents/
  │   └── page.tsx       # Incidents tab
  └── predictions/
      └── page.tsx       # Predictions tab
```

**Tab Implementation with Next.js App Router:**
```typescript
// Use parallel routes for tabs
export default function AnalyticsLayout({ children }) {
  return (
    <div>
      <TabNavigation />
      {children}
    </div>
  );
}
```

---

## Quick Development Commands

### Start Development
```bash
# Run these in separate terminals
npm run dev           # Next.js dev server
npm run dev:api      # Mock API server
npm run dev:full     # Both concurrently
```

### Component Development
```bash
# Generate component boilerplate
npx plop component [ComponentName]

# Run Storybook for component development
npm run storybook

# Test specific component
npm test -- ComponentName
```

### Real-time Testing
```bash
# Start WebSocket test server
npm run ws:test

# Monitor WebSocket connections
npm run ws:monitor
```

---

## Common Implementation Patterns

### Pattern 1: Real-time Data Hook
```typescript
// Reusable hook for real-time data
export function useRealtimeData(eventType: string) {
  const [data, setData] = useState(null);
  const { subscribe, unsubscribe } = useWebSocket();
  
  useEffect(() => {
    const handler = (newData) => setData(newData);
    subscribe(eventType, handler);
    return () => unsubscribe(eventType, handler);
  }, [eventType]);
  
  return data;
}
```

### Pattern 2: Responsive Component
```typescript
// Responsive component wrapper
export function ResponsiveComponent({ children }) {
  const isMobile = useMediaQuery('(max-width: 768px)');
  const isTablet = useMediaQuery('(max-width: 1024px)');
  
  return (
    <div className={cn(
      'base-styles',
      isMobile && 'mobile-styles',
      isTablet && 'tablet-styles'
    )}>
      {children}
    </div>
  );
}
```

### Pattern 3: Error Boundary
```typescript
// Wrap components for error handling
export function ComponentWithErrorBoundary() {
  return (
    <ErrorBoundary fallback={<ErrorFallback />}>
      <YourComponent />
    </ErrorBoundary>
  );
}
```

---

## Performance Optimization Checklist

### Before Each Component Ships
1. [ ] Lazy load if not critical path
2. [ ] Memoize expensive calculations
3. [ ] Use virtual scrolling for lists >100 items
4. [ ] Implement proper cleanup in useEffect
5. [ ] Check for memory leaks
6. [ ] Profile with React DevTools
7. [ ] Test on slow 3G network
8. [ ] Verify mobile performance

### WebSocket Optimization
1. [ ] Implement message batching
2. [ ] Use binary format for large data
3. [ ] Add compression (if supported)
4. [ ] Throttle updates to 60fps max
5. [ ] Buffer messages during reconnection

---

## Testing Priority

### Unit Tests (Required)
```bash
# Test individual components
npm test components/ui/PerformanceIndicator
npm test lib/websocket-client
```

### Integration Tests (Important)
```bash
# Test component interactions
npm run test:integration
```

### E2E Tests (Critical Paths)
```bash
# Test user journeys
npm run test:e2e -- --grep "critical"
```

---

## Deployment Readiness Checklist

### Before First Deployment
1. [ ] All P0 tasks completed
2. [ ] Core components tested
3. [ ] WebSocket connection stable
4. [ ] Performance metrics met
5. [ ] Mobile responsive
6. [ ] Error handling complete

### Before Production
1. [ ] All tasks completed
2. [ ] 90% test coverage
3. [ ] Performance audit passed
4. [ ] Security scan clean
5. [ ] Documentation complete
6. [ ] Monitoring configured

---

## Support & Resources

### Documentation
- Component docs: `/docs/components/`
- API reference: `/docs/api/`
- WebSocket protocol: `/docs/websocket.md`

### Development Tools
- Storybook: http://localhost:6006
- API Mock: http://localhost:3001
- WebSocket Test: ws://localhost:3002

### Performance Monitoring
- Bundle analyzer: `npm run analyze`
- Lighthouse: `npm run lighthouse`
- Performance test: `npm run perf:test`

---

## Emergency Procedures

### If WebSocket Won't Connect
1. Check auth token is valid
2. Verify backend is running
3. Check network tab for errors
4. Test with wscat tool
5. Review CORS settings

### If Performance Degrades
1. Check React DevTools Profiler
2. Review Network waterfall
3. Check for memory leaks
4. Disable animations temporarily
5. Reduce data update frequency

### If Build Fails
1. Clear node_modules and reinstall
2. Check for TypeScript errors
3. Verify all imports resolve
4. Check environment variables
5. Review build logs carefully

---

This priority guide ensures developers can start implementing immediately with clear direction and practical examples. Follow the weekly phases in order for the smoothest development experience.