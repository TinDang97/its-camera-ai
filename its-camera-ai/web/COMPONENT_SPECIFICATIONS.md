# ITS Camera AI - Component Specifications

## Overview

This document provides detailed specifications for custom UI components that extend the shadcn/ui foundation to meet the specific needs of traffic monitoring applications. Each component is designed with accessibility, performance, and user experience as primary considerations.

## Table of Contents

1. [Performance & Status Components](#performance--status-components)
2. [Traffic Monitoring Components](#traffic-monitoring-components)
3. [Data Visualization Components](#data-visualization-components)
4. [Camera Management Components](#camera-management-components)
5. [Alert & Notification Components](#alert--notification-components)
6. [Security Components](#security-components)
7. [Layout & Navigation Components](#layout--navigation-components)

---

## Performance & Status Components

### PerformanceIndicator

**Purpose**: Display real-time system performance metrics with color-coded status indicators.

```tsx
interface PerformanceIndicatorProps {
  latency: number; // in milliseconds
  accuracy?: number; // 0-100 percentage
  confidence?: number; // 0-100 percentage
  throughput?: number; // frames per second
  size?: 'sm' | 'md' | 'lg';
  orientation?: 'horizontal' | 'vertical';
  showLabels?: boolean;
  showValues?: boolean;
  animated?: boolean;
}

// Usage Example
<PerformanceIndicator 
  latency={45} 
  accuracy={94.2} 
  confidence={87} 
  throughput={847}
  size="md" 
  orientation="horizontal"
  showLabels
  showValues
  animated
/>
```

**Design Specifications:**
- **Latency Colors**: Green (<50ms), Lime (50-75ms), Amber (75-100ms), Red (>100ms)
- **Accuracy Colors**: Green (>90%), Amber (80-90%), Red (<80%)
- **Confidence Colors**: Green (>85%), Amber (70-85%), Red (<70%)
- **Size Variants**: sm (24px), md (32px), lg (48px)
- **Animation**: Smooth transitions on value changes (0.3s ease-out)
- **Accessibility**: ARIA labels, screen reader support

**Tailwind Classes:**
```css
.performance-indicator {
  @apply inline-flex items-center gap-2 p-2 rounded-lg border;
}

.performance-indicator--sm { @apply text-xs; }
.performance-indicator--md { @apply text-sm; }
.performance-indicator--lg { @apply text-base; }

.performance-indicator__metric {
  @apply flex items-center gap-1 font-mono;
}

.performance-indicator__dot {
  @apply w-2 h-2 rounded-full transition-colors duration-300;
}
```

### SystemHealthWidget

**Purpose**: Comprehensive system health overview with expandable details.

```tsx
interface SystemHealthWidgetProps {
  services: ServiceStatus[];
  overallHealth: 'excellent' | 'good' | 'warning' | 'critical';
  lastCheck?: Date;
  autoRefresh?: boolean;
  refreshInterval?: number; // seconds
  onRefresh?: () => void;
  expandable?: boolean;
}

interface ServiceStatus {
  name: string;
  status: 'healthy' | 'warning' | 'critical' | 'unknown';
  uptime?: number; // percentage
  responseTime?: number; // milliseconds
  lastCheck?: Date;
  details?: string;
}
```

**Design Specifications:**
- **Overall Health Colors**: Green (excellent), Lime (good), Amber (warning), Red (critical)
- **Service Status Icons**: ✅ (healthy), ⚠️ (warning), 🔴 (critical), ❓ (unknown)
- **Expandable Details**: Accordion-style expansion with smooth animation
- **Auto-refresh**: Visual countdown indicator when enabled

---

## Traffic Monitoring Components

### TrafficStatusBadge

**Purpose**: Display traffic flow status with semantic colors and optional animation.

```tsx
interface TrafficStatusBadgeProps {
  status: 'optimal' | 'moderate' | 'congested' | 'blocked' | 'unknown';
  location?: string;
  vehicleCount?: number;
  averageSpeed?: number; // km/h
  lastUpdated?: Date;
  size?: 'sm' | 'md' | 'lg';
  animated?: boolean;
  onClick?: () => void;
}

// Usage Example
<TrafficStatusBadge 
  status="moderate"
  location="Main St & 5th Ave"
  vehicleCount={42}
  averageSpeed={35}
  lastUpdated={new Date()}
  size="md"
  animated
  onClick={() => navigateToLocation()}
/>
```

**Design Specifications:**
- **Status Colors**: 
  - Optimal: `text-traffic-optimal bg-traffic-optimal/10` (Green)
  - Moderate: `text-traffic-moderate bg-traffic-moderate/10` (Amber)
  - Congested: `text-traffic-congested bg-traffic-congested/10` (Red)
  - Blocked: `text-traffic-blocked bg-traffic-blocked/10` (Dark Red)
- **Animation**: Subtle pulse effect for active status changes
- **Tooltip**: Detailed information on hover/focus

### VehicleClassificationChart

**Purpose**: Visualize vehicle type distribution with interactive elements.

```tsx
interface VehicleClassificationChartProps {
  data: VehicleTypeData[];
  totalVehicles: number;
  timeRange?: string;
  chartType?: 'pie' | 'donut' | 'bar';
  showLegend?: boolean;
  showPercentages?: boolean;
  interactive?: boolean;
  onSegmentClick?: (vehicleType: string) => void;
}

interface VehicleTypeData {
  type: 'car' | 'truck' | 'motorcycle' | 'bus' | 'other';
  count: number;
  percentage: number;
  color?: string;
}
```

**Design Specifications:**
- **Chart Colors**: Consistent with traffic monitoring color palette
- **Interactive Elements**: Hover effects, click handlers for drill-down
- **Responsive Design**: Adapts to container size
- **Accessibility**: Keyboard navigation, screen reader support

### SpeedGauge

**Purpose**: Display current speed with target ranges and historical context.

```tsx
interface SpeedGaugeProps {
  currentSpeed: number; // km/h
  speedLimit?: number; // km/h
  targetRange?: { min: number; max: number };
  unit?: 'kmh' | 'mph';
  size?: 'sm' | 'md' | 'lg';
  showHistory?: boolean;
  historicalData?: number[];
}
```

**Design Specifications:**
- **Gauge Colors**: Green (within range), Amber (near limits), Red (exceeding)
- **Speed Ranges**: Visual indicators for optimal, acceptable, and problematic speeds
- **Animation**: Smooth needle movement with realistic physics

---

## Data Visualization Components

### RealTimeChart

**Purpose**: Display streaming data with smooth animations and zoom capabilities.

```tsx
interface RealTimeChartProps {
  data: TimeSeriesData[];
  xAxis?: AxisConfig;
  yAxis?: AxisConfig;
  chartType?: 'line' | 'area' | 'bar';
  realTime?: boolean;
  updateInterval?: number; // milliseconds
  maxDataPoints?: number;
  showGrid?: boolean;
  showTooltip?: boolean;
  zoomEnabled?: boolean;
  onDataPointClick?: (point: TimeSeriesData) => void;
}

interface TimeSeriesData {
  timestamp: Date;
  value: number;
  metadata?: Record<string, any>;
}

interface AxisConfig {
  label?: string;
  min?: number;
  max?: number;
  format?: (value: number) => string;
}
```

**Design Specifications:**
- **Real-time Updates**: Smooth transitions for new data points
- **Zoom Controls**: Pinch-to-zoom, scroll wheel, time range selection
- **Grid Lines**: Subtle background grid with customizable spacing
- **Tooltip**: Context-rich information display

### TrafficHeatmap

**Purpose**: Spatial visualization of traffic density across monitored areas.

```tsx
interface TrafficHeatmapProps {
  data: HeatmapData[];
  mapBounds?: GeoBounds;
  colorScale?: ColorScale;
  intensity?: number;
  radius?: number;
  blur?: number;
  maxZoom?: number;
  onAreaClick?: (area: HeatmapData) => void;
}

interface HeatmapData {
  lat: number;
  lng: number;
  intensity: number;
  location?: string;
  vehicleCount?: number;
}
```

**Design Specifications:**
- **Color Scale**: Blue (low) → Green (medium) → Yellow (high) → Red (very high)
- **Interactive Elements**: Click/tap areas for detailed information
- **Responsive Sizing**: Adapts to container dimensions

### PredictiveChart

**Purpose**: Display predictions with confidence intervals and historical accuracy.

```tsx
interface PredictiveChartProps {
  historicalData: TimeSeriesData[];
  predictions: PredictionData[];
  confidenceIntervals?: ConfidenceInterval[];
  modelInfo?: ModelInfo;
  showAccuracy?: boolean;
  timeRange?: DateRange;
}

interface PredictionData extends TimeSeriesData {
  confidence: number; // 0-100
  modelVersion?: string;
}

interface ConfidenceInterval {
  timestamp: Date;
  lower: number;
  upper: number;
  confidence: number;
}
```

**Design Specifications:**
- **Prediction Visualization**: Dashed lines with confidence bands
- **Accuracy Indicators**: Historical vs. predicted comparison
- **Model Information**: Tooltip with model version and training date

---

## Camera Management Components

### CameraCard

**Purpose**: Compact camera information display with quick actions.

```tsx
interface CameraCardProps {
  camera: CameraInfo;
  variant?: 'compact' | 'detailed' | 'grid';
  showStream?: boolean;
  showControls?: boolean;
  onStreamClick?: (cameraId: string) => void;
  onConfigClick?: (cameraId: string) => void;
  onDiagnosticClick?: (cameraId: string) => void;
}

interface CameraInfo {
  id: string;
  name: string;
  location: string;
  status: 'online' | 'offline' | 'warning' | 'maintenance';
  signalStrength?: number; // 0-100
  aiConfidence?: number; // 0-100
  lastPing?: Date;
  streamUrl?: string;
  hasPtz?: boolean;
  model?: string;
  ipAddress?: string;
}
```

**Design Specifications:**
- **Status Colors**: Green (online), Red (offline), Amber (warning), Blue (maintenance)
- **Signal Strength**: Visual bar indicator with color coding
- **Stream Preview**: Thumbnail with loading states
- **Quick Actions**: PTZ controls, settings, diagnostics

### PTZControls

**Purpose**: Pan-tilt-zoom camera control interface.

```tsx
interface PTZControlsProps {
  cameraId: string;
  currentPosition?: PTZPosition;
  presetPositions?: PresetPosition[];
  capabilities?: PTZCapabilities;
  onMove?: (direction: PTZDirection, speed?: number) => void;
  onZoom?: (direction: 'in' | 'out', speed?: number) => void;
  onPresetSelect?: (presetId: string) => void;
  onPresetSave?: (name: string, position: PTZPosition) => void;
}

interface PTZPosition {
  pan: number; // -180 to 180 degrees
  tilt: number; // -90 to 90 degrees
  zoom: number; // zoom level
}

interface PTZCapabilities {
  canPan: boolean;
  canTilt: boolean;
  canZoom: boolean;
  panRange?: { min: number; max: number };
  tiltRange?: { min: number; max: number };
  zoomRange?: { min: number; max: number };
}
```

**Design Specifications:**
- **Control Layout**: Directional pad with center home button
- **Visual Feedback**: Active states, disabled states for unavailable movements
- **Preset Management**: Quick select with save/delete capabilities

### LiveStreamPlayer

**Purpose**: Video stream display with controls and overlays.

```tsx
interface LiveStreamPlayerProps {
  streamUrl: string;
  cameraName?: string;
  overlays?: StreamOverlay[];
  controls?: StreamControls;
  autoplay?: boolean;
  muted?: boolean;
  showStats?: boolean;
  onError?: (error: Error) => void;
  onStreamStart?: () => void;
  onStreamEnd?: () => void;
}

interface StreamOverlay {
  type: 'detection' | 'alert' | 'info';
  position: { x: number; y: number };
  content: React.ReactNode;
  persistent?: boolean;
}

interface StreamControls {
  playPause?: boolean;
  volume?: boolean;
  fullscreen?: boolean;
  screenshot?: boolean;
  record?: boolean;
}
```

**Design Specifications:**
- **Overlay System**: Non-intrusive information display
- **Controls**: Modern video player interface
- **Error Handling**: Graceful fallback with retry options
- **Performance**: Optimized for low latency streaming

---

## Alert & Notification Components

### AlertPanel

**Purpose**: Centralized alert management with filtering and actions.

```tsx
interface AlertPanelProps {
  alerts: AlertItem[];
  maxHeight?: string;
  groupBy?: 'severity' | 'type' | 'time' | 'location';
  sortBy?: 'time' | 'severity' | 'status';
  sortOrder?: 'asc' | 'desc';
  filters?: AlertFilters;
  onAlertClick?: (alert: AlertItem) => void;
  onAlertAck?: (alertId: string) => void;
  onAlertDismiss?: (alertId: string) => void;
  onBulkAction?: (action: string, alertIds: string[]) => void;
}

interface AlertItem {
  id: string;
  severity: 'info' | 'warning' | 'critical' | 'emergency';
  type: string;
  title: string;
  description: string;
  location?: string;
  timestamp: Date;
  status: 'active' | 'acknowledged' | 'resolved';
  source?: string;
  metadata?: Record<string, any>;
}
```

**Design Specifications:**
- **Severity Colors**: Info (blue), Warning (amber), Critical (red), Emergency (dark red)
- **Grouping**: Visual separation with headers and spacing
- **Bulk Actions**: Select all, acknowledge, dismiss multiple alerts
- **Real-time Updates**: Live updates with smooth animations

### NotificationToast

**Purpose**: Non-intrusive notification system for real-time updates.

```tsx
interface NotificationToastProps {
  notification: NotificationItem;
  position?: 'top-right' | 'top-left' | 'bottom-right' | 'bottom-left';
  duration?: number; // milliseconds
  persistent?: boolean;
  actions?: ToastAction[];
  onDismiss?: (id: string) => void;
  onActionClick?: (actionId: string, notificationId: string) => void;
}

interface NotificationItem {
  id: string;
  type: 'success' | 'info' | 'warning' | 'error';
  title: string;
  message?: string;
  icon?: React.ReactNode;
  timestamp?: Date;
}

interface ToastAction {
  id: string;
  label: string;
  variant?: 'default' | 'outline' | 'destructive';
}
```

**Design Specifications:**
- **Animation**: Slide-in from edge, fade-out on dismiss
- **Stacking**: Multiple notifications stack vertically
- **Auto-dismiss**: Configurable timeout with progress indicator

### IncidentCard

**Purpose**: Detailed incident information with response tracking.

```tsx
interface IncidentCardProps {
  incident: IncidentInfo;
  showDetails?: boolean;
  showActions?: boolean;
  onStatusChange?: (incidentId: string, status: IncidentStatus) => void;
  onAssignUser?: (incidentId: string, userId: string) => void;
  onAddNote?: (incidentId: string, note: string) => void;
}

interface IncidentInfo {
  id: string;
  title: string;
  description: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  status: 'open' | 'investigating' | 'resolved' | 'closed';
  location?: string;
  assignedTo?: string;
  createdAt: Date;
  updatedAt?: Date;
  notes?: IncidentNote[];
  relatedAlerts?: string[];
}
```

**Design Specifications:**
- **Status Workflow**: Clear progression indicators
- **Activity Timeline**: Chronological notes and updates
- **Assignment**: User selection with notification

---

## Security Components

### SecurityStatusIndicator

**Purpose**: Always-visible security status with detailed information.

```tsx
interface SecurityStatusIndicatorProps {
  status: 'secure' | 'monitoring' | 'warning' | 'breach';
  lastAudit?: Date;
  activeThreats?: number;
  complianceScore?: number; // 0-100
  onClick?: () => void;
  showDetails?: boolean;
  size?: 'sm' | 'md' | 'lg';
}
```

**Design Specifications:**
- **Status Icons**: Shield with checkmark (secure), eye (monitoring), warning triangle (warning), X (breach)
- **Colors**: Green (secure), blue (monitoring), amber (warning), red (breach)
- **Always Visible**: Persistent in header/navigation
- **Click Action**: Opens detailed security modal

### AuditLogViewer

**Purpose**: Security audit log display with filtering and search.

```tsx
interface AuditLogViewerProps {
  logs: AuditLogEntry[];
  filters?: AuditLogFilters;
  searchQuery?: string;
  pagination?: PaginationConfig;
  onFilterChange?: (filters: AuditLogFilters) => void;
  onExport?: (format: 'csv' | 'json' | 'pdf') => void;
}

interface AuditLogEntry {
  id: string;
  timestamp: Date;
  userId?: string;
  action: string;
  resource: string;
  ipAddress?: string;
  userAgent?: string;
  severity: 'info' | 'warning' | 'critical';
  details?: Record<string, any>;
}
```

**Design Specifications:**
- **Filterable Columns**: Date, user, action, resource, severity
- **Search**: Full-text search across all fields
- **Export Options**: Multiple formats with date range selection

### ComplianceScorecard

**Purpose**: Visual compliance status with detailed breakdown.

```tsx
interface ComplianceScorecardProps {
  overall: ComplianceScore;
  categories: ComplianceCategory[];
  showDetails?: boolean;
  onCategoryClick?: (categoryId: string) => void;
}

interface ComplianceScore {
  score: number; // 0-100
  status: 'compliant' | 'warning' | 'non-compliant';
  lastAssessment: Date;
  nextAssessment?: Date;
}

interface ComplianceCategory {
  id: string;
  name: string;
  score: number;
  status: 'pass' | 'warning' | 'fail';
  requirements: ComplianceRequirement[];
}
```

**Design Specifications:**
- **Score Visualization**: Circular progress indicator with color coding
- **Category Breakdown**: Expandable sections with requirement details
- **Status Colors**: Green (compliant), amber (warning), red (non-compliant)

---

## Layout & Navigation Components

### DashboardLayout

**Purpose**: Main application layout with responsive sidebar and header.

```tsx
interface DashboardLayoutProps {
  children: React.ReactNode;
  sidebar?: React.ReactNode;
  header?: React.ReactNode;
  breadcrumbs?: BreadcrumbItem[];
  sidebarCollapsed?: boolean;
  onSidebarToggle?: () => void;
}

interface BreadcrumbItem {
  label: string;
  href?: string;
  icon?: React.ReactNode;
}
```

**Design Specifications:**
- **Responsive Breakpoints**: Sidebar collapses on tablets and mobile
- **Persistent State**: Remember sidebar state across sessions
- **Accessibility**: Skip links, focus management

### TabNavigation

**Purpose**: Enhanced tab navigation with badges and icons.

```tsx
interface TabNavigationProps {
  tabs: TabItem[];
  activeTab: string;
  orientation?: 'horizontal' | 'vertical';
  variant?: 'default' | 'pills' | 'underline';
  onTabChange?: (tabId: string) => void;
}

interface TabItem {
  id: string;
  label: string;
  icon?: React.ReactNode;
  badge?: string | number;
  disabled?: boolean;
  content?: React.ReactNode;
}
```

**Design Specifications:**
- **Active States**: Clear visual indication of active tab
- **Badge Support**: Notifications, counts, status indicators
- **Keyboard Navigation**: Arrow keys, home, end support

### QuickActionBar

**Purpose**: Contextual action buttons with keyboard shortcuts.

```tsx
interface QuickActionBarProps {
  actions: QuickAction[];
  position?: 'top' | 'bottom' | 'fixed';
  variant?: 'default' | 'compact' | 'floating';
  showLabels?: boolean;
  showShortcuts?: boolean;
}

interface QuickAction {
  id: string;
  label: string;
  icon: React.ReactNode;
  shortcut?: string;
  onClick: () => void;
  disabled?: boolean;
  loading?: boolean;
  variant?: 'default' | 'primary' | 'destructive';
}
```

**Design Specifications:**
- **Keyboard Shortcuts**: Global hotkey support
- **Loading States**: Spinner overlays for async actions
- **Contextual Placement**: Position based on current page/view

---

## Implementation Guidelines

### Tailwind CSS Classes

```css
/* Performance Indicator Colors */
.performance-excellent { @apply text-performance-excellent border-performance-excellent/20 bg-performance-excellent/10; }
.performance-good { @apply text-performance-good border-performance-good/20 bg-performance-good/10; }
.performance-warning { @apply text-performance-warning border-performance-warning/20 bg-performance-warning/10; }
.performance-poor { @apply text-performance-poor border-performance-poor/20 bg-performance-poor/10; }

/* Traffic Status Colors */
.traffic-optimal { @apply text-traffic-optimal border-traffic-optimal/20 bg-traffic-optimal/10; }
.traffic-moderate { @apply text-traffic-moderate border-traffic-moderate/20 bg-traffic-moderate/10; }
.traffic-congested { @apply text-traffic-congested border-traffic-congested/20 bg-traffic-congested/10; }
.traffic-blocked { @apply text-traffic-blocked border-traffic-blocked/20 bg-traffic-blocked/10; }

/* Alert Severity Colors */
.alert-info { @apply text-alert-info border-alert-info/20 bg-alert-info/10; }
.alert-warning { @apply text-alert-warning border-alert-warning/20 bg-alert-warning/10; }
.alert-critical { @apply text-alert-critical border-alert-critical/20 bg-alert-critical/10; }
.alert-emergency { @apply text-alert-emergency border-alert-emergency/20 bg-alert-emergency/10; }

/* Animation Utilities */
.animate-pulse-slow { animation: pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite; }
.animate-fade-in { animation: fadeIn 0.3s ease-out; }
.animate-slide-up { animation: slideUp 0.3s ease-out; }

@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

@keyframes slideUp {
  from { 
    opacity: 0; 
    transform: translateY(10px); 
  }
  to { 
    opacity: 1; 
    transform: translateY(0); 
  }
}
```

### Accessibility Requirements

1. **Keyboard Navigation**
   - All interactive elements must be keyboard accessible
   - Tab order follows logical flow
   - Focus indicators are clearly visible

2. **Screen Reader Support**
   - ARIA labels for all interactive elements
   - Live regions for dynamic content updates
   - Semantic HTML structure

3. **Color Accessibility**
   - Minimum 4.5:1 contrast ratio for normal text
   - Color is never the sole indicator of information
   - High contrast mode support

4. **Motor Accessibility**
   - Minimum 44px touch targets on mobile
   - Generous click areas around small elements
   - Alternative interaction methods provided

### Performance Considerations

1. **Lazy Loading**: Components load only when needed
2. **Memoization**: Use React.memo for expensive components
3. **Virtualization**: Large lists use virtual scrolling
4. **Debouncing**: Search and filter inputs use appropriate delays
5. **Code Splitting**: Components are split at route level

This component specification provides a solid foundation for implementing professional, accessible, and performant UI components specifically designed for traffic monitoring applications.