# ITS Camera AI - Page Design Prototypes

## Overview

This document provides detailed wireframes, user flows, and design specifications for each page in the ITS Camera AI web application. Each page is designed to serve specific user personas and use cases while maintaining consistency with the overall design system.

## Table of Contents

1. [Enhanced Dashboard (`/dashboard`)](#enhanced-dashboard-dashboard)
2. [Advanced Camera Management (`/cameras`)](#advanced-camera-management-cameras)
3. [Analytics & Reporting (`/analytics`)](#analytics--reporting-analytics)
4. [Settings & Configuration (`/settings`)](#settings--configuration-settings)
5. [Security Dashboard (`/security`)](#security-dashboard-security)
6. [Admin Panel (`/admin`)](#admin-panel-admin)
7. [Mobile Design Patterns](#mobile-design-patterns)

---

## Enhanced Dashboard (`/dashboard`)

### User Personas & Goals
- **Traffic Operations Manager**: Real-time monitoring, incident response, system status
- **Traffic Engineer**: Quick access to key metrics and live camera feeds
- **City Planner**: Executive overview of system performance

### Page Layout Structure

```
┌─────────────────────────────────────────────────────────────────┐
│ Header: [Logo] [Performance Indicator] [Security Status] [User] │
├─────────────────────────────────────────────────────────────────┤
│ Page Header: "Traffic Monitoring Dashboard"                     │
│ Subtitle: "Real-time traffic analytics and AI insights"        │
│ Last Updated: "2 seconds ago" [Live indicator]                 │
├─────────────────────────────────────────────────────────────────┤
│ Key Performance Indicators (4-column grid)                     │
│ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐              │
│ │Latency  │ │AI Conf. │ │Cameras  │ │Alerts   │              │
│ │45ms     │ │94.2%    │ │48/52    │ │7 Active │              │
│ │[Green]  │ │[Green]  │ │[Amber]  │ │[Red]    │              │
│ └─────────┘ └─────────┘ └─────────┘ └─────────┘              │
├─────────────────────────────────────────────────────────────────┤
│ Primary Content Tabs                                           │
│ [Overview] [Live Cameras] [Alerts] [Analytics] [System Health] │
├─────────────────────────────────────────────────────────────────┤
│ Tab Content Area (dynamic based on selection)                  │
└─────────────────────────────────────────────────────────────────┘
```

### Overview Tab Layout

```
┌─────────────────────────────────────────────────────────────────┐
│ Critical Alerts Panel (if any active)                          │
│ ⚠️  3 Critical Alerts Require Attention                        │
│ • Camera_North_01 - Connection Lost (2 min ago)               │
│ • Intersection_Main_5th - Heavy Congestion (5 min ago)        │
│ • Security_Alert - Unusual Access Pattern (12 min ago)        │
├─────────────────────────────────────────────────────────────────┤
│ Real-time Metrics Grid (2-column responsive)                   │
│ ┌─────────────────────────┐ ┌─────────────────────────────────┐ │
│ │ Traffic Flow Metrics    │ │ AI Performance Metrics          │ │
│ │ • Vehicle Count: 1,284  │ │ • Model Accuracy: 94.2%         │ │
│ │ • Avg Speed: 42 km/h    │ │ • Inference Time: 45ms          │ │
│ │ • Congestion: Moderate  │ │ • GPU Utilization: 73%          │ │
│ │ • Incidents: 3 active   │ │ • Queue Length: 12 frames       │ │
│ └─────────────────────────┘ └─────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│ Live Camera Grid (2x2 preview, expandable)                     │
│ ┌─────────┐ ┌─────────┐                                        │
│ │Cam_01   │ │Cam_02   │   [View All Cameras] [Full Screen]    │
│ │[Live]   │ │[Live]   │                                        │
│ └─────────┘ └─────────┘                                        │
│ ┌─────────┐ ┌─────────┐                                        │
│ │Cam_03   │ │Cam_04   │                                        │
│ │[Live]   │ │[Offline]│                                        │
│ └─────────┘ └─────────┘                                        │
└─────────────────────────────────────────────────────────────────┘
```

### Live Cameras Tab

```
┌─────────────────────────────────────────────────────────────────┐
│ Camera Controls Bar                                             │
│ Layout: [1x1] [2x2] [3x3] [4x4] | Filter: [All] [Online] [⚠️]  │
│ View: [Grid] [List] [Map] | Sort: [Location] [Status] [Name]   │
├─────────────────────────────────────────────────────────────────┤
│ Camera Grid (responsive, based on layout selection)            │
│ ┌─────────┐ ┌─────────┐ ┌─────────┐                            │
│ │Cam_01   │ │Cam_02   │ │Cam_03   │                            │
│ │Main St  │ │5th Ave  │ │Park Blvd│                            │
│ │[🟢Live] │ │[🟢Live] │ │[🔴Off]  │                            │
│ │94% AI   │ │89% AI   │ │-- AI    │                            │
│ └─────────┘ └─────────┘ └─────────┘                            │
│ [+ Click to expand] [🔧 Settings] [📊 Analytics]              │
└─────────────────────────────────────────────────────────────────┘
```

### Alerts Tab

```
┌─────────────────────────────────────────────────────────────────┐
│ Alert Management Controls                                       │
│ Filter: [All] [Critical] [Warning] [Info] | Status: [Active] [Resolved] │
│ Time Range: [Last Hour] [Last 4 Hours] [Today] [Custom]        │
├─────────────────────────────────────────────────────────────────┤
│ Alert Timeline (chronological list)                            │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ 🔴 CRITICAL | 14:23 | Camera_North_01                      │ │
│ │    Connection Lost - No response for 2 minutes             │ │
│ │    [Acknowledge] [View Camera] [Create Ticket]             │ │
│ └─────────────────────────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ 🟡 WARNING | 14:15 | Intersection_Main_5th                 │ │
│ │    Heavy Congestion Detected - Average speed 8 km/h        │ │
│ │    [View Location] [Traffic Plan] [Notify Traffic Control] │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Analytics Tab

```
┌─────────────────────────────────────────────────────────────────┐
│ Analytics Dashboard                                             │
│ Time Range: [Last Hour] [4 Hours] [Today] [Week] [Custom]      │
│ Export: [PDF] [Excel] [PNG]                                    │
├─────────────────────────────────────────────────────────────────┤
│ Key Metrics Summary (4-column grid)                            │
│ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐              │
│ │Peak Vol │ │Avg Speed│ │AI Acc   │ │Uptime   │              │
│ │2,847    │ │38 km/h  │ │94.2%    │ │99.8%    │              │
│ │vehicles │ │(15:30)  │ │(model)  │ │(24h)    │              │
│ └─────────┘ └─────────┘ └─────────┘ └─────────┘              │
├─────────────────────────────────────────────────────────────────┤
│ Visualization Area (2x2 grid)                                  │
│ ┌─────────────────────────┐ ┌─────────────────────────────────┐ │
│ │ Traffic Flow Heatmap    │ │ Hourly Volume Trends            │ │
│ │ [Interactive heat viz]  │ │ [Time series line chart]       │ │
│ └─────────────────────────┘ └─────────────────────────────────┘ │
│ ┌─────────────────────────┐ ┌─────────────────────────────────┐ │
│ │ Vehicle Classification  │ │ AI Performance Metrics          │ │
│ │ [Pie chart breakdown]   │ │ [Multi-line performance chart]  │ │
│ └─────────────────────────┘ └─────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### System Health Tab

```
┌─────────────────────────────────────────────────────────────────┐
│ System Performance Overview                                     │
│ Overall Health: 🟢 Excellent | Last Check: 30 seconds ago      │
├─────────────────────────────────────────────────────────────────┤
│ Performance Metrics Grid                                        │
│ ┌─────────────────────────┐ ┌─────────────────────────────────┐ │
│ │ Inference Performance   │ │ System Resources                │ │
│ │ • Avg Latency: 45ms     │ │ • CPU Usage: 67%               │ │
│ │ • 99th Percentile: 78ms │ │ • Memory: 8.2GB/16GB          │ │
│ │ • Throughput: 847 fps   │ │ • GPU: 73% utilization        │ │
│ │ • Queue Depth: 12       │ │ • Storage: 2.1TB/4TB          │ │
│ └─────────────────────────┘ └─────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│ Service Status Matrix                                           │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ API Gateway      🟢 Healthy    │ Camera Service  🟢 Healthy │ │
│ │ ML Inference     🟢 Healthy    │ Database       🟢 Healthy  │ │
│ │ Stream Processor 🟡 Warning    │ Redis Cache    🟢 Healthy  │ │
│ │ Alert System     🟢 Healthy    │ File Storage   🟢 Healthy  │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## Advanced Camera Management (`/cameras`)

### User Personas & Goals
- **Traffic Operations Manager**: Monitor camera health, control PTZ functions
- **Traffic Engineer**: Configure camera settings, analyze coverage
- **Technical Staff**: Maintain and troubleshoot camera systems

### Page Layout Structure

```
┌─────────────────────────────────────────────────────────────────┐
│ Header with Camera Summary                                      │
│ "Camera Management" | 48 Total | 45 Online | 3 Issues         │
│ Quick Actions: [Bulk Config] [Health Check] [Add Camera]       │
├─────────────────────────────────────────────────────────────────┤
│ Filter & Control Bar                                           │
│ View: [Grid] [List] [Map] | Filter: [All] [Online] [Offline]  │
│ Sort: [Name] [Location] [Status] [Last Ping] | Search: [____] │
├─────────────────────────────────────────────────────────────────┤
│ Camera Display Area (context-dependent)                        │
└─────────────────────────────────────────────────────────────────┘
```

### Grid View Layout

```
┌─────────────────────────────────────────────────────────────────┐
│ Camera Grid (responsive 2x2, 3x3, 4x4)                        │
│ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐    │
│ │ CAM_MAIN_01     │ │ CAM_MAIN_02     │ │ CAM_PARK_01     │    │
│ │ Main St & 5th   │ │ Main St & 3rd   │ │ Park Blvd & Oak │    │
│ │                 │ │                 │ │                 │    │
│ │ [Live Stream]   │ │ [Live Stream]   │ │ [Live Stream]   │    │
│ │                 │ │                 │ │                 │    │
│ │ 🟢 Online       │ │ 🟢 Online       │ │ 🔴 Offline      │    │
│ │ Signal: 94%     │ │ Signal: 87%     │ │ Last: 2h ago    │    │
│ │ AI: 96% conf    │ │ AI: 91% conf    │ │ AI: N/A         │    │
│ │                 │ │                 │ │                 │    │
│ │ [🎛️] [⚙️] [📊]   │ │ [🎛️] [⚙️] [📊]   │ │ [🔄] [⚙️] [🚨]   │    │
│ │ PTZ  Set  Stats │ │ PTZ  Set  Stats │ │ Retry Set Alert │    │
│ └─────────────────┘ └─────────────────┘ └─────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### List View Layout

```
┌─────────────────────────────────────────────────────────────────┐
│ Camera List Table                                               │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Name        │Location     │Status │Signal│AI Conf│Actions   │ │
│ ├─────────────────────────────────────────────────────────────┤ │
│ │ CAM_MAIN_01 │Main & 5th   │🟢 On  │94%   │96%    │[PTZ][⚙️]│ │
│ │ CAM_MAIN_02 │Main & 3rd   │🟢 On  │87%   │91%    │[PTZ][⚙️]│ │
│ │ CAM_PARK_01 │Park & Oak   │🔴 Off │--    │--     │[🔄][🚨] │ │
│ │ CAM_SIDE_01 │Side & 1st   │🟡 Warn│76%   │82%    │[PTZ][⚙️]│ │
│ └─────────────────────────────────────────────────────────────┘ │
│ [Previous] [1] [2] [3] [Next]                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Single Camera Detail Modal

```
┌─────────────────────────────────────────────────────────────────┐
│ Camera Detail: CAM_MAIN_01                              [✕]    │
├─────────────────────────────────────────────────────────────────┤
│ ┌─────────────────────────┐ ┌─────────────────────────────────┐ │
│ │ Live Stream Preview     │ │ Camera Information              │ │
│ │                         │ │ • Location: Main St & 5th Ave  │ │
│ │   [Live Video Feed]     │ │ • Model: Axis P5654-E          │ │
│ │                         │ │ • IP: 192.168.1.101           │ │
│ │ [🎛️PTZ Controls]        │ │ • Installed: 2023-08-15       │ │
│ │ ↖️ ⬆️ ↗️                    │ │ • Last Maintenance: 30 days   │ │
│ │ ⬅️ 🏠 ➡️                    │ │                                │ │
│ │ ↙️ ⬇️ ↘️                    │ │ Status Information             │ │
│ │ [🔍-] [🔍+]               │ │ • Online: 🟢 Yes              │ │
│ └─────────────────────────┘ │ • Signal Strength: 94%        │ │
│                             │ • Temperature: 42°C            │ │
│ Tabs: [Stream] [Settings] [Analytics] [Diagnostics]            │ │
├─────────────────────────────────────────────────────────────────┤ │
│ Tab Content Area                                                │ │
│ [Save Changes] [Run Diagnostics] [Reboot Camera] [Cancel]      │ │
└─────────────────────────────────────────────────────────────────┘
```

### Map View Layout

```
┌─────────────────────────────────────────────────────────────────┐
│ Interactive Map View                                            │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │                     [Interactive Map]                       │ │
│ │                                                             │ │
│ │  🟢 CAM_01    🟢 CAM_02         🔴 CAM_03                  │ │
│ │     │             │                 │                      │ │
│ │  [Main St]     [Park Ave]       [Side St]                 │ │
│ │                                                             │ │
│ │              🟡 CAM_04                                      │ │
│ │                 │                                           │ │
│ │             [Oak Blvd]                                      │ │
│ │                                                             │ │
│ │ Legend: 🟢 Online | 🔴 Offline | 🟡 Warning                │ │
│ └─────────────────────────────────────────────────────────────┘ │
│ Selected Camera Info Panel (bottom overlay)                    │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ CAM_MAIN_01 | Main St & 5th Ave | 🟢 Online | 94% Signal   │ │
│ │ [View Stream] [Configure] [Run Diagnostics] [Details]      │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## Analytics & Reporting (`/analytics`)

### User Personas & Goals
- **Traffic Engineer**: Detailed analysis, pattern identification, report generation
- **City Planner**: Long-term trends, capacity planning, ROI analysis
- **Management**: Executive reports, KPI dashboards

### Page Layout Structure

```
┌─────────────────────────────────────────────────────────────────┐
│ Analytics Header                                                │
│ "Traffic Analytics & Insights" | Export: [PDF] [Excel] [API]   │
│ Time Range: [Last Hour] [Today] [Week] [Month] [Custom Range]  │
├─────────────────────────────────────────────────────────────────┤
│ Quick Insights Bar (Key Metrics)                               │
│ Total Volume: 45,892 | Peak Hour: 5,247 (17:00) | Avg Speed: 38km/h │
├─────────────────────────────────────────────────────────────────┤
│ Report Categories Tabs                                          │
│ [Traffic Flow] [Performance] [Incidents] [Predictions] [Custom]│
├─────────────────────────────────────────────────────────────────┤
│ Dynamic Content Area                                            │
└─────────────────────────────────────────────────────────────────┘
```

### Traffic Flow Analysis Tab

```
┌─────────────────────────────────────────────────────────────────┐
│ Traffic Flow Dashboard                                          │
├─────────────────────────────────────────────────────────────────┤
│ Visualization Controls                                          │
│ Metric: [Volume] [Speed] [Density] [Occupancy]                 │
│ Granularity: [5min] [15min] [1hour] [1day]                     │
│ Locations: [All] [Main St] [Park Ave] [Side St] [Custom]       │
├─────────────────────────────────────────────────────────────────┤
│ Primary Chart Area                                              │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Traffic Volume Over Time                                    │ │
│ │ 6000 ┐                                                      │ │
│ │      │     Peak: 5247 @ 17:00                              │ │
│ │ 4000 │   /\                                                 │ │
│ │      │  /  \                                                │ │
│ │ 2000 │ /    \                                               │ │
│ │      │/      \                                              │ │
│ │    0 └────────────────────────────────────                 │ │
│ │      00:00  06:00  12:00  18:00  24:00                     │ │
│ └─────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│ Secondary Analysis (2-column grid)                             │
│ ┌─────────────────────────┐ ┌─────────────────────────────────┐ │
│ │ Vehicle Classification  │ │ Direction Analysis              │ │
│ │ 🚗 Cars: 78% (35,896)   │ │ Northbound: 52% (23,864)       │ │
│ │ 🚛 Trucks: 15% (6,884)  │ │ Southbound: 48% (22,028)       │ │
│ │ 🏍️ Motorcycles: 4% (1,836) │ │ Peak NB: 17:30 (2,847)      │ │
│ │ 🚌 Buses: 3% (1,376)    │ │ Peak SB: 08:15 (2,634)        │ │
│ └─────────────────────────┘ └─────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Performance Analysis Tab

```
┌─────────────────────────────────────────────────────────────────┐
│ System Performance Analytics                                    │
├─────────────────────────────────────────────────────────────────┤
│ Performance KPIs (4-column grid)                               │
│ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐              │
│ │Avg Lat  │ │AI Acc   │ │Uptime   │ │Throughput│              │
│ │45ms     │ │94.2%    │ │99.8%    │ │847 fps   │              │
│ │🟢 Exc   │ │🟢 High  │ │🟢 Exc   │ │🟢 High   │              │
│ └─────────┘ └─────────┘ └─────────┘ └─────────┘              │
├─────────────────────────────────────────────────────────────────┤
│ Performance Trends (2x2 grid)                                  │
│ ┌─────────────────────────┐ ┌─────────────────────────────────┐ │
│ │ Inference Latency       │ │ AI Model Accuracy               │ │
│ │ [Time series chart      │ │ [Time series with confidence    │ │
│ │  showing latency over   │ │  bands showing model accuracy   │ │
│ │  time with thresholds]  │ │  and confidence intervals]      │ │
│ └─────────────────────────┘ └─────────────────────────────────┘ │
│ ┌─────────────────────────┐ ┌─────────────────────────────────┐ │
│ │ System Resource Usage   │ │ Error Rate Analysis             │ │
│ │ [Multi-line chart with  │ │ [Stacked area chart showing     │ │
│ │  CPU, Memory, GPU usage │ │  different types of errors     │ │
│ │  over time]             │ │  and their frequency]           │ │
│ └─────────────────────────┘ └─────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│ Performance Issues Log                                          │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Time     │ Component      │ Issue                │ Duration  │ │
│ ├─────────────────────────────────────────────────────────────┤ │
│ │ 14:23:15 │ GPU Processor  │ High latency spike  │ 45s      │ │
│ │ 12:08:42 │ Camera_03      │ Connection timeout  │ 2m 15s   │ │
│ │ 09:45:33 │ ML Model       │ Accuracy drop       │ 8m 22s   │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Predictive Analytics Tab

```
┌─────────────────────────────────────────────────────────────────┐
│ AI-Powered Traffic Predictions                                  │
├─────────────────────────────────────────────────────────────────┤
│ Prediction Settings                                             │
│ Forecast Period: [Next Hour] [Next 4 Hours] [Next Day] [Week] │
│ Confidence Level: [80%] [90%] [95%] | Model: [LSTM] [Prophet]  │
├─────────────────────────────────────────────────────────────────┤
│ Traffic Volume Predictions                                      │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Predicted Traffic Volume - Next 4 Hours                    │ │
│ │ 3000 ┐                                                      │ │
│ │      │ Actual ────── Predicted ┅┅┅┅ Confidence Band ░░░   │ │
│ │ 2000 │      ┌─────┐                                         │ │
│ │      │     /       \     ┌┅┅┅┅┅┐                          │ │
│ │ 1000 │    /         \   ┅┅     ┅┅                          │ │
│ │      │   /           \┅┅         ┅┅                        │ │
│ │    0 └──────────────────────────────────                  │ │
│ │      Now   +1h    +2h    +3h    +4h                       │ │
│ └─────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│ Prediction Insights (2-column)                                 │
│ ┌─────────────────────────┐ ┌─────────────────────────────────┐ │
│ │ Upcoming Events         │ │ Recommended Actions             │ │
│ │ • Rush Hour (17:00)     │ │ • Pre-position traffic officers │ │
│ │   Expected: 2,400 veh   │ │   at Main St & 5th Ave         │ │
│ │   Confidence: 89%       │ │ • Adjust signal timing at      │ │
│ │                         │ │   Park Ave (16:45)             │ │
│ │ • School Dismissal      │ │ • Monitor Camera_04 - predicted │ │
│ │   (15:30) Expected:     │ │   signal degradation           │ │
│ │   +15% local traffic    │ │                                 │ │
│ │   Confidence: 76%       │ │ • Weather Alert: Rain expected │ │
│ │                         │ │   - Increase following distance │ │
│ │                         │ │   monitoring                    │ │
│ └─────────────────────────┘ └─────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## Settings & Configuration (`/settings`)

### User Personas & Goals
- **System Administrator**: User management, system configuration, security settings
- **Traffic Engineer**: Threshold configuration, alert rules, camera settings
- **Operations Manager**: Notification preferences, dashboard customization

### Page Layout Structure

```
┌─────────────────────────────────────────────────────────────────┐
│ Settings Header                                                 │
│ "System Configuration" | [Save Changes] [Reset] [Export Config]│
├─────────────────────────────────────────────────────────────────┤
│ Settings Navigation Sidebar                                     │
│ ┌─────────────────┐ ┌───────────────────────────────────────┐    │
│ │ Categories      │ │ Settings Content Area                 │    │
│ │ • General       │ │                                       │    │
│ │ • Notifications │ │ [Dynamic content based on selection]  │    │
│ │ • Thresholds    │ │                                       │    │
│ │ • AI & ML       │ │                                       │    │
│ │ • Security      │ │                                       │    │
│ │ • Users & Roles │ │                                       │    │
│ │ • System        │ │                                       │    │
│ │ • Backup        │ │                                       │    │
│ └─────────────────┘ └───────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### General Settings

```
┌─────────────────────────────────────────────────────────────────┐
│ General Configuration                                           │
├─────────────────────────────────────────────────────────────────┤
│ System Information                                              │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ System Name: [ITS Camera AI - Downtown]                    │ │
│ │ Timezone: [UTC-8 Pacific Time        ▼]                   │ │
│ │ Language: [English (US)               ▼]                   │ │
│ │ Date Format: [MM/DD/YYYY              ▼]                   │ │
│ │ Currency: [USD ($)                    ▼]                   │ │
│ └─────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│ Dashboard Preferences                                           │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Default View: [Overview               ▼]                   │ │
│ │ Refresh Rate: [5 seconds              ▼]                   │ │
│ │ Theme: [Auto] [Light] [Dark]                              │ │
│ │ Density: [Comfortable] [Compact] [Spacious]               │ │
│ │                                                             │ │
│ │ ☑️ Show performance indicators in header                    │ │
│ │ ☑️ Enable real-time notifications                          │ │
│ │ ☑️ Auto-refresh camera feeds                               │ │
│ │ ☐ Enable debug mode                                        │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Notification Settings

```
┌─────────────────────────────────────────────────────────────────┐
│ Notification Configuration                                      │
├─────────────────────────────────────────────────────────────────┤
│ Notification Channels                                           │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Email Notifications                                         │ │
│ │ ☑️ Critical Alerts    ✉️ admin@city.gov                     │ │
│ │ ☑️ System Warnings    ✉️ ops-team@city.gov                  │ │
│ │ ☐ Daily Reports      ✉️ management@city.gov                │ │
│ │                                                             │ │
│ │ SMS Notifications (Emergency Only)                          │ │
│ │ ☑️ Camera Failures    📱 +1-555-0123                        │ │
│ │ ☑️ Security Alerts    📱 +1-555-0456                        │ │
│ │                                                             │ │
│ │ Slack Integration                                           │ │
│ │ ☑️ Operations Channel  #traffic-ops                         │ │
│ │ ☑️ Alerts Channel     #traffic-alerts                       │ │
│ └─────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│ Alert Rules Configuration                                       │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Trigger Conditions                    Severity    Notify     │ │
│ ├─────────────────────────────────────────────────────────────┤ │
│ │ Camera offline > 2 minutes           Critical    ✉️ 📱 💬    │ │
│ │ AI accuracy < 85%                    Warning     ✉️ 💬       │ │
│ │ Inference latency > 100ms            Warning     ✉️ 💬       │ │
│ │ Heavy congestion > 10 minutes        Info        ✉️          │ │
│ │ Security breach detected             Emergency   ✉️ 📱 💬    │ │
│ │ [+ Add Custom Rule]                                         │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### AI & ML Configuration

```
┌─────────────────────────────────────────────────────────────────┐
│ AI & Machine Learning Settings                                  │
├─────────────────────────────────────────────────────────────────┤
│ Model Configuration                                             │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Active Model: YOLO11n-traffic-v2.1    [Change Model]       │ │
│ │ Inference Device: [GPU 0 - RTX 4090    ▼]                  │ │
│ │ Batch Size: [16                        ▼]                  │ │
│ │ Confidence Threshold: [0.25] ████████░░ (0.85)             │ │
│ │ IoU Threshold: [0.45] ████████████░ (0.45)                │ │
│ │                                                             │ │
│ │ Performance Settings                                        │ │
│ │ Max Latency Target: [100ms             ▼]                  │ │
│ │ ☑️ Enable FP16 optimization                                │ │
│ │ ☑️ Use TensorRT acceleration                               │ │
│ │ ☑️ Auto-batch optimization                                 │ │
│ └─────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│ Detection Classes                                               │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Enabled Detection Classes:                                  │ │
│ │ ☑️ Person         ☑️ Car          ☑️ Truck        ☑️ Bus     │ │
│ │ ☑️ Motorcycle     ☐ Bicycle      ☐ Train         ☐ Boat    │ │
│ │                                                             │ │
│ │ Vehicle Classification Settings:                            │ │
│ │ Minimum Size (pixels): [20 x 20]                          │ │
│ │ Maximum Size (pixels): [800 x 600]                        │ │
│ │ ☑️ Enable sub-class detection (sedan, SUV, pickup)        │ │
│ └─────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│ Auto-Learning Settings                                          │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ ☑️ Enable continuous learning                              │ │
│ │ ☑️ Auto-retrain with new data (weekly)                     │ │
│ │ ☐ Share anonymous data for model improvement               │ │
│ │                                                             │ │
│ │ Training Data Retention: [30 days      ▼]                  │ │
│ │ Model Backup Frequency: [Daily         ▼]                  │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## Security Dashboard (`/security`)

### User Personas & Goals
- **Security Administrator**: Monitor security status, manage access controls
- **System Administrator**: Audit trails, compliance reporting
- **Operations Manager**: Security alerts, incident response

### Page Layout Structure

```
┌─────────────────────────────────────────────────────────────────┐
│ Security Dashboard Header                                       │
│ "System Security Overview" | Status: 🟢 Secure | Last Audit: 2h│
│ [Security Scan] [Generate Report] [Incident Response]          │
├─────────────────────────────────────────────────────────────────┤
│ Security Status Grid (4-column)                                │
│ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐              │
│ │Overall  │ │Access   │ │Data     │ │Network  │              │
│ │🟢 Secure│ │🟢 Clean │ │🟢 Encrypted│ │🟡 Monitor│            │
│ │98% Score│ │0 Threats│ │AES-256  │ │3 Alerts │              │
│ └─────────┘ └─────────┘ └─────────┘ └─────────┘              │
├─────────────────────────────────────────────────────────────────┤
│ Security Sections Tabs                                          │
│ [Overview] [Access Control] [Audit Log] [Compliance] [Threats] │
├─────────────────────────────────────────────────────────────────┤
│ Dynamic Content Area                                            │
└─────────────────────────────────────────────────────────────────┘
```

### Security Overview Tab

```
┌─────────────────────────────────────────────────────────────────┐
│ Security Overview                                               │
├─────────────────────────────────────────────────────────────────┤
│ Active Security Alerts (if any)                                │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ 🟡 MONITORING | Network Traffic Anomaly Detected           │ │
│ │    Unusual API call pattern from IP 203.0.113.42          │ │
│ │    Time: 14:23 | Severity: Medium | [Investigate] [Block]  │ │
│ └─────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│ Security Metrics Dashboard (2x2 grid)                          │
│ ┌─────────────────────────┐ ┌─────────────────────────────────┐ │
│ │ Authentication Activity │ │ Data Encryption Status          │ │
│ │ • Successful: 1,247     │ │ • Video Streams: ✅ Encrypted   │ │
│ │ • Failed: 3             │ │ • Database: ✅ Encrypted        │ │
│ │ • Locked: 0             │ │ • API Traffic: ✅ TLS 1.3       │ │
│ │ • Success Rate: 99.8%   │ │ • File Storage: ✅ AES-256     │ │
│ └─────────────────────────┘ └─────────────────────────────────┘ │
│ ┌─────────────────────────┐ ┌─────────────────────────────────┐ │
│ │ Network Security        │ │ Compliance Status               │ │
│ │ • Firewall: ✅ Active   │ │ • GDPR: ✅ Compliant           │ │
│ │ • DDoS Protection: ✅    │ │ • SOC2: ✅ Certified           │ │
│ │ • VPN Status: ✅ Secure │ │ • PCI DSS: ✅ Level 1          │ │
│ │ • Intrusion: 3 Blocked  │ │ • Last Audit: ✅ Passed       │ │
│ └─────────────────────────┘ └─────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│ Recent Security Events                                          │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Time     │ Event Type        │ Source          │ Action      │ │
│ ├─────────────────────────────────────────────────────────────┤ │
│ │ 14:23:15 │ Anomaly Detection │ API Gateway     │ Monitoring  │ │
│ │ 12:45:33 │ Login Success     │ User: j.smith   │ Allowed     │ │
│ │ 11:22:18 │ Failed Login      │ IP: 198.51.100.5│ Blocked     │ │
│ │ 09:15:42 │ Certificate Renewal│ SSL/TLS        │ Completed   │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Access Control Tab

```
┌─────────────────────────────────────────────────────────────────┐
│ Access Control Management                                       │
├─────────────────────────────────────────────────────────────────┤
│ User Sessions & Access                                          │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Active Sessions: 12 | Expired: 3 | Failed Logins: 1        │ │
│ │ [View All Sessions] [Revoke All] [Export Log]              │ │
│ └─────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│ Role-Based Access Matrix                                        │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Role          │Dashboard│Cameras│Analytics│Settings│Admin    │ │
│ ├─────────────────────────────────────────────────────────────┤ │
│ │ Traffic Ops   │ ✅ Full │✅ Full│ ✅ View │ ❌ None│ ❌ None │ │
│ │ Engineer      │ ✅ Full │✅ Full│ ✅ Full │ ✅ Limited│❌ None │ │
│ │ Manager       │ ✅ Full │✅ View│ ✅ Full │ ✅ View │ ❌ None │ │
│ │ Administrator │ ✅ Full │✅ Full│ ✅ Full │ ✅ Full │ ✅ Full │ │
│ │ [+ Add Role]  │         │       │         │        │         │ │
│ └─────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│ API Access Control                                              │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ API Keys: 8 Active | Rate Limits: ✅ Enforced              │ │
│ │                                                             │ │
│ │ Key Name        │ Permissions  │ Rate Limit │ Last Used    │ │
│ ├─────────────────────────────────────────────────────────────┤ │
│ │ Mobile App      │ Read Only    │ 1000/min   │ 2 min ago    │ │
│ │ Dashboard API   │ Full Access  │ 5000/min   │ 30s ago      │ │
│ │ Analytics Tool  │ Analytics    │ 500/min    │ 1 hour ago   │ │
│ │ [+ Generate Key]│              │            │              │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## Mobile Design Patterns

### Key Principles for Mobile Experience

1. **Touch-First Interface Design**
   - Minimum 44px touch targets
   - Generous spacing between interactive elements
   - Swipe gestures for navigation and actions

2. **Progressive Disclosure**
   - Essential information first
   - Collapsible sections for detailed data
   - Contextual actions on demand

3. **Offline Capability**
   - Cached data for offline viewing
   - Clear indicators for connection status
   - Sync status and conflict resolution

### Mobile Dashboard Layout

```
┌─────────────────────────┐
│ [☰] ITS AI    [🔔] [👤] │ Header (56px)
├─────────────────────────┤
│ System Status           │
│ 🟢 All Systems Healthy  │
│ Latency: 45ms | AI: 94% │
├─────────────────────────┤
│ Quick Stats (2x2 grid)  │
│ ┌─────────┐ ┌─────────┐ │
│ │Vehicles │ │Cameras  │ │
│ │1,284    │ │48/52    │ │
│ │+12%     │ │92% up   │ │
│ └─────────┘ └─────────┘ │
│ ┌─────────┐ ┌─────────┐ │
│ │Alerts   │ │Speed    │ │
│ │7 Active │ │42 km/h  │ │
│ │3 Crit   │ │-5 km/h  │ │
│ └─────────┘ └─────────┘ │
├─────────────────────────┤
│ Camera Quick View       │
│ ┌─────────────────────┐ │
│ │   [Live Stream]     │ │
│ │   CAM_MAIN_01       │ │
│ │   🟢 Online         │ │
│ └─────────────────────┘ │
│ [< Prev] [1/12] [Next >]│
├─────────────────────────┤
│ Recent Alerts           │
│ 🔴 CAM_03 - Offline     │
│    2 minutes ago        │
│ 🟡 Heavy traffic - M&5  │
│    15 minutes ago       │
│                         │
│ [View All Alerts]       │
├─────────────────────────┤
│ Bottom Navigation       │
│ [🏠] [📹] [⚠️] [📊] [⚙️] │
│ Home Cams Alert Chart Set│
└─────────────────────────┘
```

### Mobile Camera Management

```
┌─────────────────────────┐
│ [←] Cameras    [🔍] [⋮] │ Header with search
├─────────────────────────┤
│ Filter Bar (horizontal) │
│ [All] [Online] [Offline] │
├─────────────────────────┤
│ Camera List (cards)     │
│ ┌─────────────────────┐ │
│ │ CAM_MAIN_01         │ │
│ │ Main St & 5th Ave   │ │
│ │ 🟢 Online | 94% sig │ │
│ │ [📹 View] [⚙️ Config]│ │
│ └─────────────────────┘ │
│                         │
│ ┌─────────────────────┐ │
│ │ CAM_MAIN_02         │ │
│ │ Main St & 3rd Ave   │ │
│ │ 🟢 Online | 87% sig │ │
│ │ [📹 View] [⚙️ Config]│ │
│ └─────────────────────┘ │
│                         │
│ ┌─────────────────────┐ │
│ │ CAM_PARK_01         │ │
│ │ Park Blvd & Oak St  │ │
│ │ 🔴 Offline | 2h ago │ │
│ │ [🔄 Retry] [🚨 Alert]│ │
│ └─────────────────────┘ │
└─────────────────────────┘
```

### Mobile Alert Management

```
┌─────────────────────────┐
│ [←] Alerts        [🔔✓] │ Header with mark all
├─────────────────────────┤
│ Alert Filters (chips)   │
│ [All] [Critical] [New]  │
├─────────────────────────┤
│ Alert Timeline          │
│ ┌─────────────────────┐ │
│ │ 🔴 CRITICAL   2 min │ │
│ │ CAM_NORTH_01        │ │
│ │ Connection Lost     │ │
│ │                     │ │
│ │ [Acknowledge]       │ │
│ │ [View Camera]       │ │
│ └─────────────────────┘ │
│                         │
│ ┌─────────────────────┐ │
│ │ 🟡 WARNING   15 min │ │
│ │ Main St & 5th Ave   │ │
│ │ Heavy Congestion    │ │
│ │                     │ │
│ │ [View Location]     │ │
│ │ [Traffic Plan]      │ │
│ └─────────────────────┘ │
│                         │
│ Pull to refresh...      │
└─────────────────────────┘
```

This comprehensive design system provides the foundation for creating a professional, accessible, and user-friendly traffic monitoring interface that serves all stakeholder personas while highlighting the system's advanced AI capabilities and enterprise-grade security features.