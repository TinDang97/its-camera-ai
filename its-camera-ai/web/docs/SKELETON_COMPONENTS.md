# Skeleton Components Documentation

## Overview

The ITS Camera AI dashboard implements a comprehensive skeleton loading system to provide smooth user experiences during data loading states. These components maintain layout stability and provide visual feedback while content is being fetched.

## Core Skeleton Architecture

### Base Skeleton Component

```tsx
// components/ui/skeleton/Skeleton.tsx
'use client';

import { memo } from 'react';
import { cn } from '@/lib/utils';

interface SkeletonProps {
  className?: string;
  variant?: 'default' | 'circular' | 'rounded' | 'text';
  animation?: 'pulse' | 'wave' | 'none';
  width?: string | number;
  height?: string | number;
  children?: React.ReactNode;
}

const SKELETON_VARIANTS = {
  default: 'rounded',
  circular: 'rounded-full',
  rounded: 'rounded-md',
  text: 'rounded-sm',
} as const;

const SKELETON_ANIMATIONS = {
  pulse: 'animate-pulse',
  wave: 'animate-shimmer',
  none: '',
} as const;

export const Skeleton = memo<SkeletonProps>(({
  className,
  variant = 'default',
  animation = 'pulse',
  width,
  height,
  children,
  ...props
}) => {
  const style = {
    width: typeof width === 'number' ? `${width}px` : width,
    height: typeof height === 'number' ? `${height}px` : height,
  };

  return (
    <div
      className={cn(
        'bg-gray-200 dark:bg-gray-800',
        SKELETON_VARIANTS[variant],
        SKELETON_ANIMATIONS[animation],
        className
      )}
      style={style}
      aria-hidden="true"
      {...props}
    >
      {children}
    </div>
  );
});

Skeleton.displayName = 'Skeleton';
```

### Shimmer Animation CSS

```css
/* globals.css - Add shimmer animation */
@keyframes shimmer {
  0% {
    background-position: -200px 0;
  }
  100% {
    background-position: calc(200px + 100%) 0;
  }
}

.animate-shimmer {
  background: linear-gradient(90deg, #f0f0f0 0px, #e0e0e0 40px, #f0f0f0 80px);
  background-size: 200px;
  animation: shimmer 1.5s infinite;
}

.dark .animate-shimmer {
  background: linear-gradient(90deg, #374151 0px, #4b5563 40px, #374151 80px);
  background-size: 200px;
}
```

## ITS-Specific Skeleton Components

### 1. Alert Panel Skeleton

```tsx
// components/ui/skeleton/AlertPanelSkeleton.tsx
'use client';

import { memo } from 'react';
import { Skeleton } from './Skeleton';
import { Card, CardContent, CardHeader } from '@/components/ui/card';

export const AlertPanelSkeleton = memo(() => (
  <Card className="h-full">
    <CardHeader className="pb-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Skeleton variant="circular" width={20} height={20} />
          <Skeleton width={120} height={20} />
        </div>
        <Skeleton width={60} height={24} variant="rounded" />
      </div>
    </CardHeader>
    
    <CardContent className="p-0">
      <div className="space-y-3 p-4 pt-0">
        {Array.from({ length: 4 }, (_, i) => (
          <AlertItemSkeleton key={i} variant={i % 2 === 0 ? 'critical' : 'warning'} />
        ))}
      </div>
    </CardContent>
  </Card>
));

const AlertItemSkeleton = memo<{ variant: 'critical' | 'warning' | 'info' }>(({ variant }) => {
  const bgColor = {
    critical: 'bg-red-50 border-red-200',
    warning: 'bg-yellow-50 border-yellow-200',
    info: 'bg-blue-50 border-blue-200',
  }[variant];

  return (
    <div className={`p-4 rounded-lg border ${bgColor}`}>
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-start gap-3 flex-1 min-w-0">
          <div className="flex items-center gap-1 mt-0.5">
            <Skeleton variant="circular" width={16} height={16} />
            <Skeleton variant="circular" width={12} height={12} />
          </div>

          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1">
              <Skeleton width={180} height={16} />
              <Skeleton width={40} height={20} variant="rounded" />
            </div>

            <Skeleton width="100%" height={12} className="mb-2" />
            <Skeleton width="80%" height={12} className="mb-2" />

            <div className="flex items-center gap-3">
              <div className="flex items-center gap-1">
                <Skeleton variant="circular" width={12} height={12} />
                <Skeleton width={100} height={12} />
              </div>
              <div className="flex items-center gap-1">
                <Skeleton variant="circular" width={12} height={12} />
                <Skeleton width={60} height={12} />
              </div>
            </div>
          </div>
        </div>

        <Skeleton width={50} height={20} variant="rounded" />
      </div>
    </div>
  );
});

AlertPanelSkeleton.displayName = 'AlertPanelSkeleton';
AlertItemSkeleton.displayName = 'AlertItemSkeleton';
```

### 2. Camera Grid Skeleton

```tsx
// components/ui/skeleton/CameraGridSkeleton.tsx
'use client';

import { memo } from 'react';
import { Skeleton } from './Skeleton';
import { Card, CardContent, CardHeader } from '@/components/ui/card';

export const CameraGridSkeleton = memo<{ count?: number }>(({ count = 6 }) => (
  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
    {Array.from({ length: count }, (_, i) => (
      <CameraCardSkeleton key={i} />
    ))}
  </div>
));

const CameraCardSkeleton = memo(() => (
  <Card>
    <CardHeader className="pb-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Skeleton variant="circular" width={16} height={16} />
          <Skeleton width={100} height={16} />
        </div>
        <Skeleton width={50} height={20} variant="rounded" />
      </div>
    </CardHeader>
    
    <CardContent>
      {/* Video placeholder */}
      <div className="aspect-video bg-gray-900 rounded-lg flex items-center justify-center mb-4">
        <div className="text-center text-gray-500">
          <Skeleton variant="circular" width={48} height={48} className="mx-auto mb-2" />
          <Skeleton width={80} height={12} className="mx-auto" />
        </div>
      </div>

      {/* Camera info */}
      <div className="space-y-2">
        <div className="flex justify-between">
          <Skeleton width={60} height={12} />
          <Skeleton width={40} height={12} />
        </div>
        <div className="flex justify-between">
          <Skeleton width={80} height={12} />
          <Skeleton width={60} height={12} />
        </div>
        <div className="flex justify-between">
          <Skeleton width={70} height={12} />
          <Skeleton width={50} height={12} />
        </div>
      </div>

      {/* Action buttons */}
      <div className="flex gap-2 mt-4">
        <Skeleton width="100%" height={32} variant="rounded" />
        <Skeleton width="100%" height={32} variant="rounded" />
      </div>
    </CardContent>
  </Card>
));

CameraGridSkeleton.displayName = 'CameraGridSkeleton';
CameraCardSkeleton.displayName = 'CameraCardSkeleton';
```

### 3. Dashboard Metrics Skeleton

```tsx
// components/ui/skeleton/DashboardMetricsSkeleton.tsx
'use client';

import { memo } from 'react';
import { Skeleton } from './Skeleton';
import { Card, CardContent, CardHeader } from '@/components/ui/card';

export const DashboardMetricsSkeleton = memo(() => (
  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
    {Array.from({ length: 4 }, (_, i) => (
      <MetricCardSkeleton key={i} />
    ))}
  </div>
));

const MetricCardSkeleton = memo(() => (
  <Card>
    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
      <Skeleton width={100} height={14} />
      <Skeleton variant="circular" width={20} height={20} />
    </CardHeader>
    <CardContent>
      <Skeleton width={80} height={32} className="mb-2" />
      <div className="flex items-center gap-1">
        <Skeleton variant="circular" width={12} height={12} />
        <Skeleton width={120} height={12} />
      </div>
    </CardContent>
  </Card>
));

DashboardMetricsSkeleton.displayName = 'DashboardMetricsSkeleton';
MetricCardSkeleton.displayName = 'MetricCardSkeleton';
```

### 4. Traffic Flow Chart Skeleton

```tsx
// components/ui/skeleton/TrafficFlowChartSkeleton.tsx
'use client';

import { memo } from 'react';
import { Skeleton } from './Skeleton';
import { Card, CardContent, CardHeader } from '@/components/ui/card';

export const TrafficFlowChartSkeleton = memo(() => (
  <Card className="col-span-full">
    <CardHeader>
      <div className="flex items-center justify-between">
        <div>
          <Skeleton width={150} height={20} className="mb-2" />
          <Skeleton width={200} height={14} />
        </div>
        <div className="flex gap-2">
          <Skeleton width={80} height={32} variant="rounded" />
          <Skeleton width={80} height={32} variant="rounded" />
        </div>
      </div>
    </CardHeader>
    
    <CardContent>
      <div className="h-80 flex items-end justify-between px-4">
        {/* Chart bars */}
        {Array.from({ length: 12 }, (_, i) => (
          <div key={i} className="flex flex-col items-center gap-2">
            <Skeleton 
              width={20} 
              height={Math.random() * 200 + 50} 
              variant="rounded"
              animation="wave"
            />
            <Skeleton width={24} height={12} />
          </div>
        ))}
      </div>

      {/* Legend */}
      <div className="flex justify-center gap-6 mt-6">
        {Array.from({ length: 3 }, (_, i) => (
          <div key={i} className="flex items-center gap-2">
            <Skeleton variant="circular" width={12} height={12} />
            <Skeleton width={60} height={12} />
          </div>
        ))}
      </div>
    </CardContent>
  </Card>
));

TrafficFlowChartSkeleton.displayName = 'TrafficFlowChartSkeleton';
```

### 5. Table Skeleton

```tsx
// components/ui/skeleton/TableSkeleton.tsx
'use client';

import { memo } from 'react';
import { Skeleton } from './Skeleton';

interface TableSkeletonProps {
  rows?: number;
  columns?: number;
  showHeader?: boolean;
  showActions?: boolean;
}

export const TableSkeleton = memo<TableSkeletonProps>(({
  rows = 5,
  columns = 4,
  showHeader = true,
  showActions = true,
}) => (
  <div className="space-y-4">
    {/* Table header */}
    {showHeader && (
      <div className="flex items-center justify-between">
        <Skeleton width={200} height={24} />
        <div className="flex gap-2">
          <Skeleton width={80} height={32} variant="rounded" />
          <Skeleton width={80} height={32} variant="rounded" />
        </div>
      </div>
    )}

    {/* Table */}
    <div className="border rounded-lg overflow-hidden">
      {/* Table header row */}
      <div className="border-b bg-gray-50 p-4">
        <div className="grid gap-4" style={{ gridTemplateColumns: `repeat(${columns}, 1fr)${showActions ? ' auto' : ''}` }}>
          {Array.from({ length: columns }, (_, i) => (
            <Skeleton key={i} width={100} height={16} />
          ))}
          {showActions && <Skeleton width={60} height={16} />}
        </div>
      </div>

      {/* Table rows */}
      {Array.from({ length: rows }, (_, i) => (
        <div key={i} className="border-b last:border-b-0 p-4">
          <div className="grid gap-4" style={{ gridTemplateColumns: `repeat(${columns}, 1fr)${showActions ? ' auto' : ''}` }}>
            {Array.from({ length: columns }, (_, j) => (
              <Skeleton 
                key={j} 
                width={j === 0 ? 150 : Math.random() > 0.5 ? 100 : 80} 
                height={16} 
              />
            ))}
            {showActions && (
              <div className="flex gap-1">
                <Skeleton variant="circular" width={24} height={24} />
                <Skeleton variant="circular" width={24} height={24} />
              </div>
            )}
          </div>
        </div>
      ))}
    </div>
  </div>
));

TableSkeleton.displayName = 'TableSkeleton';
```

## Advanced Skeleton Patterns

### 1. Progressive Loading Skeleton

```tsx
// components/ui/skeleton/ProgressiveSkeleton.tsx
'use client';

import { memo, useEffect, useState } from 'react';
import { Skeleton } from './Skeleton';

interface ProgressiveSkeletonProps {
  stages: {
    component: React.ReactNode;
    delay: number;
  }[];
  finalContent?: React.ReactNode;
}

export const ProgressiveSkeleton = memo<ProgressiveSkeletonProps>(({
  stages,
  finalContent,
}) => {
  const [currentStage, setCurrentStage] = useState(0);
  const [isComplete, setIsComplete] = useState(false);

  useEffect(() => {
    if (currentStage < stages.length - 1) {
      const timer = setTimeout(() => {
        setCurrentStage(currentStage + 1);
      }, stages[currentStage].delay);

      return () => clearTimeout(timer);
    } else if (finalContent) {
      const timer = setTimeout(() => {
        setIsComplete(true);
      }, stages[currentStage]?.delay || 1000);

      return () => clearTimeout(timer);
    }
  }, [currentStage, stages, finalContent]);

  if (isComplete && finalContent) {
    return <>{finalContent}</>;
  }

  return <>{stages[currentStage]?.component}</>;
});

ProgressiveSkeleton.displayName = 'ProgressiveSkeleton';
```

### 2. Contextual Skeleton Hook

```tsx
// hooks/useSkeleton.ts
'use client';

import { useState, useEffect } from 'react';

interface UseSkeletonOptions {
  delay?: number;
  minDuration?: number;
  showSkeleton?: boolean;
}

export function useSkeleton(
  isLoading: boolean,
  options: UseSkeletonOptions = {}
) {
  const { delay = 0, minDuration = 500, showSkeleton: forceShow = false } = options;
  const [showSkeleton, setShowSkeleton] = useState(forceShow);
  const [loadingStartTime, setLoadingStartTime] = useState<number | null>(null);

  useEffect(() => {
    let delayTimer: NodeJS.Timeout;
    let minDurationTimer: NodeJS.Timeout;

    if (isLoading) {
      setLoadingStartTime(Date.now());
      
      if (delay > 0) {
        delayTimer = setTimeout(() => {
          setShowSkeleton(true);
        }, delay);
      } else {
        setShowSkeleton(true);
      }
    } else {
      const elapsedTime = loadingStartTime ? Date.now() - loadingStartTime : 0;
      const remainingTime = Math.max(0, minDuration - elapsedTime);

      if (remainingTime > 0) {
        minDurationTimer = setTimeout(() => {
          setShowSkeleton(false);
        }, remainingTime);
      } else {
        setShowSkeleton(false);
      }
    }

    return () => {
      clearTimeout(delayTimer);
      clearTimeout(minDurationTimer);
    };
  }, [isLoading, delay, minDuration, loadingStartTime]);

  return showSkeleton || forceShow;
}
```

### 3. Skeleton Factory

```tsx
// lib/skeleton-factory.ts
'use client';

import { ReactNode } from 'react';
import { Skeleton } from '@/components/ui/skeleton/Skeleton';

export class SkeletonFactory {
  static text(width?: string | number, lines: number = 1): ReactNode {
    if (lines === 1) {
      return <Skeleton variant="text" width={width} height={16} />;
    }

    return (
      <div className="space-y-2">
        {Array.from({ length: lines }, (_, i) => (
          <Skeleton
            key={i}
            variant="text"
            width={i === lines - 1 ? '80%' : '100%'}
            height={16}
          />
        ))}
      </div>
    );
  }

  static avatar(size: number = 40): ReactNode {
    return <Skeleton variant="circular" width={size} height={size} />;
  }

  static button(width: string | number = 100, height: number = 32): ReactNode {
    return <Skeleton variant="rounded" width={width} height={height} />;
  }

  static card(options: {
    showHeader?: boolean;
    contentLines?: number;
    showActions?: boolean;
  } = {}): ReactNode {
    const { showHeader = true, contentLines = 3, showActions = false } = options;

    return (
      <div className="p-4 border rounded-lg space-y-4">
        {showHeader && (
          <div className="flex items-center justify-between">
            <Skeleton width={150} height={20} />
            <Skeleton variant="circular" width={24} height={24} />
          </div>
        )}
        
        <div className="space-y-2">
          {Array.from({ length: contentLines }, (_, i) => (
            <Skeleton
              key={i}
              width={i === contentLines - 1 ? '75%' : '100%'}
              height={16}
            />
          ))}
        </div>

        {showActions && (
          <div className="flex gap-2">
            <Skeleton width={80} height={32} variant="rounded" />
            <Skeleton width={80} height={32} variant="rounded" />
          </div>
        )}
      </div>
    );
  }

  static list(items: number = 5, showAvatar: boolean = true): ReactNode {
    return (
      <div className="space-y-3">
        {Array.from({ length: items }, (_, i) => (
          <div key={i} className="flex items-center gap-3">
            {showAvatar && <Skeleton variant="circular" width={40} height={40} />}
            <div className="flex-1 space-y-1">
              <Skeleton width="60%" height={16} />
              <Skeleton width="40%" height={14} />
            </div>
          </div>
        ))}
      </div>
    );
  }
}
```

## Usage Examples

### Basic Implementation

```tsx
// components/AlertPanel.tsx
import { AlertPanelSkeleton } from '@/components/ui/skeleton/AlertPanelSkeleton';
import { useSkeleton } from '@/hooks/useSkeleton';

export function AlertPanel() {
  const { data: alerts, isLoading } = useAlerts();
  const showSkeleton = useSkeleton(isLoading, { delay: 200, minDuration: 500 });

  if (showSkeleton) {
    return <AlertPanelSkeleton />;
  }

  return (
    <div className="space-y-4">
      {alerts.map(alert => (
        <AlertItem key={alert.id} alert={alert} />
      ))}
    </div>
  );
}
```

### Progressive Loading Example

```tsx
// components/Dashboard.tsx
import { ProgressiveSkeleton } from '@/components/ui/skeleton/ProgressiveSkeleton';
import { DashboardMetricsSkeleton } from '@/components/ui/skeleton/DashboardMetricsSkeleton';

export function Dashboard() {
  const { data, isLoading } = useDashboardData();

  if (isLoading) {
    return (
      <ProgressiveSkeleton
        stages={[
          {
            component: <DashboardMetricsSkeleton />,
            delay: 300,
          },
          {
            component: (
              <>
                <DashboardMetricsSkeleton />
                <AlertPanelSkeleton />
              </>
            ),
            delay: 600,
          },
          {
            component: (
              <>
                <DashboardMetricsSkeleton />
                <AlertPanelSkeleton />
                <TrafficFlowChartSkeleton />
              </>
            ),
            delay: 900,
          },
        ]}
        finalContent={<DashboardContent data={data} />}
      />
    );
  }

  return <DashboardContent data={data} />;
}
```

### Conditional Skeletons

```tsx
// components/CameraGrid.tsx
export function CameraGrid() {
  const { data: cameras, isLoading, error } = useCameras();
  
  if (error) {
    return <ErrorState error={error} />;
  }

  if (isLoading) {
    return <CameraGridSkeleton count={6} />;
  }

  if (!cameras.length) {
    return <EmptyState message="No cameras found" />;
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      {cameras.map(camera => (
        <CameraCard key={camera.id} camera={camera} />
      ))}
    </div>
  );
}
```

## Best Practices

### 1. Performance Optimization
- Use `React.memo` for all skeleton components
- Implement proper cleanup for timers
- Avoid unnecessary re-renders during loading states
- Use CSS animations instead of JavaScript when possible

### 2. Accessibility
- Include `aria-hidden="true"` on skeleton elements
- Provide loading announcements for screen readers
- Maintain focus management during transitions

### 3. User Experience
- Match skeleton dimensions to actual content
- Use appropriate animation speeds (1-2 seconds)
- Implement minimum loading duration for stability
- Progressive loading for complex layouts

### 4. Design Guidelines
- Maintain consistent spacing and proportions
- Use subtle animations that don't distract
- Match border radius and styling of actual components
- Consider dark mode variations

This comprehensive skeleton system ensures smooth loading experiences across all components of the ITS Camera AI dashboard while maintaining performance and accessibility standards.