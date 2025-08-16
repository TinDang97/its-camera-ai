/**
 * Dashboard Widget Base Component
 *
 * Reusable base widget component for analytics dashboard with
 * standardized layout, loading states, and configuration options.
 */

'use client';

import React, { Suspense, useOptimistic, useTransition } from 'react';
import { cn } from '@/lib/utils';
import { IconLoader2, IconSettings, IconMaximize2, IconMinimize2, IconX } from '@tabler/icons-react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Skeleton } from '@/components/ui/skeleton';

export interface WidgetConfig {
  id: string;
  title: string;
  subtitle?: string;
  size: 'small' | 'medium' | 'large' | 'xlarge';
  refreshInterval?: number; // milliseconds
  showHeader?: boolean;
  showControls?: boolean;
  allowResize?: boolean;
  allowClose?: boolean;
  customActions?: WidgetAction[];
  priority: 'low' | 'medium' | 'high' | 'critical';
  category: 'traffic' | 'cameras' | 'system' | 'incidents' | 'analytics';
}

export interface WidgetAction {
  id: string;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
  onClick: () => void;
}

export interface WidgetData {
  timestamp: number;
  isLoading: boolean;
  error?: string;
  stale?: boolean;
  data: any;
}

export interface DashboardWidgetProps {
  config: WidgetConfig;
  data?: WidgetData;
  className?: string;
  children: React.ReactNode;
  onConfigChange?: (config: Partial<WidgetConfig>) => void;
  onRefresh?: () => void;
  onResize?: (size: WidgetConfig['size']) => void;
  onClose?: () => void;
  ErrorBoundary?: React.ComponentType<{ children: React.ReactNode }>;
}

interface OptimisticUpdate {
  action: 'loading' | 'error' | 'success' | 'config';
  payload?: any;
}

// Widget size configurations
const sizeClasses = {
  small: 'col-span-1 row-span-1 min-h-[200px]',
  medium: 'col-span-2 row-span-1 min-h-[250px]',
  large: 'col-span-2 row-span-2 min-h-[400px]',
  xlarge: 'col-span-3 row-span-2 min-h-[500px]',
};

// Priority indicator colors
const priorityColors = {
  low: 'border-l-muted-foreground',
  medium: 'border-l-primary',
  high: 'border-l-warning',
  critical: 'border-l-destructive',
};

// Widget loading skeleton
const WidgetSkeleton: React.FC<{ size: WidgetConfig['size'] }> = ({ size }) => {
  const height = {
    small: 150,
    medium: 200,
    large: 350,
    xlarge: 450,
  }[size];

  return (
    <div className="space-y-3">
      <div className="flex items-center space-x-2">
        <Skeleton className="h-4 w-24" />
        <Skeleton className="h-3 w-16" />
      </div>
      <Skeleton className={`w-full h-[${height}px]`} />
      <div className="flex space-x-2">
        <Skeleton className="h-3 w-12" />
        <Skeleton className="h-3 w-16" />
        <Skeleton className="h-3 w-10" />
      </div>
    </div>
  );
};

export const DashboardWidget: React.FC<DashboardWidgetProps> = ({
  config,
  data,
  className,
  children,
  onConfigChange,
  onRefresh,
  onResize,
  onClose,
  ErrorBoundary,
}) => {
  const [isPending, startTransition] = useTransition();

  // Optimistic updates for smooth UI interactions
  const [optimisticConfig, updateOptimisticConfig] = useOptimistic(
    config,
    (state: WidgetConfig, update: OptimisticUpdate) => {
      switch (update.action) {
        case 'config':
          return { ...state, ...update.payload };
        default:
          return state;
      }
    }
  );

  // Handle configuration changes
  const handleConfigChange = (changes: Partial<WidgetConfig>) => {
    startTransition(() => {
      updateOptimisticConfig({ action: 'config', payload: changes });
      onConfigChange?.(changes);
    });
  };

  // Handle resize
  const handleResize = () => {
    const sizes: WidgetConfig['size'][] = ['small', 'medium', 'large', 'xlarge'];
    const currentIndex = sizes.indexOf(optimisticConfig.size);
    const nextSize = sizes[(currentIndex + 1) % sizes.length];

    handleConfigChange({ size: nextSize });
    onResize?.(nextSize);
  };

  // Handle refresh
  const handleRefresh = () => {
    startTransition(() => {
      onRefresh?.();
    });
  };

  // Determine if widget is stale
  const isStale = data ?
    (Date.now() - data.timestamp > (optimisticConfig.refreshInterval || 30000)) : false;

  // Error boundary wrapper
  const WidgetErrorBoundary = ErrorBoundary || React.Fragment;

  return (
    <Card
      className={cn(
        'relative transition-all duration-200 hover:shadow-md',
        'border-l-4',
        sizeClasses[optimisticConfig.size],
        priorityColors[optimisticConfig.priority],
        {
          'opacity-75': data?.isLoading || isPending,
          'border-destructive/50': data?.error,
          'border-warning/50': isStale,
        },
        className
      )}
    >
      {/* Widget Header */}
      {optimisticConfig.showHeader !== false && (
        <div className="flex items-center justify-between p-4 pb-2">
          <div className="flex-1 min-w-0">
            <div className="flex items-center space-x-2">
              <h3 className="text-sm font-semibold text-foreground truncate">
                {optimisticConfig.title}
              </h3>

              {/* Status indicators */}
              {data?.isLoading && (
                <IconLoader2 className="h-3 w-3 animate-spin text-muted-foreground" />
              )}

              {data?.error && (
                <div className="h-2 w-2 rounded-full bg-destructive" title={data.error} />
              )}

              {isStale && (
                <div className="h-2 w-2 rounded-full bg-warning" title="Data is stale" />
              )}
            </div>

            {optimisticConfig.subtitle && (
              <p className="text-xs text-muted-foreground truncate">
                {optimisticConfig.subtitle}
              </p>
            )}
          </div>

          {/* Widget Controls */}
          {optimisticConfig.showControls !== false && (
            <div className="flex items-center space-x-1">
              {/* Custom actions */}
              {optimisticConfig.customActions?.map(action => (
                <Button
                  key={action.id}
                  variant="ghost"
                  size="sm"
                  className="h-6 w-6 p-0"
                  onClick={action.onClick}
                  title={action.label}
                >
                  <action.icon className="h-3 w-3" />
                </Button>
              ))}

              {/* Refresh button */}
              <Button
                variant="ghost"
                size="sm"
                className="h-6 w-6 p-0"
                onClick={handleRefresh}
                disabled={data?.isLoading || isPending}
                title="Refresh"
              >
                <IconLoader2 className={cn(
                  'h-3 w-3',
                  (data?.isLoading || isPending) && 'animate-spin'
                )} />
              </Button>

              {/* Resize button */}
              {optimisticConfig.allowResize && (
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-6 w-6 p-0"
                  onClick={handleResize}
                  title="Resize"
                >
                  {optimisticConfig.size === 'small' || optimisticConfig.size === 'medium' ? (
                    <IconMaximize2 className="h-3 w-3" />
                  ) : (
                    <IconMinimize2 className="h-3 w-3" />
                  )}
                </Button>
              )}

              {/* Settings button */}
              <Button
                variant="ghost"
                size="sm"
                className="h-6 w-6 p-0"
                title="Settings"
              >
                <IconSettings className="h-3 w-3" />
              </Button>

              {/* Close button */}
              {optimisticConfig.allowClose && (
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-6 w-6 p-0 text-muted-foreground hover:text-destructive"
                  onClick={onClose}
                  title="Close"
                >
                  <IconX className="h-3 w-3" />
                </Button>
              )}
            </div>
          )}
        </div>
      )}

      {/* Widget Content */}
      <div className={cn(
        'px-4 pb-4',
        optimisticConfig.showHeader === false && 'pt-4'
      )}>
        <WidgetErrorBoundary>
          <Suspense fallback={<WidgetSkeleton size={optimisticConfig.size} />}>
            {data?.error ? (
              <div className="flex items-center justify-center h-24 text-sm text-muted-foreground">
                <div className="text-center">
                  <p>Failed to load data</p>
                  <p className="text-xs mt-1">{data.error}</p>
                  <Button
                    variant="outline"
                    size="sm"
                    className="mt-2"
                    onClick={handleRefresh}
                  >
                    Retry
                  </Button>
                </div>
              </div>
            ) : data?.isLoading ? (
              <WidgetSkeleton size={optimisticConfig.size} />
            ) : (
              children
            )}
          </Suspense>
        </WidgetErrorBoundary>
      </div>

      {/* Widget Footer */}
      {data && (
        <div className="px-4 pb-2">
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>
              {optimisticConfig.category} • {optimisticConfig.priority} priority
            </span>
            <span>
              Updated {data.timestamp ? new Date(data.timestamp).toLocaleTimeString() : 'Never'}
            </span>
          </div>
        </div>
      )}

      {/* Processing overlay */}
      {isPending && (
        <div className="absolute inset-0 bg-background/20 flex items-center justify-center rounded-lg">
          <IconLoader2 className="h-6 w-6 animate-spin text-primary" />
        </div>
      )}
    </Card>
  );
};

export default DashboardWidget;
