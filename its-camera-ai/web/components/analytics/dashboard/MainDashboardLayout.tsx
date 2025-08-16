/**
 * Main Dashboard Layout Component
 *
 * Comprehensive dashboard layout with drag-and-drop support, customizable
 * widgets, and multi-view management for the ITS Camera AI system.
 */

'use client';

import React, { useEffect, useState, useMemo, useCallback } from 'react';
import { useDeferredValue, useTransition, useOptimistic } from 'react';
import { cn } from '@/lib/utils';
import {
  IconLayoutDashboard,
  IconSettings,
  IconPlus,
  IconGrid3x3,
  IconMaximize2,
  IconMinimize2,
  IconRefresh,
  IconDownload,
  IconFilter,
  IconSearch,
  IconBookmark,
  IconShare,
  IconEdit,
  IconTrash,
  IconMove,
  IconEye,
  IconEyeOff,
  IconLock,
  IconUnlock
} from '@tabler/icons-react';
import { TrafficAnalyticsDashboard } from './TrafficAnalyticsDashboard';
import { IncidentManagementDashboard } from './IncidentManagementDashboard';
import { CameraGridDashboard } from './CameraGridDashboard';
import { SystemHealthMonitoringDashboard } from './SystemHealthMonitoringDashboard';
import { DashboardWidget, WidgetConfig } from './DashboardWidget';
import { MetricCard, MetricCardData } from './MetricCard';
import { useAnalyticsStore } from '@/stores/analytics';
import { useRealTimeMetrics } from '@/providers/RealTimeProvider';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  ToggleGroup,
  ToggleGroupItem,
} from '@/components/ui/toggle-group';
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from '@/components/ui/sheet';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';

export interface DashboardView {
  id: string;
  name: string;
  description: string;
  layout: 'grid' | 'masonry' | 'custom';
  widgets: Array<{
    id: string;
    type: 'traffic' | 'incidents' | 'cameras' | 'system' | 'metric' | 'chart';
    position: { x: number; y: number; w: number; h: number };
    config: any;
    visible: boolean;
    locked: boolean;
  }>;
  settings: {
    autoRefresh: boolean;
    refreshInterval: number;
    theme: 'light' | 'dark' | 'auto';
    density: 'compact' | 'comfortable' | 'spacious';
  };
  permissions: {
    canEdit: boolean;
    canShare: boolean;
    canDelete: boolean;
  };
  metadata: {
    createdAt: number;
    updatedAt: number;
    createdBy: string;
    shared: boolean;
    favorite: boolean;
  };
}

export interface MainDashboardLayoutProps {
  className?: string;
  initialView?: string;
  showHeader?: boolean;
  showSidebar?: boolean;
  allowCustomization?: boolean;
  onViewChange?: (viewId: string) => void;
  onWidgetAdd?: (widget: any) => void;
  onWidgetRemove?: (widgetId: string) => void;
  onViewSave?: (view: DashboardView) => void;
  onViewShare?: (viewId: string) => void;
}

// Available widget types
const availableWidgets = [
  {
    id: 'traffic-analytics',
    name: 'Traffic Analytics',
    description: 'Real-time traffic monitoring and analysis',
    type: 'traffic',
    icon: IconLayoutDashboard,
    component: TrafficAnalyticsDashboard,
    defaultSize: { w: 12, h: 8 },
  },
  {
    id: 'incident-management',
    name: 'Incident Management',
    description: 'Active incident monitoring and response',
    type: 'incidents',
    icon: IconLayoutDashboard,
    component: IncidentManagementDashboard,
    defaultSize: { w: 12, h: 6 },
  },
  {
    id: 'camera-grid',
    name: 'Camera Grid',
    description: 'Live camera monitoring and status',
    type: 'cameras',
    icon: IconGrid3x3,
    component: CameraGridDashboard,
    defaultSize: { w: 12, h: 8 },
  },
  {
    id: 'system-health',
    name: 'System Health',
    description: 'Infrastructure monitoring and performance',
    type: 'system',
    icon: IconLayoutDashboard,
    component: SystemHealthMonitoringDashboard,
    defaultSize: { w: 12, h: 6 },
  },
];

// Generate default dashboard views
const generateDefaultViews = (): DashboardView[] => [
  {
    id: 'overview',
    name: 'Overview',
    description: 'Complete system overview with all key metrics',
    layout: 'grid',
    widgets: [
      {
        id: 'traffic-widget',
        type: 'traffic',
        position: { x: 0, y: 0, w: 12, h: 8 },
        config: { showControls: true, autoRefresh: true },
        visible: true,
        locked: false,
      },
      {
        id: 'incidents-widget',
        type: 'incidents',
        position: { x: 0, y: 8, w: 6, h: 6 },
        config: { showControls: true, autoRefresh: true },
        visible: true,
        locked: false,
      },
      {
        id: 'system-widget',
        type: 'system',
        position: { x: 6, y: 8, w: 6, h: 6 },
        config: { showControls: true, autoRefresh: true },
        visible: true,
        locked: false,
      },
    ],
    settings: {
      autoRefresh: true,
      refreshInterval: 30000,
      theme: 'auto',
      density: 'comfortable',
    },
    permissions: {
      canEdit: true,
      canShare: true,
      canDelete: false,
    },
    metadata: {
      createdAt: Date.now(),
      updatedAt: Date.now(),
      createdBy: 'system',
      shared: false,
      favorite: true,
    },
  },
  {
    id: 'traffic-focus',
    name: 'Traffic Focus',
    description: 'Detailed traffic analytics and monitoring',
    layout: 'grid',
    widgets: [
      {
        id: 'traffic-main',
        type: 'traffic',
        position: { x: 0, y: 0, w: 12, h: 12 },
        config: { showControls: true, autoRefresh: true },
        visible: true,
        locked: false,
      },
    ],
    settings: {
      autoRefresh: true,
      refreshInterval: 15000,
      theme: 'auto',
      density: 'spacious',
    },
    permissions: {
      canEdit: true,
      canShare: true,
      canDelete: true,
    },
    metadata: {
      createdAt: Date.now(),
      updatedAt: Date.now(),
      createdBy: 'system',
      shared: false,
      favorite: false,
    },
  },
  {
    id: 'operations',
    name: 'Operations',
    description: 'Incident management and system operations',
    layout: 'grid',
    widgets: [
      {
        id: 'incidents-main',
        type: 'incidents',
        position: { x: 0, y: 0, w: 8, h: 8 },
        config: { showControls: true, autoRefresh: true },
        visible: true,
        locked: false,
      },
      {
        id: 'cameras-main',
        type: 'cameras',
        position: { x: 8, y: 0, w: 4, h: 8 },
        config: { viewMode: 'list', showControls: true },
        visible: true,
        locked: false,
      },
      {
        id: 'system-health',
        type: 'system',
        position: { x: 0, y: 8, w: 12, h: 6 },
        config: { showControls: true, autoRefresh: true },
        visible: true,
        locked: false,
      },
    ],
    settings: {
      autoRefresh: true,
      refreshInterval: 20000,
      theme: 'auto',
      density: 'compact',
    },
    permissions: {
      canEdit: true,
      canShare: true,
      canDelete: true,
    },
    metadata: {
      createdAt: Date.now(),
      updatedAt: Date.now(),
      createdBy: 'system',
      shared: false,
      favorite: false,
    },
  },
];

export const MainDashboardLayout: React.FC<MainDashboardLayoutProps> = ({
  className,
  initialView = 'overview',
  showHeader = true,
  showSidebar = true,
  allowCustomization = true,
  onViewChange,
  onWidgetAdd,
  onWidgetRemove,
  onViewSave,
  onViewShare,
}) => {
  const [isPending, startTransition] = useTransition();
  const [views, setViews] = useState<DashboardView[]>(generateDefaultViews());
  const [currentViewId, setCurrentViewId] = useState(initialView);
  const [isEditMode, setIsEditMode] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedDensity, setSelectedDensity] = useState<'compact' | 'comfortable' | 'spacious'>('comfortable');

  // Optimistic state for view updates
  const [optimisticViews, updateOptimisticViews] = useOptimistic(
    views,
    (state: DashboardView[], update: { action: string; payload: any }) => {
      switch (update.action) {
        case 'update_view':
          return state.map(view =>
            view.id === update.payload.id ? { ...view, ...update.payload } : view
          );
        case 'add_widget':
          return state.map(view =>
            view.id === currentViewId
              ? { ...view, widgets: [...view.widgets, update.payload] }
              : view
          );
        case 'remove_widget':
          return state.map(view =>
            view.id === currentViewId
              ? { ...view, widgets: view.widgets.filter(w => w.id !== update.payload) }
              : view
          );
        default:
          return state;
      }
    }
  );

  // Real-time data
  const analyticsData = useAnalyticsStore(state => state.currentMetrics);
  const { isConnected, metrics } = useRealTimeMetrics();

  // Current view
  const currentView = useMemo(() =>
    optimisticViews.find(view => view.id === currentViewId) || optimisticViews[0],
    [optimisticViews, currentViewId]
  );

  const deferredCurrentView = useDeferredValue(currentView);

  // Handle view change
  const handleViewChange = useCallback((viewId: string) => {
    startTransition(() => {
      setCurrentViewId(viewId);
      onViewChange?.(viewId);
    });
  }, [onViewChange]);

  // Handle widget operations
  const handleWidgetAdd = useCallback((widgetType: string) => {
    const widget = availableWidgets.find(w => w.id === widgetType);
    if (!widget) return;

    const newWidget = {
      id: `${widget.id}-${Date.now()}`,
      type: widget.type,
      position: {
        x: 0,
        y: 0,
        w: widget.defaultSize.w,
        h: widget.defaultSize.h
      },
      config: { showControls: true, autoRefresh: true },
      visible: true,
      locked: false,
    };

    updateOptimisticViews({ action: 'add_widget', payload: newWidget });
    onWidgetAdd?.(newWidget);
  }, [updateOptimisticViews, onWidgetAdd]);

  const handleWidgetRemove = useCallback((widgetId: string) => {
    updateOptimisticViews({ action: 'remove_widget', payload: widgetId });
    onWidgetRemove?.(widgetId);
  }, [updateOptimisticViews, onWidgetRemove]);

  // Handle widget visibility toggle
  const handleWidgetVisibilityToggle = useCallback((widgetId: string) => {
    const updatedView = {
      ...currentView,
      widgets: currentView.widgets.map(widget =>
        widget.id === widgetId ? { ...widget, visible: !widget.visible } : widget
      ),
    };

    updateOptimisticViews({ action: 'update_view', payload: updatedView });
  }, [currentView, updateOptimisticViews]);

  // Render widget component
  const renderWidget = useCallback((widget: any) => {
    const widgetInfo = availableWidgets.find(w => w.type === widget.type);
    if (!widgetInfo || !widget.visible) return null;

    const Component = widgetInfo.component;

    return (
      <div
        key={widget.id}
        className={cn(
          'relative',
          selectedDensity === 'compact' && 'space-y-2',
          selectedDensity === 'comfortable' && 'space-y-4',
          selectedDensity === 'spacious' && 'space-y-6'
        )}
        style={{
          gridColumn: `span ${widget.position.w}`,
          gridRow: `span ${widget.position.h}`,
        }}
      >
        {isEditMode && (
          <div className="absolute top-2 right-2 z-10 flex space-x-1">
            <Button
              variant="ghost"
              size="sm"
              className="h-6 w-6 p-0 bg-background/80 hover:bg-background"
              onClick={() => handleWidgetVisibilityToggle(widget.id)}
              title={widget.visible ? 'Hide widget' : 'Show widget'}
            >
              {widget.visible ? (
                <IconEye className="h-3 w-3" />
              ) : (
                <IconEyeOff className="h-3 w-3" />
              )}
            </Button>
            <Button
              variant="ghost"
              size="sm"
              className="h-6 w-6 p-0 bg-background/80 hover:bg-background text-destructive"
              onClick={() => handleWidgetRemove(widget.id)}
              title="Remove widget"
            >
              <IconTrash className="h-3 w-3" />
            </Button>
          </div>
        )}

        <Component
          {...widget.config}
          className={cn(
            isEditMode && 'border-2 border-dashed border-primary/50',
            'transition-all duration-200'
          )}
        />
      </div>
    );
  }, [selectedDensity, isEditMode, handleWidgetVisibilityToggle, handleWidgetRemove]);

  // Grid classes based on density
  const gridClasses = {
    compact: 'grid-cols-12 gap-2',
    comfortable: 'grid-cols-12 gap-4',
    spacious: 'grid-cols-12 gap-6',
  };

  return (
    <div className={cn('flex h-screen bg-background', className)}>
      {/* Sidebar */}
      {showSidebar && (
        <div className="w-64 border-r bg-muted/30 flex flex-col">
          {/* Sidebar Header */}
          <div className="p-4 border-b">
            <h2 className="text-lg font-semibold">Dashboard Views</h2>
            <p className="text-sm text-muted-foreground">
              {optimisticViews.length} views available
            </p>
          </div>

          {/* Search */}
          <div className="p-4">
            <div className="relative">
              <IconSearch className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search views..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-10"
              />
            </div>
          </div>

          {/* View List */}
          <ScrollArea className="flex-1 px-4">
            <div className="space-y-2">
              {optimisticViews
                .filter(view =>
                  view.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                  view.description.toLowerCase().includes(searchTerm.toLowerCase())
                )
                .map((view) => (
                  <Card
                    key={view.id}
                    className={cn(
                      'p-3 cursor-pointer transition-all hover:shadow-md',
                      currentViewId === view.id && 'ring-2 ring-primary',
                      view.metadata.favorite && 'border-warning'
                    )}
                    onClick={() => handleViewChange(view.id)}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center space-x-2">
                          <h4 className="text-sm font-semibold truncate">
                            {view.name}
                          </h4>
                          {view.metadata.favorite && (
                            <IconBookmark className="h-3 w-3 text-warning" />
                          )}
                          {view.metadata.shared && (
                            <IconShare className="h-3 w-3 text-info" />
                          )}
                        </div>
                        <p className="text-xs text-muted-foreground truncate mt-1">
                          {view.description}
                        </p>
                        <div className="flex items-center space-x-2 mt-2 text-xs text-muted-foreground">
                          <span>{view.widgets.filter(w => w.visible).length} widgets</span>
                          <span>•</span>
                          <span>{view.layout}</span>
                        </div>
                      </div>
                    </div>
                  </Card>
                ))}
            </div>
          </ScrollArea>

          {/* Sidebar Footer */}
          {allowCustomization && (
            <div className="p-4 border-t space-y-2">
              <Sheet>
                <SheetTrigger asChild>
                  <Button variant="outline" className="w-full" size="sm">
                    <IconPlus className="h-4 w-4 mr-2" />
                    Add Widget
                  </Button>
                </SheetTrigger>
                <SheetContent>
                  <SheetHeader>
                    <SheetTitle>Add Widget</SheetTitle>
                    <SheetDescription>
                      Choose a widget to add to your dashboard
                    </SheetDescription>
                  </SheetHeader>
                  <div className="space-y-4 mt-6">
                    {availableWidgets.map((widget) => {
                      const WidgetIcon = widget.icon;
                      return (
                        <Card
                          key={widget.id}
                          className="p-4 cursor-pointer hover:shadow-md transition-all"
                          onClick={() => handleWidgetAdd(widget.id)}
                        >
                          <div className="flex items-center space-x-3">
                            <div className="p-2 rounded-lg bg-primary/20">
                              <WidgetIcon className="h-5 w-5 text-primary" />
                            </div>
                            <div className="flex-1">
                              <h4 className="text-sm font-semibold">{widget.name}</h4>
                              <p className="text-xs text-muted-foreground">
                                {widget.description}
                              </p>
                            </div>
                          </div>
                        </Card>
                      );
                    })}
                  </div>
                </SheetContent>
              </Sheet>

              <Button
                variant={isEditMode ? 'default' : 'outline'}
                className="w-full"
                size="sm"
                onClick={() => setIsEditMode(!isEditMode)}
              >
                <IconEdit className="h-4 w-4 mr-2" />
                {isEditMode ? 'Exit Edit' : 'Edit Mode'}
              </Button>
            </div>
          )}
        </div>
      )}

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        {showHeader && (
          <div className="border-b bg-background p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4">
                <div>
                  <h1 className="text-xl font-bold">{deferredCurrentView.name}</h1>
                  <p className="text-sm text-muted-foreground">
                    {deferredCurrentView.description}
                  </p>
                </div>

                {/* Connection status */}
                <Badge variant={isConnected ? 'default' : 'destructive'}>
                  {isConnected ? 'Connected' : 'Disconnected'}
                </Badge>
              </div>

              <div className="flex items-center space-x-2">
                {/* Density selector */}
                <Select value={selectedDensity} onValueChange={setSelectedDensity}>
                  <SelectTrigger className="w-32">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="compact">Compact</SelectItem>
                    <SelectItem value="comfortable">Comfortable</SelectItem>
                    <SelectItem value="spacious">Spacious</SelectItem>
                  </SelectContent>
                </Select>

                {/* View controls */}
                <Button variant="outline" size="sm">
                  <IconRefresh className="h-4 w-4 mr-2" />
                  Refresh
                </Button>

                <Button variant="outline" size="sm">
                  <IconDownload className="h-4 w-4 mr-2" />
                  Export
                </Button>

                <Button variant="outline" size="sm">
                  <IconSettings className="h-4 w-4" />
                </Button>
              </div>
            </div>
          </div>
        )}

        {/* Dashboard Content */}
        <div className="flex-1 overflow-auto p-4">
          <div
            className={cn(
              'grid auto-rows-min',
              gridClasses[selectedDensity],
              isEditMode && 'border-2 border-dashed border-primary/30 rounded-lg p-4'
            )}
          >
            {deferredCurrentView.widgets.map(renderWidget)}

            {/* Add widget placeholder in edit mode */}
            {isEditMode && (
              <div className="col-span-12 min-h-[200px] border-2 border-dashed border-muted-foreground/50 rounded-lg flex items-center justify-center">
                <div className="text-center text-muted-foreground">
                  <IconPlus className="h-8 w-8 mx-auto mb-2" />
                  <p>Add widgets from the sidebar</p>
                </div>
              </div>
            )}
          </div>

          {/* Empty state */}
          {deferredCurrentView.widgets.filter(w => w.visible).length === 0 && !isEditMode && (
            <div className="flex items-center justify-center h-96">
              <div className="text-center text-muted-foreground">
                <IconLayoutDashboard className="h-16 w-16 mx-auto mb-4" />
                <h3 className="text-lg font-semibold mb-2">No widgets in this view</h3>
                <p className="mb-4">Add some widgets to get started</p>
                {allowCustomization && (
                  <Button onClick={() => setIsEditMode(true)}>
                    <IconPlus className="h-4 w-4 mr-2" />
                    Add Widgets
                  </Button>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default MainDashboardLayout;
