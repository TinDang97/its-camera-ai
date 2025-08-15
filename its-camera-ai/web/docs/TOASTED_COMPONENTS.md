# Toasted Components Documentation

## Overview

The ITS Camera AI dashboard uses a comprehensive toast notification system for user feedback, alerts, and system status updates. This system provides real-time notifications with different types, animations, and positioning options.

## Toast Component Architecture

### Core Components

#### 1. Toast Provider
```tsx
// components/ui/toast/ToastProvider.tsx
'use client';

import { createContext, useContext, useReducer, ReactNode } from 'react';

interface Toast {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info' | 'loading';
  title: string;
  description?: string;
  duration?: number;
  action?: {
    label: string;
    onClick: () => void;
  };
  dismissible?: boolean;
  position?: 'top-right' | 'top-left' | 'bottom-right' | 'bottom-left' | 'top-center' | 'bottom-center';
}

interface ToastContextType {
  toasts: Toast[];
  addToast: (toast: Omit<Toast, 'id'>) => string;
  removeToast: (id: string) => void;
  updateToast: (id: string, updates: Partial<Toast>) => void;
}
```

#### 2. Toast Component
```tsx
// components/ui/toast/Toast.tsx
'use client';

import { memo, useEffect, useState, useCallback } from 'react';
import { X, CheckCircle, AlertCircle, AlertTriangle, Info, Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';

const TOAST_ICONS = {
  success: CheckCircle,
  error: AlertCircle,
  warning: AlertTriangle,
  info: Info,
  loading: Loader2,
};

const TOAST_STYLES = {
  success: 'bg-green-50 border-green-200 text-green-900',
  error: 'bg-red-50 border-red-200 text-red-900',
  warning: 'bg-yellow-50 border-yellow-200 text-yellow-900',
  info: 'bg-blue-50 border-blue-200 text-blue-900',
  loading: 'bg-gray-50 border-gray-200 text-gray-900',
};

export const ToastComponent = memo<Toast>(({
  id,
  type,
  title,
  description,
  duration = 5000,
  action,
  dismissible = true,
  onRemove,
}) => {
  const [isVisible, setIsVisible] = useState(true);
  const [isExiting, setIsExiting] = useState(false);
  const Icon = TOAST_ICONS[type];

  const handleDismiss = useCallback(() => {
    setIsExiting(true);
    setTimeout(() => {
      onRemove(id);
    }, 300);
  }, [id, onRemove]);

  useEffect(() => {
    if (type !== 'loading' && duration > 0) {
      const timer = setTimeout(handleDismiss, duration);
      return () => clearTimeout(timer);
    }
  }, [duration, type, handleDismiss]);

  return (
    <div
      className={cn(
        'flex items-start gap-3 p-4 rounded-lg border shadow-lg transition-all duration-300',
        TOAST_STYLES[type],
        isExiting ? 'opacity-0 translate-x-full' : 'opacity-100 translate-x-0'
      )}
      role="alert"
      aria-live="polite"
    >
      <Icon 
        className={cn(
          'h-5 w-5 mt-0.5 flex-shrink-0',
          type === 'loading' && 'animate-spin'
        )} 
      />
      
      <div className="flex-1 min-w-0">
        <h3 className="font-semibold text-sm">{title}</h3>
        {description && (
          <p className="text-sm opacity-90 mt-1">{description}</p>
        )}
        
        {action && (
          <button
            onClick={action.onClick}
            className="mt-2 text-sm font-medium underline hover:no-underline focus:outline-none focus:ring-2 focus:ring-current rounded"
          >
            {action.label}
          </button>
        )}
      </div>

      {dismissible && (
        <button
          onClick={handleDismiss}
          className="flex-shrink-0 p-1 rounded hover:bg-black/10 focus:outline-none focus:ring-2 focus:ring-current transition-colors"
          aria-label="Dismiss notification"
        >
          <X className="h-4 w-4" />
        </button>
      )}
    </div>
  );
});
```

### Hook Implementation

#### useToast Hook
```tsx
// hooks/useToast.ts
'use client';

import { useContext, useCallback } from 'react';
import { ToastContext } from '@/components/ui/toast/ToastProvider';

export function useToast() {
  const context = useContext(ToastContext);
  
  if (!context) {
    throw new Error('useToast must be used within a ToastProvider');
  }

  const { addToast, removeToast, updateToast } = context;

  const toast = useCallback({
    success: (title: string, description?: string) =>
      addToast({ type: 'success', title, description }),
    
    error: (title: string, description?: string) =>
      addToast({ type: 'error', title, description, duration: 8000 }),
    
    warning: (title: string, description?: string) =>
      addToast({ type: 'warning', title, description, duration: 6000 }),
    
    info: (title: string, description?: string) =>
      addToast({ type: 'info', title, description }),
    
    loading: (title: string, description?: string) =>
      addToast({ type: 'loading', title, description, duration: 0, dismissible: false }),
    
    custom: (toast: Omit<Toast, 'id'>) => addToast(toast),
    
    dismiss: (id: string) => removeToast(id),
    
    update: (id: string, updates: Partial<Toast>) => updateToast(id, updates),
  }, [addToast, removeToast, updateToast]);

  return toast;
}
```

## Usage Examples

### Basic Usage
```tsx
// In your component
import { useToast } from '@/hooks/useToast';

export function MyComponent() {
  const toast = useToast();

  const handleSuccess = () => {
    toast.success('Operation completed!', 'Your changes have been saved successfully.');
  };

  const handleError = () => {
    toast.error('Something went wrong', 'Please try again or contact support.');
  };

  const handleLoading = async () => {
    const loadingId = toast.loading('Processing...', 'Please wait while we process your request.');
    
    try {
      await someAsyncOperation();
      toast.update(loadingId, {
        type: 'success',
        title: 'Success!',
        description: 'Operation completed successfully.',
        duration: 5000,
        dismissible: true,
      });
    } catch (error) {
      toast.update(loadingId, {
        type: 'error',
        title: 'Error',
        description: 'Operation failed. Please try again.',
        duration: 8000,
        dismissible: true,
      });
    }
  };
}
```

### ITS-Specific Toast Examples

#### Camera Alert Toasts
```tsx
// Camera-specific notifications
const cameraToasts = {
  cameraOffline: (cameraId: string) =>
    toast.error(
      'Camera Offline',
      `Camera ${cameraId} has gone offline. Check connection.`,
      {
        action: {
          label: 'View Details',
          onClick: () => router.push(`/cameras/${cameraId}`),
        },
      }
    ),

  alertDetected: (alertType: string, location: string) =>
    toast.warning(
      `${alertType} Detected`,
      `Alert detected at ${location}`,
      {
        duration: 10000,
        action: {
          label: 'View Alert',
          onClick: () => router.push('/alerts'),
        },
      }
    ),

  systemHealthWarning: (message: string) =>
    toast.warning('System Health Warning', message, {
      duration: 0, // Persistent
      position: 'top-center',
    }),
};
```

#### Real-time Event Toasts
```tsx
// Real-time event notifications
const eventToasts = {
  newIncident: (incidentData: IncidentData) =>
    toast.info(
      'New Incident Reported',
      `${incidentData.type} at ${incidentData.location}`,
      {
        action: {
          label: 'Investigate',
          onClick: () => handleIncidentInvestigation(incidentData.id),
        },
      }
    ),

  trafficCongestion: (location: string, severity: string) =>
    toast.warning(
      'Traffic Congestion Alert',
      `${severity} congestion detected at ${location}`,
      {
        duration: 15000,
        position: 'top-right',
      }
    ),

  systemMaintenance: (scheduledTime: string) =>
    toast.info(
      'Scheduled Maintenance',
      `System maintenance scheduled for ${scheduledTime}`,
      {
        duration: 0,
        dismissible: true,
        position: 'bottom-center',
      }
    ),
};
```

## Toast Container Implementation

### Position-based Containers
```tsx
// components/ui/toast/ToastContainer.tsx
'use client';

import { createPortal } from 'react-dom';
import { useEffect, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';

const POSITION_CLASSES = {
  'top-right': 'top-4 right-4',
  'top-left': 'top-4 left-4',
  'bottom-right': 'bottom-4 right-4',
  'bottom-left': 'bottom-4 left-4',
  'top-center': 'top-4 left-1/2 transform -translate-x-1/2',
  'bottom-center': 'bottom-4 left-1/2 transform -translate-x-1/2',
};

export function ToastContainer({ position = 'top-right', toasts }) {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) return null;

  return createPortal(
    <div
      className={cn(
        'fixed z-[100] flex flex-col gap-2 w-full max-w-sm',
        POSITION_CLASSES[position]
      )}
    >
      <AnimatePresence mode="popLayout">
        {toasts.map((toast) => (
          <motion.div
            key={toast.id}
            initial={{ opacity: 0, x: position.includes('right') ? 100 : -100 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: position.includes('right') ? 100 : -100 }}
            transition={{ duration: 0.3, ease: 'easeInOut' }}
          >
            <ToastComponent {...toast} />
          </motion.div>
        ))}
      </AnimatePresence>
    </div>,
    document.body
  );
}
```

## Advanced Features

### Toast Queue Management
```tsx
// Advanced toast queue with priority and grouping
class ToastQueue {
  private queue: Toast[] = [];
  private maxVisible = 5;
  private priorities = { error: 3, warning: 2, success: 1, info: 1, loading: 0 };

  add(toast: Toast) {
    // Add priority-based insertion
    const priority = this.priorities[toast.type];
    const insertIndex = this.queue.findIndex(t => this.priorities[t.type] < priority);
    
    if (insertIndex === -1) {
      this.queue.push(toast);
    } else {
      this.queue.splice(insertIndex, 0, toast);
    }

    this.processQueue();
  }

  private processQueue() {
    // Show only maxVisible toasts
    const visible = this.queue.slice(0, this.maxVisible);
    const hidden = this.queue.slice(this.maxVisible);
    
    // Group similar toasts
    const grouped = this.groupSimilarToasts(visible);
    
    return grouped;
  }

  private groupSimilarToasts(toasts: Toast[]) {
    // Implementation for grouping similar notifications
    const groups = new Map();
    
    toasts.forEach(toast => {
      const key = `${toast.type}-${toast.title}`;
      if (groups.has(key)) {
        groups.get(key).count++;
      } else {
        groups.set(key, { ...toast, count: 1 });
      }
    });

    return Array.from(groups.values());
  }
}
```

### Accessibility Features
```tsx
// Enhanced accessibility support
export function AccessibleToast({ toast }: { toast: Toast }) {
  useEffect(() => {
    // Announce to screen readers
    const announcement = `${toast.type} notification: ${toast.title}${
      toast.description ? `. ${toast.description}` : ''
    }`;
    
    const announcer = document.createElement('div');
    announcer.setAttribute('aria-live', 'polite');
    announcer.setAttribute('aria-atomic', 'true');
    announcer.className = 'sr-only';
    announcer.textContent = announcement;
    
    document.body.appendChild(announcer);
    
    return () => {
      document.body.removeChild(announcer);
    };
  }, [toast]);

  return <ToastComponent {...toast} />;
}
```

## Integration Guidelines

### 1. Provider Setup
```tsx
// app/layout.tsx
import { ToastProvider } from '@/components/ui/toast/ToastProvider';

export default function RootLayout({ children }) {
  return (
    <html>
      <body>
        <ToastProvider>
          {children}
        </ToastProvider>
      </body>
    </html>
  );
}
```

### 2. Global Error Handling
```tsx
// lib/error-handler.ts
import { useToast } from '@/hooks/useToast';

export function setupGlobalErrorHandling() {
  const toast = useToast();
  
  window.addEventListener('unhandledrejection', (event) => {
    toast.error('Unexpected Error', 'Something went wrong. Please refresh the page.');
    console.error('Unhandled promise rejection:', event.reason);
  });

  window.addEventListener('error', (event) => {
    toast.error('Application Error', 'A critical error occurred. Please contact support.');
    console.error('Global error:', event.error);
  });
}
```

### 3. API Integration
```tsx
// lib/api-client.ts
import { useToast } from '@/hooks/useToast';

export function createApiClient() {
  const toast = useToast();
  
  return {
    async request(url: string, options: RequestInit) {
      const loadingId = toast.loading('Loading...', 'Please wait...');
      
      try {
        const response = await fetch(url, options);
        
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        toast.dismiss(loadingId);
        return await response.json();
      } catch (error) {
        toast.update(loadingId, {
          type: 'error',
          title: 'Request Failed',
          description: error.message,
          duration: 8000,
          dismissible: true,
        });
        throw error;
      }
    }
  };
}
```

## Best Practices

### 1. Toast Content Guidelines
- **Keep titles concise** (max 3-4 words)
- **Make descriptions informative** but brief
- **Use action buttons** for important follow-ups
- **Consider user context** when choosing duration

### 2. Performance Optimization
- **Limit concurrent toasts** (max 5 visible)
- **Use React.memo** for toast components
- **Debounce similar notifications**
- **Clean up timers** properly

### 3. User Experience
- **Position toasts consistently**
- **Use appropriate durations** based on importance
- **Provide dismissal options**
- **Group similar notifications**

### 4. Accessibility
- **Include proper ARIA labels**
- **Support keyboard navigation**
- **Announce important toasts** to screen readers
- **Provide sufficient color contrast**

This comprehensive toast system provides a robust foundation for user notifications in the ITS Camera AI dashboard, with built-in performance optimizations, accessibility features, and extensive customization options.