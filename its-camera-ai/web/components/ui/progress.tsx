/**
 * Progress Component
 *
 * A customizable progress bar component built on Radix UI.
 * Supports different sizes, colors, and accessibility features.
 */

'use client';

import * as React from 'react';
import * as ProgressPrimitive from '@radix-ui/react-progress';
import { cn } from '@/lib/utils';

export interface ProgressProps
  extends React.ComponentPropsWithoutRef<typeof ProgressPrimitive.Root> {
  value?: number;
  max?: number;
  size?: 'sm' | 'md' | 'lg';
  variant?: 'default' | 'success' | 'warning' | 'destructive';
  showValue?: boolean;
  animated?: boolean;
}

const Progress = React.forwardRef<
  React.ElementRef<typeof ProgressPrimitive.Root>,
  ProgressProps
>(({
  className,
  value = 0,
  max = 100,
  size = 'md',
  variant = 'default',
  showValue = false,
  animated = false,
  ...props
}, ref) => {
  const percentage = Math.min(Math.max((value / max) * 100, 0), 100);

  const sizeClasses = {
    sm: 'h-1',
    md: 'h-2',
    lg: 'h-3',
  };

  const variantClasses = {
    default: 'bg-primary',
    success: 'bg-success',
    warning: 'bg-warning',
    destructive: 'bg-destructive',
  };

  return (
    <div className="relative">
      <ProgressPrimitive.Root
        ref={ref}
        className={cn(
          'relative overflow-hidden rounded-full bg-secondary',
          sizeClasses[size],
          className
        )}
        {...props}
      >
        <ProgressPrimitive.Indicator
          className={cn(
            'h-full w-full flex-1 transition-all duration-300 ease-in-out',
            variantClasses[variant],
            animated && 'animate-pulse'
          )}
          style={{
            transform: `translateX(-${100 - percentage}%)`,
            transition: 'transform 0.3s ease-in-out'
          }}
        />
      </ProgressPrimitive.Root>

      {showValue && (
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-xs font-medium text-foreground">
            {Math.round(percentage)}%
          </span>
        </div>
      )}
    </div>
  );
});

Progress.displayName = ProgressPrimitive.Root.displayName;

export { Progress };
