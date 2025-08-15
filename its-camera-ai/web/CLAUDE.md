# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **ITS Camera AI Web Dashboard** - a React 19+ frontend for the AI-powered camera traffic monitoring system. The dashboard provides real-time traffic analytics, camera monitoring, and alert management with internationalization support and a comprehensive design system.

## Development Commands

### Environment Setup & Development

```bash
# Install dependencies (using Yarn)
yarn install

# Start development server with Turbopack
yarn dev

# Start development server on specific port
yarn dev --port 3002

# Build for production
yarn build

# Start production server
yarn start

# Run linting
yarn lint

# Start development with API mock server
yarn dev:api

# Start both frontend and API mock server
yarn dev:full
```

### Design System & Components

The project uses a comprehensive design system with Tabler Icons and custom color palette. Key commands for design system work:

```bash
# Design system colors are defined in app/globals.css
# Tabler Icons are imported from @tabler/icons-react
# Refer to docs/DESIGN_SYSTEM_IMPLEMENTATION.md for usage guidelines
```

## Architecture Overview

### Technology Stack

- **Framework**: Next.js 15.4.6 with App Router and Turbopack
- **React**: React 19+ with concurrent features (useOptimistic, useDeferredValue, useTransition)
- **TypeScript**: Strict mode with comprehensive type safety
- **Styling**: Tailwind CSS 4.x with custom design system
- **UI Components**: Radix UI primitives with custom styling
- **Icons**: Tabler Icons React (@tabler/icons-react)
- **Internationalization**: next-intl with English/Vietnamese support
- **State Management**: Zustand for global state, TanStack Query for server state
- **Charts & Visualization**: Recharts, D3.js
- **Animations**: Framer Motion

### Core Architecture Patterns

#### Next.js App Router Structure

```
app/
├── [locale]/                 # Internationalized routes
│   ├── cameras/page.tsx     # Camera monitoring dashboard
│   ├── dashboard/page.tsx   # Main overview dashboard  
│   ├── analytics/page.tsx   # Traffic analytics
│   └── layout.tsx          # Locale-specific layout
├── globals.css             # Design system CSS variables
└── layout.tsx             # Root layout with providers
```

#### Component Organization

```
components/
├── ui/                     # Reusable UI primitives (Radix-based)
├── layout/                 # Layout components (header, sidebar)
├── features/              # Feature-specific components
│   ├── camera/           # Camera-related components
│   ├── dashboard/        # Dashboard widgets
│   └── alerts/           # Alert management
├── analytics/            # Analytics visualizations
├── common/              # Shared utilities (ErrorBoundary)
└── providers/           # Context providers
```

#### Design System Implementation

- **Color System**: CSS custom properties in HSL format with semantic naming
- **Icon System**: Tabler Icons with consistent sizing and accessibility
- **Typography**: Inter font with responsive scaling
- **Spacing**: Tailwind-based system with mobile-first responsive design
- **Components**: Radix UI primitives with custom Tailwind styling

### State Management Patterns

#### React 19+ Concurrent Features

```tsx
// Optimistic updates for real-time data
const [optimisticState, updateOptimistic] = useOptimistic(state, reducer);

// Deferred values for search/filtering
const deferredQuery = useDeferredValue(searchQuery);

// Transitions for non-blocking updates
const [isPending, startTransition] = useTransition();
```

#### Error Boundaries & Recovery

```tsx
// Comprehensive error handling with recovery mechanisms
<ErrorBoundary level="component">
  <Suspense fallback={<LoadingSkeleton />}>
    <ComponentWithRealTimeData />
  </Suspense>
</ErrorBoundary>
```

### Internationalization Architecture

- **Routing**: Locale-based routing with `[locale]` dynamic segments
- **Content**: JSON-based translations in `messages/` directory
- **Middleware**: Automatic locale detection and redirection
- **Components**: next-intl hooks for translations and formatting

## Development Guidelines

### Component Development Standards

#### Design System Compliance

- **ALWAYS use design system colors**: `hsl(var(--primary))` instead of hardcoded colors
- **ALWAYS use Tabler Icons**: Import from `@tabler/icons-react` with `Icon` prefix
- **ALWAYS use responsive design**: Mobile-first approach with proper breakpoints
- **ALWAYS include accessibility**: ARIA labels, semantic HTML, keyboard navigation

#### React 19+ Patterns

```tsx
// Use concurrent features for better UX
const [isPending, startTransition] = useTransition();
const deferredValue = useDeferredValue(expensiveValue);
const [optimisticState, updateOptimistic] = useOptimistic(state, updateFn);

// Proper error boundaries for resilience  
<ErrorBoundary level="component">
  <Suspense fallback={<SkeletonLoader />}>
    <RealTimeComponent />
  </Suspense>
</ErrorBoundary>
```

#### TypeScript Standards

- Use strict TypeScript configuration
- Define proper interfaces for all props and data structures
- Leverage Zod for runtime type validation
- Use proper generic types for reusable components

### Styling & Design System

#### CSS Custom Properties Usage

```css
/* Use design system variables */
color: hsl(var(--primary));
background: hsl(var(--secondary-light));
border: 1px solid hsl(var(--border));

/* Status indicators */
color: hsl(var(--online));    /* Green for operational */
color: hsl(var(--offline));   /* Red for disconnected */
color: hsl(var(--maintenance)); /* Yellow for maintenance */
```

#### Responsive Design Patterns

```tsx
// Mobile-first responsive classes
<div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-6">
<h1 className="text-lg sm:text-xl lg:text-2xl">
<div className="p-4 sm:p-6 lg:p-8">
```

#### Icon Usage Standards

```tsx
// Tabler Icons with accessibility
import { IconCamera, IconAlertTriangle } from '@tabler/icons-react';

// Decorative icons
<IconCamera className="h-4 w-4 text-primary" aria-hidden="true" />

// Meaningful icons  
<IconAlertTriangle className="h-5 w-5 text-destructive" aria-label="Critical alert" />
```

### Accessibility Requirements (WCAG 2.1 AA)

#### Semantic HTML & ARIA

```tsx
// Proper semantic structure
<header role="banner">
<main id="main-content">
<section aria-label="Camera statistics">

// ARIA labels for complex interactions
<button aria-label="Export camera data and analytics report">
<input aria-label="Search cameras by name or location" role="searchbox">
```

#### Skip Navigation & Focus Management

```tsx
// Skip link for keyboard users
<a href="#main-content" className="sr-only focus:not-sr-only">
  Skip to main content
</a>

// Proper focus indicators
<button className="focus:outline-none focus:ring-2 focus:ring-primary">
```

### Internationalization Patterns

#### Component Translation

```tsx
import { useTranslations } from 'next-intl';

function Component() {
  const t = useTranslations('Dashboard');
  return <h1>{t('title')}</h1>;
}
```

#### Locale-Aware Formatting

```tsx
import { useLocalizedFormat } from '@/components/ui/language-switcher';

const { formatDateTime, formatNumber } = useLocalizedFormat();
```

### Performance Optimization

#### Real-Time Data Handling

- Use `useOptimistic` for immediate UI feedback
- Implement `useDeferredValue` for expensive filtering operations
- Utilize `useTransition` for non-blocking state updates
- Proper memoization with `React.memo` and `useMemo`

#### Bundle Optimization

- Tree-shaking enabled for Tabler Icons
- Dynamic imports for heavy components
- Proper code splitting with Next.js
- Optimized image loading with Next.js Image component

## File Structure & Conventions

### Import Organization (via isort/eslint)

```tsx
// 1. React & Next.js imports
import React, { useState, useCallback } from 'react';
import { useTranslations } from 'next-intl';

// 2. Third-party libraries
import { IconCamera } from '@tabler/icons-react';

// 3. Internal components & utilities
import { Button } from '@/components/ui/button';
import { useOptimisticCameraState } from '@/hooks/useOptimisticCameraState';
```

### Component File Structure

```tsx
'use client'; // Only when needed for client components

// Types & interfaces
interface ComponentProps {
  // ...
}

// Constants & utilities
const COMPONENT_CONSTANTS = {
  // ...
};

// Main component with proper TypeScript
export const Component: React.FC<ComponentProps> = ({ ...props }) => {
  // Component implementation
};

// Default export for page components
export default Component;
```

## Critical Implementation Notes

### Design System Compliance

- **NEVER use hardcoded colors** - always use CSS custom properties
- **NEVER mix icon libraries** - use only Tabler Icons throughout
- **NEVER skip responsive design** - implement mobile-first approach
- **NEVER ignore accessibility** - include proper ARIA attributes

### Performance Requirements

- **Real-time updates**: Must handle frequent data updates without blocking UI
- **Responsive performance**: Components must work smoothly across all device sizes
- **Bundle efficiency**: Proper tree-shaking and code splitting implementation
- **Accessibility performance**: Screen reader compatibility without performance degradation

### File Management

- **NEVER create files unless absolutely necessary** for achieving the goal
- **ALWAYS prefer editing existing files** to creating new ones
- **NEVER proactively create documentation files** (*.md) unless explicitly requested
- **ALWAYS follow established file organization** patterns

### Testing & Quality

- Write tests for complex React 19+ patterns (useOptimistic, transitions)
- Test accessibility features with screen readers
- Verify responsive design across breakpoints
- Validate internationalization with different locales
- Test real-time data updates and error recovery

## Browser Support

- Chrome/Chromium 90+
- Firefox 88+  
- Safari 14+
- Edge 90+
- Mobile Safari 14+
- Chrome Mobile 90+

## Integration Points

- **Backend API**: FastAPI server with WebSocket support for real-time data
- **Authentication**: JWT-based with role-based access control
- **Real-time Events**: Server-Sent Events (SSE) for live camera data
- **Internationalization**: Dynamic locale switching with persistent preferences