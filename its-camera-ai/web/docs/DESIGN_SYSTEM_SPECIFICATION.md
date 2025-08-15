# ITS Camera AI - Design System Specification

## Overview

This document defines the comprehensive design system for the ITS Camera AI dashboard, emphasizing minimalist design principles, professional monitoring aesthetics, and excellent user experience for 24/7 operations.

## Color Palette

### Primary Brand Colors

```css
/* Orange Peel - Primary Actions & Alerts */
--primary: 33 100% 55% /* #ff9f1c */
--primary-foreground: 0 0% 100%
--primary-hover: 33 100% 48%
--primary-light: 33 100% 90%

/* Light Sea Green - Success States & Positive Metrics */
--secondary: 177 58% 47% /* #2ec4b6 */
--secondary-foreground: 0 0% 100%
--secondary-hover: 177 58% 40%
--secondary-light: 177 58% 90%

/* Hunyadi Yellow - Warnings & Highlights */
--accent: 37 100% 71% /* #ffbf69 */
--accent-foreground: 220 13% 18%
--accent-hover: 37 100% 64%
--accent-light: 37 100% 95%

/* Mint Green - Subtle Backgrounds & Cards */
--muted: 168 53% 93% /* #cbf3f0 */
--muted-foreground: 220 13% 45%
--muted-hover: 168 53% 88%

/* White - Main Backgrounds */
--background: 0 0% 100% /* #ffffff */
--foreground: 220 13% 18% /* #2a2d3a */
```

### Functional Status Colors

```css
/* System Status Indicators */
--online: 142 71% 45% /* Green - Operational */
--offline: 0 84% 60% /* Red - Disconnected */
--maintenance: 45 93% 47% /* Yellow - Under Maintenance */
--critical: 0 84% 60% /* Red - Critical Alerts */

/* Semantic Status Colors */
--success: 177 58% 47% /* Light Sea Green */
--warning: 37 100% 71% /* Hunyadi Yellow */
--destructive: 33 100% 55% /* Orange Peel for critical alerts */
```

## Typography

### Font Family
- **Primary**: Inter (web font)
- **Fallbacks**: var(--font-geist-sans), system-ui, sans-serif
- **Features**: cv11, ss01, tabular-nums

### Typography Scale

```css
/* Heading Hierarchy */
h1: 2.5rem (40px) - font-weight: 700 - letter-spacing: -0.04em
h2: 2rem (32px) - font-weight: 650 - letter-spacing: -0.03em
h3: 1.5rem (24px) - font-weight: 600 - letter-spacing: -0.02em
h4: 1.25rem (20px) - font-weight: 600
h5: 1.125rem (18px) - font-weight: 550
h6: 1rem (16px) - font-weight: 500

/* Body Text */
text-base: 1rem (16px) - line-height: 1.6
text-sm: 0.875rem (14px) - line-height: 1.25
text-xs: 0.75rem (12px) - line-height: 1
text-2xs: 0.6875rem (11px) - line-height: 1
```

### Typography Guidelines
- Use `font-variant-numeric: tabular-nums` for numerical data
- Apply `-webkit-font-smoothing: antialiased` for crisp rendering
- Maintain consistent line heights for optimal readability
- Use letter-spacing sparingly for headings only

## Spacing System

### Base Spacing Scale
```css
/* Tailwind-based spacing */
0.25rem (4px) - gap-1, p-1
0.5rem (8px) - gap-2, p-2
0.75rem (12px) - gap-3, p-3
1rem (16px) - gap-4, p-4
1.5rem (24px) - gap-6, p-6
2rem (32px) - gap-8, p-8
```

### Component-Specific Spacing
- **Cards**: 1.5rem (24px) padding
- **Buttons**: 0.625rem × 1.25rem (10px × 20px) padding
- **Form Elements**: 0.75rem (12px) internal padding
- **Grid Gaps**: 1rem - 1.5rem (16px - 24px)

## Component Specifications

### Cards

```css
/* Base Card Styling */
.card {
  background: hsl(var(--card));
  border: 1px solid hsl(var(--border));
  border-radius: 0.75rem; /* 12px */
  box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.03);
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}

/* Enhanced Cards */
.card-elevated {
  box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.05);
}

.card-interactive:hover {
  transform: translateY(-1px);
  box-shadow: 0 8px 25px -8px rgb(0 0 0 / 0.1);
}
```

### Buttons

```css
/* Primary Button */
.btn-primary {
  background: hsl(var(--primary));
  color: hsl(var(--primary-foreground));
  border: 1px solid hsl(var(--primary));
  padding: 0.625rem 1.25rem;
  border-radius: calc(var(--radius) - 2px);
  font-weight: 500;
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}

.btn-primary:hover {
  background: hsl(var(--primary-hover));
}

/* Secondary Button */
.btn-secondary {
  background: hsl(var(--secondary));
  color: hsl(var(--secondary-foreground));
}

/* Ghost Button */
.btn-ghost {
  background: transparent;
  color: hsl(var(--foreground));
  border: 1px solid transparent;
}

.btn-ghost:hover {
  background: hsl(var(--muted));
}
```

### Status Indicators

```css
/* Status Dots */
.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
}

.status-dot-online {
  background: hsl(var(--online));
  box-shadow: 0 0 0 2px hsl(var(--online) / 0.2);
}

.status-dot-critical {
  background: hsl(var(--critical));
  animation: status-alert 2s ease-in-out infinite;
}

/* Status Animations */
@keyframes status-alert {
  0%, 100% { background: hsl(var(--destructive)); }
  50% { background: hsl(var(--critical)); }
}
```

## Layout Guidelines

### Grid Systems

```css
/* Data Grid - Responsive Cards */
.data-grid {
  display: grid;
  gap: 1rem;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
}

/* Metrics Grid - Smaller Cards */
.metrics-grid {
  display: grid;
  gap: 1rem;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
}

/* Camera Grid - Video Cards */
.camera-grid {
  display: grid;
  gap: 1.5rem;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
}
```

### Responsive Breakpoints

```css
/* Mobile First Approach */
@media (min-width: 640px) { /* sm */ }
@media (min-width: 768px) { /* md */ }
@media (min-width: 1024px) { /* lg */ }
@media (min-width: 1280px) { /* xl */ }
@media (min-width: 1536px) { /* 2xl */ }
```

## Data Visualization

### Metrics Display

```css
/* Metric Values */
.metric-value {
  font-size: 2rem;
  font-weight: 700;
  line-height: 1;
  font-variant-numeric: tabular-nums;
  color: hsl(var(--foreground));
}

/* Metric Labels */
.metric-label {
  font-size: 0.875rem;
  font-weight: 500;
  color: hsl(var(--muted-foreground));
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

/* Trend Indicators */
.metric-change-positive { color: hsl(var(--online)); }
.metric-change-negative { color: hsl(var(--offline)); }
.metric-change-neutral { color: hsl(var(--muted-foreground)); }
```

### Chart Containers

```css
.chart-container {
  background: hsl(var(--card));
  border: 1px solid hsl(var(--border));
  border-radius: var(--radius);
  padding: 1.5rem;
}
```

## Accessibility Standards

### WCAG 2.1 AA Compliance

1. **Color Contrast Ratios**:
   - Normal text: 4.5:1 minimum
   - Large text: 3:1 minimum
   - UI components: 3:1 minimum

2. **Interactive Elements**:
   - Minimum touch target: 44px × 44px
   - Focus indicators: 2px solid outline
   - Keyboard navigation support

3. **Motion Preferences**:
   ```css
   @media (prefers-reduced-motion: reduce) {
     .animate-* { animation: none; }
     .transition { transition: none; }
   }
   ```

4. **High Contrast Support**:
   ```css
   @media (prefers-contrast: high) {
     .card { border-width: 2px; }
     .btn { border-width: 2px; }
   }
   ```

## Animation Guidelines

### Timing Functions
- **Default**: `cubic-bezier(0.4, 0, 0.2, 1)` (ease-out)
- **Bounce**: `cubic-bezier(0.68, -0.55, 0.265, 1.55)`
- **Sharp**: `cubic-bezier(0.4, 0, 0.6, 1)`

### Duration Standards
- **Micro-interactions**: 150ms - 200ms
- **Component transitions**: 200ms - 300ms
- **Page transitions**: 300ms - 500ms
- **Status animations**: 1s - 2s (infinite)

### Key Animations

```css
/* Fade In */
@keyframes fade-in {
  from { opacity: 0; transform: translateY(4px); }
  to { opacity: 1; transform: translateY(0); }
}

/* Scale In */
@keyframes scale-in {
  from { transform: scale(0.95); opacity: 0; }
  to { transform: scale(1); opacity: 1; }
}

/* Loading Shimmer */
@keyframes shimmer {
  0% { background-position: -200% 0; }
  100% { background-position: 200% 0; }
}
```

## Implementation Guidelines

### CSS Custom Properties
All colors must use CSS custom properties for theme consistency:
```css
/* ✅ Correct */
color: hsl(var(--primary));

/* ❌ Incorrect */
color: #ff9f1c;
```

### Tailwind CSS Classes
Prefer Tailwind utility classes for consistent spacing and sizing:
```tsx
// ✅ Correct
<div className="p-4 rounded-xl border border-border/50">

// ❌ Incorrect
<div style={{ padding: '16px', borderRadius: '12px' }}>
```

### Component Composition
Build components using the established design tokens:
```tsx
// ✅ Correct
<Button variant="primary" size="sm">
  Save Changes
</Button>

// ❌ Incorrect
<button className="bg-orange-500 text-white px-3 py-1">
  Save Changes
</button>
```

## Monitoring-Specific Patterns

### Critical Information Hierarchy
1. **Status Indicators**: Most prominent (color + animation)
2. **Key Metrics**: Large, tabular numbers
3. **Secondary Data**: Muted colors, smaller text
4. **Actions**: Clear call-to-action buttons

### Real-time Data Display
- Use `font-variant-numeric: tabular-nums` for consistent number alignment
- Apply subtle animations for data updates
- Maintain visual stability during frequent updates
- Use color coding for quick status recognition

### Emergency States
- Animated status indicators for critical alerts
- High contrast colors for urgent actions
- Clear visual hierarchy for emergency information
- Accessible color combinations for colorblind users

## Quality Assurance

### Design Review Checklist
- [ ] Color contrast meets WCAG 2.1 AA standards
- [ ] Interactive elements have proper focus states
- [ ] Typography maintains consistent hierarchy
- [ ] Spacing follows the established scale
- [ ] Animations respect motion preferences
- [ ] Components work across all breakpoints
- [ ] Status indicators are clearly distinguishable
- [ ] Data is readable in monitoring contexts

### Browser Testing
- Chrome/Chromium 90+
- Firefox 88+
- Safari 14+
- Edge 90+

### Performance Considerations
- Minimize layout shifts during data updates
- Use `transform` for animations (GPU acceleration)
- Optimize for high-frequency real-time updates
- Consider reduced motion preferences
- Implement loading states for better perceived performance

---

This design system ensures consistent, accessible, and professional user interfaces optimized for 24/7 traffic monitoring operations while maintaining excellent usability and visual appeal.