import type { Config } from "tailwindcss";

export default {
  darkMode: ["class"],
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        // Base system colors
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",

        // ITS Blue System - Professional Monitoring Colors
        'its-blue': {
          DEFAULT: '#3b82f6',
          50: '#eff6ff',
          100: '#dbeafe',
          200: '#bfdbfe',
          300: '#93c5fd',
          400: '#60a5fa',
          500: '#3b82f6',
          600: '#2563eb',
          700: '#1d4ed8',
          800: '#1e40af',
          900: '#1e3a8a',
        },
        'professional-cyan': {
          DEFAULT: '#06b6d4',
          50: '#ecfeff',
          100: '#cffafe',
          200: '#a5f3fc',
          300: '#67e8f9',
          400: '#22d3ee',
          500: '#06b6d4',
          600: '#0891b2',
          700: '#0e7490',
          800: '#155e75',
          900: '#164e63',
        },
        'neutral-slate': {
          DEFAULT: '#64748b',
          50: '#f8fafc',
          100: '#f1f5f9',
          200: '#e2e8f0',
          300: '#cbd5e1',
          400: '#94a3b8',
          500: '#64748b',
          600: '#475569',
          700: '#334155',
          800: '#1e293b',
          900: '#0f172a',
        },

        // Traffic Status Colors
        'traffic': {
          optimal: '#10b981',     // Green - free flow >60 km/h
          moderate: '#f59e0b',    // Amber - slow flow 30-60 km/h
          congested: '#ef4444',   // Red - stop-and-go <30 km/h
          blocked: '#dc2626',     // Dark red - stopped traffic
        },

        // Alert Severity Colors
        'alert': {
          info: '#06b6d4',        // Cyan - informational
          warning: '#f59e0b',     // Amber - warning
          critical: '#ef4444',    // Red - critical
          emergency: '#dc2626',   // Dark red - emergency
        },

        // AI & Performance Colors
        'ai-confidence': {
          high: '#10b981',        // Green - >90% confidence
          medium: '#f59e0b',      // Amber - 70-90% confidence
          low: '#ef4444',         // Red - <70% confidence
        },
        'performance': {
          excellent: '#10b981',   // Green - <50ms latency
          good: '#84cc16',        // Light green - 50-100ms latency
          warning: '#f59e0b',     // Amber - 100-200ms latency
          poor: '#ef4444',        // Red - >200ms latency
        },

        // Security Status Colors
        'security': {
          secure: '#10b981',      // Green - all systems secure
          monitoring: '#06b6d4',  // Cyan - active monitoring
          warning: '#f59e0b',     // Amber - potential issue
          breach: '#dc2626',      // Dark red - security breach
        },

        // Semantic Color Mapping
        primary: {
          DEFAULT: "hsl(var(--primary))", // ITS Blue #3b82f6
          foreground: "hsl(var(--primary-foreground))",
          hover: "hsl(var(--primary-hover))",
          light: "hsl(var(--primary-light))",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary))", // Professional Cyan #06b6d4
          foreground: "hsl(var(--secondary-foreground))",
          hover: "hsl(var(--secondary-hover))",
          light: "hsl(var(--secondary-light))",
        },
        accent: {
          DEFAULT: "hsl(var(--accent))", // Emerald Green #10b981
          foreground: "hsl(var(--accent-foreground))",
          hover: "hsl(var(--accent-hover))",
          light: "hsl(var(--accent-light))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))", // Neutral Slate #f1f5f9
          foreground: "hsl(var(--muted-foreground))",
          hover: "hsl(var(--muted-hover))",
        },

        // Status colors for monitoring
        success: {
          DEFAULT: "hsl(var(--success))",
          foreground: "hsl(var(--success-foreground))",
        },
        warning: {
          DEFAULT: "hsl(var(--warning))",
          foreground: "hsl(var(--warning-foreground))",
        },
        destructive: {
          DEFAULT: "hsl(var(--destructive))",
          foreground: "hsl(var(--destructive-foreground))",
        },

        // Monitoring-specific status colors
        online: "hsl(var(--online))",
        offline: "hsl(var(--offline))",
        maintenance: "hsl(var(--maintenance))",
        critical: {
          DEFAULT: "hsl(var(--critical))",
          foreground: "hsl(var(--critical-foreground))",
        },

        // UI element colors
        popover: {
          DEFAULT: "hsl(var(--popover))",
          foreground: "hsl(var(--popover-foreground))",
        },
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
        xl: "calc(var(--radius) + 4px)",
      },
      fontFamily: {
        sans: ['Inter', 'var(--font-geist-sans)', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'monospace'],
      },
      fontSize: {
        '2xs': ['0.6875rem', { lineHeight: '1rem' }],
      },
      spacing: {
        '18': '4.5rem',
        '88': '22rem',
      },
      animation: {
        'bounce-subtle': 'bounce-subtle 2s infinite',
        'fade-in': 'fade-in 0.3s ease-out',
        'scale-in': 'scale-in 0.2s ease-out',
        'status-online': 'status-online 2s ease-in-out infinite',
        'status-alert': 'status-alert 1s ease-in-out infinite',
        'shimmer': 'shimmer 1.5s infinite',
      },
      keyframes: {
        'bounce-subtle': {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-2px)' },
        },
        'fade-in': {
          'from': { opacity: '0', transform: 'translateY(4px)' },
          'to': { opacity: '1', transform: 'translateY(0)' },
        },
        'scale-in': {
          'from': { transform: 'scale(0.95)', opacity: '0' },
          'to': { transform: 'scale(1)', opacity: '1' },
        },
        'status-online': {
          '0%, 100%': { backgroundColor: 'hsl(var(--online))' },
          '50%': { backgroundColor: 'hsl(var(--online) / 0.8)' },
        },
        'status-alert': {
          '0%, 100%': { backgroundColor: 'hsl(var(--destructive))' },
          '50%': { backgroundColor: 'hsl(var(--critical))' },
        },
        'shimmer': {
          '0%': { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
      },
      boxShadow: {
        'soft': '0 2px 8px 0 rgb(0 0 0 / 0.04)',
        'medium': '0 4px 12px 0 rgb(0 0 0 / 0.08)',
        'strong': '0 8px 24px 0 rgb(0 0 0 / 0.12)',
      },
    },
  },
  plugins: [],
} satisfies Config;
