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

        // ITS Camera AI Brand Colors - Complete Palette
        'orange-peel': {
          DEFAULT: '#ff9f1c',
          100: '#382100',
          200: '#704100',
          300: '#a86200',
          400: '#e08300',
          500: '#ff9f1c',
          600: '#ffb347',
          700: '#ffc675',
          800: '#ffd9a3',
          900: '#ffecd1',
        },
        'hunyadi-yellow': {
          DEFAULT: '#ffbf69',
          100: '#482900',
          200: '#915200',
          300: '#d97b00',
          400: '#ffa023',
          500: '#ffbf69',
          600: '#ffcc89',
          700: '#ffd9a6',
          800: '#ffe5c4',
          900: '#fff2e1',
        },
        'mint-green': {
          DEFAULT: '#cbf3f0',
          100: '#114844',
          200: '#229088',
          300: '#3ad1c7',
          400: '#81e2db',
          500: '#cbf3f0',
          600: '#d4f5f3',
          700: '#dff7f6',
          800: '#eafaf9',
          900: '#f4fcfc',
        },
        'light-sea-green': {
          DEFAULT: '#2ec4b6',
          100: '#092724',
          200: '#124e48',
          300: '#1b746c',
          400: '#249b8f',
          500: '#2ec4b6',
          600: '#50d6c9',
          700: '#7ce0d6',
          800: '#a7eae4',
          900: '#d3f5f1',
        },
        'white': {
          DEFAULT: '#ffffff',
          100: '#333333',
          200: '#666666',
          300: '#999999',
          400: '#cccccc',
          500: '#ffffff',
          600: '#ffffff',
          700: '#ffffff',
          800: '#ffffff',
          900: '#ffffff',
        },

        // Semantic Color Mapping
        primary: {
          DEFAULT: "hsl(var(--primary))", // Orange Peel #ff9f1c
          foreground: "hsl(var(--primary-foreground))",
          hover: "hsl(var(--primary-hover))",
          light: "hsl(var(--primary-light))",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary))", // Light Sea Green #2ec4b6
          foreground: "hsl(var(--secondary-foreground))",
          hover: "hsl(var(--secondary-hover))",
          light: "hsl(var(--secondary-light))",
        },
        accent: {
          DEFAULT: "hsl(var(--accent))", // Hunyadi Yellow #ffbf69
          foreground: "hsl(var(--accent-foreground))",
          hover: "hsl(var(--accent-hover))",
          light: "hsl(var(--accent-light))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))", // Mint Green #cbf3f0
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
