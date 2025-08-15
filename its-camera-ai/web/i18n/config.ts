import {defineRouting} from 'next-intl/routing';
import {createNavigation} from 'next-intl/navigation';

export const routing = defineRouting({
  // A list of all locales that are supported
  locales: ['en', 'vi'],

  // Used when no locale matches
  defaultLocale: 'en',

  // The pathnames that should be localized
  pathnames: {
    '/': '/',
    '/dashboard': {
      en: '/dashboard',
      vi: '/bang-dieu-khien'
    },
    '/alerts': {
      en: '/alerts',
      vi: '/canh-bao'
    },
    '/cameras': {
      en: '/cameras',
      vi: '/camera'
    }
  },

  // Configure locale detection
  localeDetection: true,

  // Configure locale prefix strategy
  localePrefix: 'as-needed'
});

// Lightweight wrappers around Next.js' navigation APIs
// that will consider the routing configuration
export const {Link, redirect, usePathname, useRouter, getPathname} = createNavigation(routing);

// Supported locales configuration
export const locales = routing.locales;
export const defaultLocale = routing.defaultLocale;

// Locale metadata for display
export const localeConfig = {
  en: {
    name: 'English',
    nativeName: 'English',
    flag: '🇺🇸',
    rtl: false,
    dateFormat: 'MM/dd/yyyy',
    timeFormat: '12h',
    numberFormat: 'en-US'
  },
  vi: {
    name: 'Vietnamese',
    nativeName: 'Tiếng Việt',
    flag: '🇻🇳',
    rtl: false,
    dateFormat: 'dd/MM/yyyy',
    timeFormat: '24h',
    numberFormat: 'vi-VN'
  }
} as const;

export type Locale = keyof typeof localeConfig;
export type LocaleConfig = typeof localeConfig[Locale];
