'use client';

import * as React from 'react';
import { useLocale, useTranslations } from 'next-intl';
import { useRouter, usePathname } from '@/i18n/config';
import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Badge } from '@/components/ui/badge';
import { IconWorld, IconCheck } from '@tabler/icons-react';
import { localeConfig, type Locale } from '@/i18n/config';
import { cn } from '@/lib/utils';

interface LanguageSwitcherProps {
  className?: string;
  variant?: 'default' | 'minimal' | 'flag-only';
  showLabel?: boolean;
}

export const LanguageSwitcher = React.memo(({
  className,
  variant = 'default',
  showLabel = true
}: LanguageSwitcherProps) => {
  const t = useTranslations('LanguageSwitcher');
  const locale = useLocale() as Locale;
  const router = useRouter();
  const pathname = usePathname();
  const [isPending, startTransition] = React.useTransition();

  const currentLocaleConfig = React.useMemo(() => localeConfig[locale], [locale]);

  const localeEntries = React.useMemo(() =>
    Object.entries(localeConfig),
    []
  );

  const handleLocaleChange = React.useCallback((newLocale: Locale) => {
    if (newLocale === locale) return; // Prevent unnecessary navigation

    startTransition(() => {
      router.replace(pathname, { locale: newLocale });
    });
  }, [locale, router, pathname]);

  if (variant === 'flag-only') {
    return (
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            variant="ghost"
            size="sm"
            className={cn("h-8 w-8 p-0", className)}
            disabled={isPending}
          >
            <span className="text-lg" role="img" aria-label={currentLocaleConfig.nativeName}>
              {currentLocaleConfig.flag}
            </span>
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-48">
          {localeEntries.map(([code, config]) => (
            <DropdownMenuItem
              key={code}
              onClick={() => handleLocaleChange(code as Locale)}
              className="flex items-center justify-between"
            >
              <div className="flex items-center gap-2">
                <span className="text-base" role="img" aria-label={config.nativeName}>
                  {config.flag}
                </span>
                <span className={cn(
                  "text-sm",
                  config.rtl && "font-arabic" // You'd need to add this CSS class
                )}>
                  {config.nativeName}
                </span>
              </div>
              {locale === code && (
                <IconCheck className="h-4 w-4 text-blue-600" />
              )}
            </DropdownMenuItem>
          ))}
        </DropdownMenuContent>
      </DropdownMenu>
    );
  }

  if (variant === 'minimal') {
    return (
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            variant="outline"
            size="sm"
            className={cn("gap-2", className)}
            disabled={isPending}
          >
            <IconWorld className="h-4 w-4" />
            <span className="uppercase font-mono text-xs">
              {locale}
            </span>
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-48">
          {localeEntries.map(([code, config]) => (
            <DropdownMenuItem
              key={code}
              onClick={() => handleLocaleChange(code as Locale)}
              className="flex items-center justify-between"
            >
              <div className="flex items-center gap-2">
                <span className="text-base" role="img" aria-label={config.nativeName}>
                  {config.flag}
                </span>
                <div className="flex flex-col">
                  <span className="text-sm font-medium">{config.name}</span>
                  <span className={cn(
                    "text-xs text-gray-500",
                    config.rtl && "font-arabic"
                  )}>
                    {config.nativeName}
                  </span>
                </div>
              </div>
              {locale === code && (
                <IconCheck className="h-4 w-4 text-blue-600" />
              )}
            </DropdownMenuItem>
          ))}
        </DropdownMenuContent>
      </DropdownMenu>
    );
  }

  return (
    <div className={cn("flex items-center gap-2", className)}>
      {showLabel && (
        <span className="text-sm text-gray-600 dark:text-gray-400">
          {t('currentLanguage')}:
        </span>
      )}

      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            variant="outline"
            className="gap-2 min-w-[120px]"
            disabled={isPending}
          >
            <span className="text-base" role="img" aria-label={currentLocaleConfig.nativeName}>
              {currentLocaleConfig.flag}
            </span>
            <span className={cn(
              "text-sm",
              currentLocaleConfig.rtl && "font-arabic"
            )}>
              {currentLocaleConfig.nativeName}
            </span>
          </Button>
        </DropdownMenuTrigger>

        <DropdownMenuContent align="end" className="w-64">
          <div className="px-2 py-1.5 text-sm font-semibold text-gray-700 dark:text-gray-200">
            {t('selectLanguage')}
          </div>
          <div className="border-t my-1" />

          {localeEntries.map(([code, config]) => (
            <DropdownMenuItem
              key={code}
              onClick={() => handleLocaleChange(code as Locale)}
              className="flex items-center justify-between p-3 focus:bg-blue-50 dark:focus:bg-blue-900"
            >
              <div className="flex items-center gap-3">
                <span className="text-lg" role="img" aria-label={config.nativeName}>
                  {config.flag}
                </span>
                <div className="flex flex-col">
                  <span className="text-sm font-medium">{config.name}</span>
                  <span className={cn(
                    "text-xs text-gray-500 dark:text-gray-400",
                    config.rtl && "font-arabic"
                  )}>
                    {config.nativeName}
                  </span>
                </div>
              </div>

              <div className="flex items-center gap-2">
                {locale === code && (
                  <IconCheck className="h-4 w-4 text-blue-600" />
                )}
              </div>
            </DropdownMenuItem>
          ))}
        </DropdownMenuContent>
      </DropdownMenu>

      {isPending && (
        <div className="h-4 w-4 animate-spin rounded-full border-2 border-gray-300 border-t-blue-600" />
      )}
    </div>
  );
});

LanguageSwitcher.displayName = 'LanguageSwitcher';

// Helper hook for getting localized date/time formatting with caching
export function useLocalizedFormat() {
  const locale = useLocale() as Locale;
  const config = React.useMemo(() => localeConfig[locale], [locale]);

  // Cache formatters for better performance
  const formatters = React.useMemo(() => ({
    dateFormatter: new Intl.DateTimeFormat(config.numberFormat, {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    }),
    timeFormatter: new Intl.DateTimeFormat(config.numberFormat, {
      hour: 'numeric',
      minute: '2-digit',
      hour12: config.timeFormat === '12h'
    }),
    dateTimeFormatter: new Intl.DateTimeFormat(config.numberFormat, {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
      hour12: config.timeFormat === '12h'
    }),
    numberFormatter: new Intl.NumberFormat(config.numberFormat),
    percentageFormatter: new Intl.NumberFormat(config.numberFormat, {
      style: 'percent',
      maximumFractionDigits: 1
    })
  }), [config]);

  return React.useMemo(() => ({
    formatDate: (date: Date) => formatters.dateFormatter.format(date),
    formatTime: (date: Date) => formatters.timeFormatter.format(date),
    formatDateTime: (date: Date) => formatters.dateTimeFormatter.format(date),
    formatNumber: (num: number) => formatters.numberFormatter.format(num),
    formatPercentage: (num: number) => formatters.percentageFormatter.format(num),
    locale: config.numberFormat
  }), [formatters, config.numberFormat]);
}

export default LanguageSwitcher;
