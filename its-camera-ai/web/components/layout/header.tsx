'use client';

import {
  IconCamera,
  IconBell,
  IconSettings,
  IconUser,
  IconActivity
} from '@tabler/icons-react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';

export function Header() {
  return (
    <>
      {/* Skip Navigation Link */}
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 focus:z-50 focus:px-4 focus:py-2 focus:bg-primary focus:text-white focus:rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-foreground"
      >
        Skip to main content
      </a>
      <header className="bg-card border-b border-border/50 shadow-soft sticky top-0 z-50 backdrop-blur-sm" role="banner">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center py-4">
          {/* Brand Section */}
          <div className="flex items-center gap-2 sm:gap-4 min-w-0">
            <div className="flex items-center gap-2 sm:gap-3">
              <div className="p-2 sm:p-2.5 rounded-xl bg-primary/10 border border-primary/20">
                <IconCamera className="h-5 w-5 sm:h-6 sm:w-6 text-primary" />
              </div>
              <div className="min-w-0">
                <h1 className="text-lg sm:text-xl font-bold text-foreground tracking-tight truncate">
                  ITS Camera AI
                </h1>
                <p className="hidden sm:block text-sm text-muted-foreground font-medium">
                  Real-time Traffic Monitoring System
                </p>
              </div>
            </div>

            {/* System Status */}
            <div className="hidden md:flex items-center gap-2 ml-6" role="status" aria-label="System status">
              <div className="status-indicator">
                <div className="status-dot status-dot-online" aria-hidden="true" />
                <span className="text-online font-medium">System Online</span>
              </div>
            </div>
          </div>

          {/* Actions Section */}
          <div className="flex items-center gap-1 sm:gap-3 shrink-0">
            {/* Notifications */}
            <Button
              variant="ghost"
              size="sm"
              className="relative hover:bg-muted"
              aria-label="Notifications, 3 unread"
            >
              <IconBell className="h-4 w-4" aria-hidden="true" />
              <Badge className="absolute -top-1 -right-1 h-4 w-4 sm:h-5 sm:w-5 text-2xs p-0 bg-primary border-2 border-card flex items-center justify-center" aria-label="3 unread notifications">
                3
              </Badge>
            </Button>

            {/* Settings */}
            <Button
              variant="ghost"
              size="sm"
              className="hover:bg-muted"
              aria-label="Open settings"
            >
              <IconSettings className="h-4 w-4" aria-hidden="true" />
            </Button>

            {/* Performance Indicator */}
            <div className="hidden lg:flex items-center gap-2 px-3 py-1.5 bg-secondary-light rounded-lg border border-secondary/20" role="status" aria-label="System performance">
              <IconActivity className="h-4 w-4 text-secondary" aria-hidden="true" />
              <span className="text-sm font-medium text-secondary-foreground">98.5% Uptime</span>
            </div>

            {/* User Profile */}
            <div className="flex items-center gap-2 sm:gap-3 pl-2 sm:pl-3 border-l border-border">
              <div className="hidden md:block text-right">
                <p className="text-sm font-medium text-foreground">Admin User</p>
                <p className="text-2xs text-muted-foreground">System Administrator</p>
              </div>
              <div className="relative">
                <div className="w-8 h-8 sm:w-9 sm:h-9 bg-gradient-to-br from-primary to-secondary rounded-xl flex items-center justify-center text-white text-sm font-semibold shadow-soft">
                  A
                </div>
                <div className="absolute -bottom-0.5 -right-0.5 w-2.5 h-2.5 sm:w-3 sm:h-3 bg-online border-2 border-card rounded-full" />
              </div>
            </div>
          </div>
        </div>
      </div>
    </header>
    </>
  );
}
