'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { cn } from '@/lib/utils'
import { Badge } from '@/components/ui/badge'
import {
  LayoutDashboard, Camera, BarChart3, Brain, Settings,
  AlertTriangle, Users, Shield, FileText, Activity
} from 'lucide-react'

interface NavItem {
  title: string
  href: string
  icon: React.ReactNode
  badge?: string
  description?: string
}

const navItems: NavItem[] = [
  {
    title: 'Dashboard',
    href: '/dashboard',
    icon: <LayoutDashboard className="h-4 w-4" />,
    description: 'Traffic monitoring overview'
  },
  {
    title: 'Cameras',
    href: '/cameras',
    icon: <Camera className="h-4 w-4" />,
    badge: '48',
    description: 'Camera management'
  },
  {
    title: 'Analytics',
    href: '/analytics',
    icon: <BarChart3 className="h-4 w-4" />,
    description: 'Traffic analytics & reports'
  },
  {
    title: 'ML Models',
    href: '/models',
    icon: <Brain className="h-4 w-4" />,
    badge: '3',
    description: 'AI model management'
  },
  {
    title: 'Alerts',
    href: '/alerts',
    icon: <AlertTriangle className="h-4 w-4" />,
    badge: '7',
    description: 'Active alerts & incidents'
  },
  {
    title: 'Users',
    href: '/users',
    icon: <Users className="h-4 w-4" />,
    description: 'User management'
  },
  {
    title: 'Security',
    href: '/security',
    icon: <Shield className="h-4 w-4" />,
    description: 'Security & audit logs'
  },
  {
    title: 'Reports',
    href: '/reports',
    icon: <FileText className="h-4 w-4" />,
    description: 'Traffic reports'
  },
  {
    title: 'Settings',
    href: '/settings',
    icon: <Settings className="h-4 w-4" />,
    description: 'System configuration'
  }
]

export function Sidebar() {
  const pathname = usePathname()

  return (
    <div className="fixed left-0 top-0 h-full w-64 bg-card border-r border-border">
      {/* Logo/Brand */}
      <div className="p-6 border-b border-border">
        <Link href="/dashboard" className="flex items-center space-x-2">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary">
            <Activity className="h-4 w-4 text-primary-foreground" />
          </div>
          <div>
            <div className="text-lg font-semibold">ITS Camera AI</div>
            <div className="text-xs text-muted-foreground">Traffic Monitoring</div>
          </div>
        </Link>
      </div>

      {/* Navigation */}
      <nav className="p-4 space-y-1">
        {navItems.map((item) => {
          const isActive = pathname === item.href
          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                'flex items-center justify-between rounded-lg px-3 py-2 text-sm transition-colors hover:bg-accent hover:text-accent-foreground',
                isActive
                  ? 'bg-accent text-accent-foreground'
                  : 'text-muted-foreground'
              )}
            >
              <div className="flex items-center space-x-3">
                {item.icon}
                <span>{item.title}</span>
              </div>
              {item.badge && (
                <Badge variant="secondary" className="ml-auto">
                  {item.badge}
                </Badge>
              )}
            </Link>
          )
        })}
      </nav>

      {/* Status Indicator */}
      <div className="absolute bottom-4 left-4 right-4">
        <div className="rounded-lg bg-accent/50 p-3">
          <div className="flex items-center space-x-2 text-sm">
            <div className="h-2 w-2 rounded-full bg-green-500 animate-pulse" />
            <span className="text-muted-foreground">System Online</span>
          </div>
          <div className="mt-1 text-xs text-muted-foreground">
            48 cameras active • 7 alerts
          </div>
        </div>
      </div>
    </div>
  )
}