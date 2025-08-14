import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { WebSocketProvider } from '@/components/providers/websocket-provider'
import { Sidebar } from '@/components/layout/sidebar'
import { Header } from '@/components/layout/header'
import { Toaster } from '@/components/ui/toaster'

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "ITS Camera AI - Traffic Monitoring System",
  description: "AI-powered camera traffic monitoring system for real-time traffic analytics and vehicle tracking",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        <WebSocketProvider>
          <div className="min-h-screen bg-background">
            <Sidebar />
            <div className="pl-64"> {/* Sidebar width offset */}
              <Header />
              <main className="pt-16"> {/* Header height offset */}
                {children}
              </main>
            </div>
          </div>
          <Toaster />
        </WebSocketProvider>
      </body>
    </html>
  );
}
