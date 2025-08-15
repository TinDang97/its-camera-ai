'use client';

import React, { Component, ErrorInfo, ReactNode } from 'react';
import { IconAlertTriangle, IconRefresh, IconHome, IconBug } from '@tabler/icons-react';

interface Props {
  children: ReactNode;
  level?: 'component' | 'page' | 'critical';
  fallback?: ReactNode;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
  retryCount: number;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      retryCount: 0,
    };
  }

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    this.setState({ errorInfo });

    // Call the onError callback if provided
    if (this.props.onError) {
      this.props.onError(error, errorInfo);
    }

    // Log error for monitoring
    console.error('ErrorBoundary caught an error:', error, errorInfo);

    // In a real app, you'd send this to an error reporting service
    if (typeof window !== 'undefined') {
      // Store error info for debugging
      window.__ERROR_BOUNDARY_ERRORS__ = window.__ERROR_BOUNDARY_ERRORS__ || [];
      window.__ERROR_BOUNDARY_ERRORS__.push({
        error: error.toString(),
        errorInfo,
        timestamp: new Date().toISOString(),
        level: this.props.level || 'component',
        stack: error.stack,
      });
    }
  }

  private handleRetry = () => {
    if (this.state.retryCount >= 3) {
      // After 3 retries, reload the page
      window.location.reload();
      return;
    }

    // Exponential backoff for retries
    const delay = Math.pow(2, this.state.retryCount) * 1000;

    setTimeout(() => {
      this.setState({
        hasError: false,
        error: null,
        errorInfo: null,
        retryCount: this.state.retryCount + 1,
      });
    }, delay);
  };

  private handleReload = () => {
    window.location.reload();
  };

  private handleGoHome = () => {
    window.location.href = '/';
  };

  private getErrorLevel() {
    const { level = 'component' } = this.props;
    return level;
  }

  private renderComponentError() {
    return (
      <div className="flex flex-col items-center justify-center p-8 border border-red-200 rounded-lg bg-red-50 min-h-[200px]">
        <IconAlertTriangle className="w-12 h-12 text-red-500 mb-4" />
        <h3 className="text-lg font-semibold text-red-900 mb-2">
          Component Error
        </h3>
        <p className="text-red-700 text-center mb-4">
          This component encountered an error and couldn't render properly.
        </p>
        <div className="flex gap-2">
          <button
            onClick={this.handleRetry}
            disabled={this.state.retryCount >= 3}
            className="flex items-center gap-2 px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 transition-colors disabled:opacity-50"
          >
            <IconRefresh className="w-4 h-4" />
            {this.state.retryCount >= 3 ? 'Reloading...' : `Retry (${3 - this.state.retryCount} left)`}
          </button>
          {this.state.retryCount >= 3 && (
            <button
              onClick={this.handleReload}
              className="flex items-center gap-2 px-4 py-2 border border-red-300 text-red-700 rounded hover:bg-red-100 transition-colors"
            >
              <IconRefresh className="w-4 h-4" />
              Reload Page
            </button>
          )}
        </div>
      </div>
    );
  }

  private renderPageError() {
    return (
      <div className="min-h-screen bg-gray-50 flex flex-col items-center justify-center p-4">
        <div className="max-w-md w-full bg-white rounded-lg shadow-lg p-8 text-center">
          <div className="w-16 h-16 bg-red-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <IconAlertTriangle className="w-8 h-8 text-red-600" />
          </div>
          <h1 className="text-2xl font-bold text-gray-900 mb-2">
            Page Error
          </h1>
          <p className="text-gray-600 mb-6">
            Something went wrong while loading this page. We're working to fix the issue.
          </p>

          <div className="space-y-3">
            <button
              onClick={this.handleRetry}
              disabled={this.state.retryCount >= 3}
              className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors disabled:opacity-50"
            >
              <IconRefresh className="w-4 h-4" />
              {this.state.retryCount >= 3 ? 'Reloading...' : 'Try Again'}
            </button>

            <button
              onClick={this.handleGoHome}
              className="w-full flex items-center justify-center gap-2 px-4 py-2 border border-gray-300 text-gray-700 rounded hover:bg-gray-50 transition-colors"
            >
              <IconHome className="w-4 h-4" />
              Go to Home
            </button>
          </div>

          {process.env.NODE_ENV === 'development' && this.state.error && (
            <details className="mt-6 text-left">
              <summary className="cursor-pointer text-sm text-gray-500 hover:text-gray-700">
                <IconBug className="w-4 h-4 inline mr-1" />
                Error Details (Development)
              </summary>
              <pre className="mt-2 p-3 bg-gray-100 rounded text-xs overflow-auto text-red-700">
                {this.state.error.toString()}
                {this.state.errorInfo?.componentStack}
              </pre>
            </details>
          )}
        </div>
      </div>
    );
  }

  private renderCriticalError() {
    return (
      <div className="min-h-screen bg-red-50 flex flex-col items-center justify-center p-4">
        <div className="max-w-lg w-full bg-white rounded-lg shadow-xl p-8 text-center border-2 border-red-200">
          <div className="w-20 h-20 bg-red-100 rounded-full flex items-center justify-center mx-auto mb-6">
            <IconAlertTriangle className="w-10 h-10 text-red-600" />
          </div>
          <h1 className="text-3xl font-bold text-red-900 mb-4">
            Critical System Error
          </h1>
          <p className="text-red-700 mb-6 text-lg">
            A critical error occurred that prevents the application from functioning properly.
          </p>

          <div className="bg-red-100 border border-red-200 rounded p-4 mb-6">
            <p className="text-red-800 text-sm">
              Please refresh the page or contact support if the problem persists.
            </p>
          </div>

          <button
            onClick={this.handleReload}
            className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors text-lg font-medium"
          >
            <IconRefresh className="w-5 h-5" />
            Reload Application
          </button>

          {process.env.NODE_ENV === 'development' && this.state.error && (
            <details className="mt-6 text-left">
              <summary className="cursor-pointer text-sm text-red-600 hover:text-red-800">
                <IconBug className="w-4 h-4 inline mr-1" />
                Technical Details (Development)
              </summary>
              <pre className="mt-3 p-4 bg-gray-900 text-green-400 rounded text-xs overflow-auto max-h-40">
                {this.state.error.toString()}
                {this.state.error.stack}
              </pre>
            </details>
          )}
        </div>
      </div>
    );
  }

  render() {
    if (this.state.hasError) {
      // Use custom fallback if provided
      if (this.props.fallback) {
        return this.props.fallback;
      }

      // Render different error UIs based on the level
      switch (this.getErrorLevel()) {
        case 'component':
          return this.renderComponentError();
        case 'page':
          return this.renderPageError();
        case 'critical':
          return this.renderCriticalError();
        default:
          return this.renderComponentError();
      }
    }

    return this.props.children;
  }
}

// Convenience wrapper components
export const ComponentErrorBoundary = ({ children, onError }: { children: ReactNode; onError?: (error: Error, errorInfo: ErrorInfo) => void }) => (
  <ErrorBoundary level="component" onError={onError}>
    {children}
  </ErrorBoundary>
);

export const PageErrorBoundary = ({ children, onError }: { children: ReactNode; onError?: (error: Error, errorInfo: ErrorInfo) => void }) => (
  <ErrorBoundary level="page" onError={onError}>
    {children}
  </ErrorBoundary>
);

export const CriticalErrorBoundary = ({ children, onError }: { children: ReactNode; onError?: (error: Error, errorInfo: ErrorInfo) => void }) => (
  <ErrorBoundary level="critical" onError={onError}>
    {children}
  </ErrorBoundary>
);

// Type augmentation for window object (for development error tracking)
declare global {
  interface Window {
    __ERROR_BOUNDARY_ERRORS__?: Array<{
      error: string;
      errorInfo: ErrorInfo;
      timestamp: string;
      level: string;
      stack?: string;
    }>;
  }
}
