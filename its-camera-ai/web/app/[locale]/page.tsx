'use client';

export default function Home() {
  return (
    <div className="flex items-center justify-center min-h-screen">
      <div className="text-center">
        <h1 className="text-4xl font-bold mb-4">ITS Camera AI Dashboard</h1>
        <p className="text-muted-foreground mb-6">Welcome to the ITS Camera AI monitoring system</p>
        <div className="space-y-4">
          <a
            href="/dashboard"
            className="inline-block px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            Go to Dashboard
          </a>
          <br />
          <a
            href="/cameras"
            className="inline-block px-6 py-3 border border-blue-600 text-blue-600 rounded-lg hover:bg-blue-50 transition-colors"
          >
            View Cameras
          </a>
        </div>
      </div>
    </div>
  );
}
