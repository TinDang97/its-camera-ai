# ITS Camera AI - Web Dashboard

A modern Next.js 15 web application for the ITS Camera AI traffic monitoring system.

## Features

- **Real-time Dashboard** - Monitor traffic analytics and camera feeds
- **Camera Management** - View and control multiple camera streams
- **Traffic Analytics** - Visualize vehicle detection and traffic patterns
- **Alert System** - Real-time notifications for traffic incidents
- **Dark Mode** - Support for light/dark theme
- **Responsive Design** - Works on desktop and mobile devices

## Tech Stack

- **Next.js 15** - React framework with App Router
- **TypeScript** - Type-safe development
- **Tailwind CSS** - Utility-first CSS framework
- **Lucide Icons** - Beautiful icon library
- **Recharts** - Data visualization
- **React Query** - Server state management

## Getting Started

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build

# Start production server
npm start
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

## Project Structure

```
web/
├── app/              # Next.js App Router pages
│   ├── page.tsx     # Dashboard page
│   ├── layout.tsx   # Root layout
│   └── globals.css  # Global styles
├── components/       # React components
│   └── ui/          # Reusable UI components
├── lib/             # Utilities and API client
│   ├── utils.ts     # Helper functions
│   └── api.ts       # FastAPI integration
└── public/          # Static assets
```

## API Integration

The application connects to the ITS Camera AI FastAPI backend at `http://localhost:8000`.
