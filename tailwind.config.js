/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        soc: {
          bg: '#0f172a',
          card: '#1e293b',
          border: '#334155',
          text: '#f8fafc',
          muted: '#94a3b8',
          accent: '#3b82f6',
          danger: '#ef4444',
          warning: '#f59e0b',
          success: '#10b981'
        }
      }
    },
  },
  plugins: [],
}