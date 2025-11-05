import './globals.css'
import type { Metadata } from 'next'
import Link from 'next/link'
import { ReactNode } from 'react'

export const metadata: Metadata = {
  title: 'Recomendador Agrícola 2030',
  description: 'Herramienta para recomendar cultivos y proyectar producción alineada con la meta 2030.'
}

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="es" suppressHydrationWarning>
      <head />
      <body className="antialiased">
        <header className="w-full border-b border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900">
          <nav className="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between">
            <Link href="/" className="text-lg font-semibold text-primary-dark">
              Recomendador Agrícola
            </Link>
            <div className="space-x-4">
              <Link href="/ingreso" className="hover:underline">
                Ingreso de datos
              </Link>
              <Link href="/acerca" className="hover:underline">
                Acerca de
              </Link>
            </div>
          </nav>
        </header>
        <main className="max-w-6xl mx-auto px-4 py-6">
          {children}
        </main>
        <footer className="max-w-6xl mx-auto px-4 py-4 border-t border-gray-200 dark:border-gray-700 text-sm text-gray-500">
          © {new Date().getFullYear()} Recomendador Agrícola — Meta 2030
        </footer>
      </body>
    </html>
  )
}