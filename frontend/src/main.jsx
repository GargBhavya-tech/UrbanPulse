import React from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import './tokens.css'

// Display + body + mono faces (observatory instrument look)
const fonts = document.createElement('link')
fonts.rel = 'stylesheet'
fonts.href =
  'https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap'
document.head.appendChild(fonts)

createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
