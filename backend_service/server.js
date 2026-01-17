const { createServer } = require('http');
const { parse } = require('url');
const next = require('next');
const { spawn } = require('child_process');
const path = require('path');

const dev = process.env.NODE_ENV !== 'production';
const app = next({ dev });
const handle = app.getRequestHandler();

// Start FastAPI backend
const backendPath = path.join(__dirname, '..', 'api');
const backend = spawn('python', ['-m', 'uvicorn', 'main:app', '--port', '8001'], {
  cwd: backendPath,
  shell: true,
  stdio: 'inherit',
  env: {
    ...process.env,
    PYTHONUNBUFFERED: '1',
  },
});

backend.on('error', (err) => {
  console.error('Failed to start backend:', err);
  process.exit(1);
});

// Start Next.js server
app.prepare().then(() => {
  const server = createServer(async (req, res) => {
    const parsedUrl = parse(req.url, true);
    const { pathname } = parsedUrl;

    // Proxy API requests to FastAPI
    if (pathname.startsWith('/api/chat')) {
      const targetUrl = new URL(req.url, `http://localhost:8001`);
      const proxyReq = createServer({
        hostname: 'localhost',
        port: 8001,
        path: targetUrl.pathname.replace(/^\/api/, ''),
        method: req.method,
        headers: req.headers,
      }, (proxyRes) => {
        res.writeHead(proxyRes.statusCode, proxyRes.headers);
        proxyRes.pipe(res, { end: true });
      });

      req.pipe(proxyReq, { end: true });
      return;
    }

    // Handle Next.js pages
    handle(req, res, parsedUrl);
  });

  const PORT = process.env.PORT || 3000;
  server.listen(PORT, (err) => {
    if (err) throw err;
    console.log(`> Ready on http://localhost:${PORT}`);
  });

  // Cleanup on exit
  process.on('SIGTERM', () => {
    console.log('Shutting down...');
    backend.kill();
    server.close(() => {
      console.log('Server closed');
      process.exit(0);
    });
  });
});
