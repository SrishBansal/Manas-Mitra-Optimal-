import type { NextApiRequest, NextApiResponse } from 'next';
import httpProxy from 'http-proxy';

const API_URL = process.env.BACKEND_URL || 'http://localhost:8001';
const proxy = httpProxy.createProxyServer();

// Disable default body parser to allow streaming
// @ts-ignore
// export const config = {
//   api: {
//     bodyParser: false,
//   },
// };

export default function handler(req: NextApiRequest, res: NextApiResponse) {
  // Set CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

  // Handle preflight requests
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  // Proxy the request to the FastAPI backend
  return new Promise((resolve, reject) => {
    proxy.web(
      req,
      res,
      { 
        target: API_URL,
        changeOrigin: true,
        pathRewrite: {
          '^/api': '', // Remove /api prefix when forwarding to backend
        },
      },
      (err) => {
        if (err) {
          console.error('Proxy error:', err);
          return reject(err);
        }
        resolve(true);
      }
    );
  });
}
