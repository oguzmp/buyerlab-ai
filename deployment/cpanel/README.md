# BuyerLab AI cPanel Deployment Notes

This package is prepared for uploading BuyerLab AI to a cPanel-based server.

## Important

BuyerLab AI is a Streamlit app. It is not a static HTML/PHP site.

This package also includes a Node.js Passenger wrapper:

- `passenger_app.js`
- `package.json`

Use this wrapper if your cPanel only shows the Node.js application screen.
The Node app starts the Python Streamlit app in the background and proxies web
traffic to it.

The Gemini AI integration will work on cPanel when:

- the Streamlit Python app is running as a server process
- `GEMINI_API_KEY` is set in the cPanel environment
- outbound HTTPS requests to Gemini are allowed by the hosting provider

It can run on cPanel only if the hosting plan supports one of these:

- Python applications with long-running processes
- Terminal/SSH access where a Streamlit process can stay alive
- Reverse proxy support from the public domain to the Streamlit port
- Node.js app support plus a working Python command

If the cPanel plan only supports static files or classic PHP hosting, the Streamlit app cannot start there, so Gemini cannot be called from that hosting plan. In that case, deploy BuyerLab AI on a VPS, Streamlit Community Cloud, Render, Railway, or another Python app host instead.

## Upload Steps

1. Upload `buyerlab-ai-cpanel.zip` to your cPanel file manager.
2. Extract it into your application folder.
3. Create environment variables in cPanel:
   - `GEMINI_API_KEY`
   - `GEMINI_MODEL=gemini-2.5-flash`
   - `BUYERLAB_MOCK_MODE=false`
4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. If you are using Python app / SSH, start the app:

```bash
bash deployment/cpanel/start_streamlit.sh
```

If cPanel gives you a custom port through `PORT`, the script uses it automatically.

## Node.js cPanel Screen

If you are on the cPanel Node.js screen, use:

- Application root: the folder where you extracted BuyerLab AI
- Application URL: your chosen URL path
- Application startup file: `passenger_app.js`
- Application mode: `Production`
- Node.js version: Node 18+ if available

After creating the Node app:

1. Upload/extract the package into the application root.
2. Add environment variables:
   - `GEMINI_API_KEY`
   - `GEMINI_MODEL=gemini-2.5-flash`
   - `BUYERLAB_MOCK_MODE=false`
   - Optional if Python command is different: `PYTHON_BIN=/path/to/python`
3. Install Python dependencies from terminal:

```bash
python -m pip install -r requirements.txt
```

4. Restart the Node.js app from cPanel.

If the host does not allow Node to spawn Python processes, this wrapper cannot
run Streamlit on that plan. In that case use a VPS, Render, Railway, or a Python
app host.

## Safe Demo Mode

For testing without Gemini calls:

```bash
BUYERLAB_MOCK_MODE=true bash deployment/cpanel/start_streamlit.sh
```

## Security

Do not upload `.env` with a real API key.

This package intentionally excludes `.env`, `.git`, caches, and local runtime folders.
