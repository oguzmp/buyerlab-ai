const http = require("http");
const net = require("net");
const { spawn } = require("child_process");

const passengerPort = Number(process.env.PORT || 3000);
const streamlitPort = Number(process.env.STREAMLIT_PORT || passengerPort + 1000);
const host = "127.0.0.1";

let streamlitProcess = null;

function startStreamlit() {
  if (streamlitProcess) {
    return;
  }

  const pythonCommand = process.env.PYTHON_BIN || "python";
  const env = {
    ...process.env,
    STREAMLIT_SERVER_HEADLESS: "true",
    STREAMLIT_BROWSER_GATHER_USAGE_STATS: "false",
  };

  streamlitProcess = spawn(
    pythonCommand,
    [
      "-m",
      "streamlit",
      "run",
      "app.py",
      "--server.address",
      host,
      "--server.port",
      String(streamlitPort),
      "--server.headless",
      "true",
      "--browser.gatherUsageStats",
      "false",
    ],
    { env, stdio: "inherit" }
  );

  streamlitProcess.on("exit", (code, signal) => {
    console.error(`Streamlit exited with code=${code} signal=${signal}`);
    streamlitProcess = null;
  });
}

function proxyHttp(req, res) {
  const options = {
    hostname: host,
    port: streamlitPort,
    path: req.url,
    method: req.method,
    headers: {
      ...req.headers,
      host: `${host}:${streamlitPort}`,
    },
  };

  const proxyReq = http.request(options, (proxyRes) => {
    res.writeHead(proxyRes.statusCode || 502, proxyRes.headers);
    proxyRes.pipe(res, { end: true });
  });

  proxyReq.on("error", () => {
    res.writeHead(503, { "Content-Type": "text/plain; charset=utf-8" });
    res.end("BuyerLab AI is starting. Refresh in a few seconds.");
  });

  req.pipe(proxyReq, { end: true });
}

function proxyUpgrade(req, socket, head) {
  const upstream = net.connect(streamlitPort, host, () => {
    upstream.write(
      `${req.method} ${req.url} HTTP/${req.httpVersion}\r\n` +
        Object.entries({
          ...req.headers,
          host: `${host}:${streamlitPort}`,
        })
          .map(([key, value]) => `${key}: ${value}`)
          .join("\r\n") +
        "\r\n\r\n"
    );
    if (head && head.length) {
      upstream.write(head);
    }
    upstream.pipe(socket);
    socket.pipe(upstream);
  });

  upstream.on("error", () => {
    socket.destroy();
  });
}

startStreamlit();

const server = http.createServer(proxyHttp);
server.on("upgrade", proxyUpgrade);
server.listen(passengerPort, "0.0.0.0", () => {
  console.log(`BuyerLab AI Node wrapper listening on ${passengerPort}`);
  console.log(`Proxying Streamlit on ${host}:${streamlitPort}`);
});

process.on("SIGTERM", () => {
  if (streamlitProcess) {
    streamlitProcess.kill("SIGTERM");
  }
  process.exit(0);
});
