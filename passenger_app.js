const http = require("http");

const port = Number(process.env.PORT || 3000);
const GEMINI_MODEL = process.env.GEMINI_MODEL || "gemini-2.5-flash";

const samples = {
  earbuds: {
    title: "SoundPeak AirBass X2 kablosuz kulaklık",
    category: "Elektronik / Aksesuar",
    price: 799,
    currency: "TRY",
    description:
      "Günlük müzik, online toplantı ve öğrenci kullanımı için kompakt kablosuz kulaklık. Daha iyi mikrofon ve daha uzun pil ömrü iddia ediliyor ancak net teknik kanıt yok.",
    proof:
      "Garanti süresi belirsiz. Pil ömrü, mikrofon testi ve gerçek kullanıcı kanıtı eksik.",
  },
  course: {
    title: "DataPath Academy SQL Foundations",
    category: "Online Kurs",
    price: 899,
    currency: "TRY",
    description:
      "SQL öğrenmek isteyen yeni başlayanlar için online kurs. Müfredat genel anlatılmış, eğitmen kanıtı ve örnek ders zayıf.",
    proof:
      "Eğitmen geçmişi, örnek ders, net öğrenme çıktıları ve iade politikası eksik.",
  },
};

function wantsMockMode() {
  return String(process.env.BUYERLAB_MOCK_MODE || "").toLowerCase() === "true";
}

function send(res, status, body, contentType = "text/html; charset=utf-8") {
  res.writeHead(status, {
    "Content-Type": contentType,
    "Cache-Control": "no-store",
  });
  res.end(body);
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    let body = "";
    req.on("data", (chunk) => {
      body += chunk.toString();
      if (body.length > 1_000_000) {
        reject(new Error("Request body too large"));
        req.destroy();
      }
    });
    req.on("end", () => resolve(body));
    req.on("error", reject);
  });
}

function normalizeProduct(input = {}) {
  return {
    title: String(input.title || "").trim(),
    category: String(input.category || "Genel Ürün").trim(),
    price: Number(input.price || 0),
    currency: String(input.currency || "TRY").trim() || "TRY",
    description: String(input.description || "").trim(),
    proof: String(input.proof || "").trim(),
  };
}

function buildPrompt(product) {
  return `BuyerLab AI bir yayın öncesi ürün sayfası denetim aracıdır.

Ürün:
- Başlık: ${product.title}
- Kategori: ${product.category}
- Fiyat: ${product.price} ${product.currency}
- Açıklama: ${product.description}
- Kanıt / güven bilgileri: ${product.proof}

Görev:
Bu ürün sayfasını Türkçe, kısa, profesyonel ve jüri demosuna uygun şekilde denetle.
Gerçek pazar tahmini yapma. "simüle dönüşüm skoru" ifadesini kullan.
Gerçek göz takibi iddia etme.

Sadece geçerli JSON döndür:
{
  "launch_status": "Hazır" | "Düzeltme Gerekli" | "Hazır Değil",
  "simulated_conversion_score": 0-100,
  "main_blocker": "kısa ana engel",
  "executive_verdict": "1-2 cümlelik net karar",
  "required_fixes": ["somut düzeltme", "somut düzeltme", "somut düzeltme"],
  "persona_verdicts": [
    {"persona": "Şüpheci Müşteri", "decision": "Satın alır|Kararsız|Reddeder", "reason": "kısa neden"},
    {"persona": "Fiyat Odaklı Müşteri", "decision": "Satın alır|Kararsız|Reddeder", "reason": "kısa neden"},
    {"persona": "Dürtüsel Müşteri", "decision": "Satın alır|Kararsız|Reddeder", "reason": "kısa neden"},
    {"persona": "Güven Arayan Müşteri", "decision": "Satın alır|Kararsız|Reddeder", "reason": "kısa neden"}
  ],
  "friction_map": [
    {"section": "Başlık", "attention": 0-100, "friction": 0-100, "fix": "kısa düzeltme"},
    {"section": "Fiyat", "attention": 0-100, "friction": 0-100, "fix": "kısa düzeltme"},
    {"section": "Açıklama", "attention": 0-100, "friction": 0-100, "fix": "kısa düzeltme"},
    {"section": "Güven kanıtı", "attention": 0-100, "friction": 0-100, "fix": "kısa düzeltme"}
  ],
  "fix_pack": {
    "title": "daha iyi ürün başlığı",
    "value_proposition": "daha net değer önerisi",
    "description": "kısa iyileştirilmiş açıklama",
    "cta": "kısa CTA"
  }
}`;
}

function extractJson(text) {
  const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/i);
  const candidate = fenced ? fenced[1] : text;
  const start = candidate.indexOf("{");
  const end = candidate.lastIndexOf("}");
  if (start === -1 || end === -1 || end <= start) {
    throw new Error("Gemini JSON response could not be parsed");
  }
  return JSON.parse(candidate.slice(start, end + 1));
}

async function callGemini(product) {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error("GEMINI_API_KEY is missing");
  }

  const response = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(
      GEMINI_MODEL.replace(/^models\//, "")
    )}:generateContent?key=${encodeURIComponent(apiKey)}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        contents: [{ parts: [{ text: buildPrompt(product) }] }],
        generationConfig: {
          temperature: 0.2,
          responseMimeType: "application/json",
        },
      }),
    }
  );

  const body = await response.text();
  if (!response.ok) {
    throw new Error(`Gemini HTTP ${response.status}`);
  }

  const parsed = JSON.parse(body);
  const text =
    parsed?.candidates?.[0]?.content?.parts
      ?.map((part) => part.text || "")
      .join("\n")
      .trim() || "";
  return extractJson(text);
}

function fallbackAudit(product) {
  const weakProof = !product.proof || product.proof.length < 60;
  const weakDescription = !product.description || product.description.length < 80;
  const highPrice = product.currency.toUpperCase().includes("TRY") && product.price >= 750;
  const score = Math.max(35, 78 - (weakProof ? 20 : 0) - (weakDescription ? 12 : 0) - (highPrice ? 10 : 0));
  const status = score >= 75 ? "Hazır" : score >= 55 ? "Düzeltme Gerekli" : "Hazır Değil";
  const blocker = weakProof
    ? "Güven kanıtları ve fiyat gerekçesi yeterince net değil."
    : "Ürün sayfası daha net fayda ve satın alma nedeni göstermeli.";

  return {
    launch_status: status,
    simulated_conversion_score: score,
    main_blocker: blocker,
    executive_verdict:
      status === "Hazır"
        ? "Sayfa yayına alınabilir; yine de güven ve kanıt alanları güçlendirilirse satış itirazları azalır."
        : "Bu ürün sayfası şu haliyle yayına hazır değil. Yayından önce güven kanıtı, net ürün detayları ve fiyat gerekçesi güçlendirilmelidir.",
    required_fixes: [
      "Garanti süresi, iade koşulları ve destek bilgisini net yaz.",
      "Fiyatı savunmak için ölçülebilir ürün kanıtı ekle.",
      "Genel kalite iddiaları yerine somut teknik/fayda bilgileri kullan.",
    ],
    persona_verdicts: [
      { persona: "Şüpheci Müşteri", decision: weakProof ? "Reddeder" : "Kararsız", reason: "Kanıt ve teknik detay görmek ister." },
      { persona: "Fiyat Odaklı Müşteri", decision: highPrice ? "Kararsız" : "Satın alır", reason: "TL fiyatın neden değerli olduğunu görmek ister." },
      { persona: "Dürtüsel Müşteri", decision: weakDescription ? "Kararsız" : "Satın alır", reason: "Teklifin hızlı ve heyecan verici anlaşılması gerekir." },
      { persona: "Güven Arayan Müşteri", decision: weakProof ? "Reddeder" : "Kararsız", reason: "Garanti, iade ve sosyal kanıt bekler." },
    ],
    friction_map: [
      { section: "Başlık", attention: 72, friction: weakDescription ? 42 : 24, fix: "Başlığa ürün tipi ve ana faydayı ekle." },
      { section: "Fiyat", attention: 80, friction: highPrice ? 72 : 38, fix: "Fiyatı kanıtlayan fayda ve karşılaştırma ekle." },
      { section: "Açıklama", attention: 66, friction: weakDescription ? 70 : 35, fix: "Açıklamayı ölçülebilir detaylarla netleştir." },
      { section: "Güven kanıtı", attention: 75, friction: weakProof ? 85 : 40, fix: "Gerçek garanti, iade ve kanıt varlıkları ekle." },
    ],
    fix_pack: {
      title: product.title || "Net fayda odaklı ürün başlığı",
      value_proposition: "Bu ürünün kime, hangi durumda, neden değer sunduğunu tek cümlede anlat.",
      description:
        "Ürün açıklamasını somut özellikler, kullanım senaryosu, garanti/iade bilgisi ve gerçek kanıtlarla destekle.",
      cta: "Ürünü güvenle incele",
    },
  };
}

function sanitizeAudit(audit, product) {
  const fallback = fallbackAudit(product);
  return {
    launch_status: audit.launch_status || fallback.launch_status,
    simulated_conversion_score: Number(audit.simulated_conversion_score ?? fallback.simulated_conversion_score),
    main_blocker: audit.main_blocker || fallback.main_blocker,
    executive_verdict: audit.executive_verdict || fallback.executive_verdict,
    required_fixes: Array.isArray(audit.required_fixes) && audit.required_fixes.length ? audit.required_fixes.slice(0, 5) : fallback.required_fixes,
    persona_verdicts: Array.isArray(audit.persona_verdicts) && audit.persona_verdicts.length ? audit.persona_verdicts.slice(0, 4) : fallback.persona_verdicts,
    friction_map: Array.isArray(audit.friction_map) && audit.friction_map.length ? audit.friction_map.slice(0, 6) : fallback.friction_map,
    fix_pack: audit.fix_pack || fallback.fix_pack,
    ai_mode: wantsMockMode() || !process.env.GEMINI_API_KEY ? "demo/fallback" : "live-gemini",
  };
}

async function auditProduct(product) {
  if (wantsMockMode() || !process.env.GEMINI_API_KEY) {
    return sanitizeAudit(fallbackAudit(product), product);
  }

  try {
    const audit = await callGemini(product);
    return sanitizeAudit(audit, product);
  } catch (error) {
    console.error(`Gemini audit failed safely: ${error.message}`);
    return sanitizeAudit(fallbackAudit(product), product);
  }
}

function html() {
  return `<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>BuyerLab AI Demo</title>
  <style>
    :root { color-scheme: dark; --bg:#07101d; --panel:#111b2b; --card:#141f31; --line:#253550; --text:#eef5ff; --muted:#9fb2cb; --cyan:#38d5ff; --red:#ff5d6c; --green:#55e091; --amber:#ffc857; }
    * { box-sizing: border-box; }
    body { margin: 0; background: radial-gradient(circle at top left, #122543 0, #07101d 35%, #060b14 100%); color: var(--text); font-family: Inter, Arial, sans-serif; }
    main { width: min(1180px, calc(100% - 32px)); margin: 0 auto; padding: 28px 0 52px; }
    .hero { display: grid; grid-template-columns: 1.15fr .85fr; gap: 22px; align-items: stretch; margin-bottom: 22px; }
    .box, .card { background: rgba(17,27,43,.92); border: 1px solid var(--line); border-radius: 18px; box-shadow: 0 18px 80px rgba(0,0,0,.24); }
    .box { padding: 26px; }
    .eyebrow { color: var(--cyan); font-weight: 800; letter-spacing: .02em; font-size: 13px; margin: 0 0 8px; }
    h1 { margin: 0; font-size: clamp(32px, 5vw, 58px); line-height: .95; }
    h2 { margin: 0 0 14px; font-size: 22px; }
    h3 { margin: 0 0 10px; font-size: 17px; }
    p { color: var(--muted); line-height: 1.55; }
    label { display:block; font-size: 13px; color: #c9d7ea; margin: 12px 0 6px; font-weight: 700; }
    input, textarea, select { width: 100%; background: #0a1322; color: var(--text); border: 1px solid #2d405f; border-radius: 12px; padding: 12px 13px; font: inherit; outline: none; }
    textarea { min-height: 92px; resize: vertical; }
    button { border: 0; border-radius: 13px; padding: 13px 16px; color: white; background: linear-gradient(135deg, #ff3c5f, #ff7a45); font-weight: 900; cursor:pointer; }
    button.secondary { background: #172942; color: #cfe7ff; border: 1px solid #2b466a; }
    .grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
    .two { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
    .card { padding: 18px; }
    .metric { font-size: 34px; font-weight: 900; }
    .status { display:inline-flex; padding: 8px 11px; border-radius: 999px; font-weight: 900; background: rgba(255,200,87,.14); color: var(--amber); }
    .status.ready { background: rgba(85,224,145,.14); color: var(--green); }
    .status.bad { background: rgba(255,93,108,.15); color: var(--red); }
    .muted { color: var(--muted); }
    .actions { display:flex; flex-wrap: wrap; gap: 10px; margin-top: 16px; }
    .results { display:none; margin-top: 18px; }
    .list { margin: 0; padding-left: 18px; color: #dce8f7; }
    .list li { margin: 8px 0; }
    table { width: 100%; border-collapse: collapse; overflow: hidden; border-radius: 14px; }
    th, td { text-align:left; border-bottom: 1px solid #263753; padding: 12px; vertical-align: top; }
    th { color: var(--muted); font-size: 13px; }
    .small { font-size: 13px; color: var(--muted); }
    .loading { display:none; color: var(--cyan); font-weight: 800; margin-top: 12px; }
    footer { margin-top: 28px; color: #7d90aa; font-size: 13px; }
    @media (max-width: 860px) {
      main { width: min(100% - 20px, 720px); padding-top: 16px; }
      .hero, .two, .grid { grid-template-columns: 1fr; }
      .box { padding: 20px; }
      .metric { font-size: 28px; }
      table { display:block; overflow-x:auto; }
    }
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <div class="box">
        <p class="eyebrow">Yayın Öncesi Ürün Sayfası Denetimi</p>
        <h1>BuyerLab AI Demo</h1>
        <p>Ürün sayfanı AI müşteri profilleriyle hızlıca test eder; güven eksiklerini, fiyat sürtünmesini ve yayına çıkmadan önce düzeltilmesi gereken noktaları net raporlar.</p>
        <div class="actions">
          <button onclick="loadSample('earbuds', true)">Kablosuz Kulaklık (Demo) çalıştır</button>
          <button class="secondary" onclick="loadSample('course', true)">Online Kurs (Demo) çalıştır</button>
        </div>
        <div id="loading" class="loading">Analiz hazırlanıyor...</div>
      </div>
      <form id="auditForm" class="box">
        <h2>Hızlı Ürün Girişi</h2>
        <label>Ürün başlığı</label>
        <input id="title" placeholder="Örn. kablosuz kulaklık, online kurs, kahve makinesi" />
        <div class="two">
          <div>
            <label>Kategori</label>
            <select id="category">
              <option>Elektronik / Aksesuar</option>
              <option>Moda / Ayakkabı</option>
              <option>Küçük Ev Aleti</option>
              <option>Online Kurs</option>
              <option>Dijital Hizmet</option>
              <option>Genel Ürün</option>
            </select>
          </div>
          <div>
            <label>Fiyat</label>
            <input id="price" type="number" min="0" value="799" />
          </div>
        </div>
        <label>Ürün açıklaması</label>
        <textarea id="description" placeholder="Ürün ne işe yarıyor, kim kullanıyor, ana iddia ne?"></textarea>
        <label>Kanıt / güven bilgileri</label>
        <textarea id="proof" placeholder="Garanti, iade, teknik kanıt, müşteri yorumu, örnek ders, portfolyo vb."></textarea>
        <div class="actions">
          <button type="submit">Ürün Sayfasını Test Et</button>
        </div>
      </form>
    </section>

    <section id="results" class="results">
      <div class="grid">
        <div class="card"><h3>Yayın Kararı</h3><span id="status" class="status">-</span></div>
        <div class="card"><h3>Simüle Dönüşüm Skoru</h3><div id="score" class="metric">-</div></div>
        <div class="card"><h3>Ana Engel</h3><p id="blocker">-</p></div>
      </div>
      <div class="two" style="margin-top:14px">
        <div class="card"><h2>1 Dakikalık Rapor</h2><p id="verdict">-</p></div>
        <div class="card"><h2>İlk Düzeltmeler</h2><ul id="fixes" class="list"></ul></div>
      </div>
      <div class="card" style="margin-top:14px">
        <h2>Müşteri İtirazları</h2>
        <table><thead><tr><th>Persona</th><th>Karar</th><th>Neden</th></tr></thead><tbody id="personas"></tbody></table>
      </div>
      <div class="card" style="margin-top:14px">
        <h2>Dönüşüm Sürtünme Haritası</h2>
        <p class="small">Bu analiz gerçek göz takibi değildir; AI destekli müşteri dikkat ve sürtünme simülasyonudur.</p>
        <table><thead><tr><th>Bölüm</th><th>Dikkat</th><th>Sürtünme</th><th>Öneri</th></tr></thead><tbody id="friction"></tbody></table>
      </div>
      <div class="card" style="margin-top:14px">
        <h2>Düzeltme Planı</h2>
        <div class="two">
          <p><b>Başlık:</b><br><span id="fpTitle"></span></p>
          <p><b>Değer önerisi:</b><br><span id="fpValue"></span></p>
          <p><b>Açıklama:</b><br><span id="fpDescription"></span></p>
          <p><b>CTA:</b><br><span id="fpCta"></span></p>
        </div>
      </div>
    </section>
    <footer>BuyerLab AI, yayın öncesi ürün sayfası zayıflıklarını bulmak için AI destekli tanı üretir. Gerçek satış tahmini değildir.</footer>
  </main>
  <script>
    const samples = ${JSON.stringify(samples)};
    const $ = (id) => document.getElementById(id);

    function loadSample(key, run) {
      const sample = samples[key];
      $("title").value = sample.title;
      $("category").value = sample.category;
      $("price").value = sample.price;
      $("description").value = sample.description;
      $("proof").value = sample.proof;
      if (run) runAudit();
    }

    function productFromForm() {
      return {
        title: $("title").value,
        category: $("category").value,
        price: Number($("price").value || 0),
        currency: "TRY",
        description: $("description").value,
        proof: $("proof").value,
      };
    }

    function statusClass(status) {
      if (status === "Hazır") return "status ready";
      if (status === "Hazır Değil") return "status bad";
      return "status";
    }

    function renderList(id, items) {
      $(id).innerHTML = (items || []).map((item) => '<li>' + escapeHtml(item) + '</li>').join('');
    }

    function escapeHtml(value) {
      return String(value ?? "").replace(/[&<>"']/g, (char) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[char]));
    }

    function renderAudit(audit) {
      $("results").style.display = "block";
      $("status").className = statusClass(audit.launch_status);
      $("status").textContent = audit.launch_status;
      $("score").textContent = String(audit.simulated_conversion_score) + "/100";
      $("blocker").textContent = audit.main_blocker;
      $("verdict").textContent = audit.executive_verdict;
      renderList("fixes", audit.required_fixes);
      $("personas").innerHTML = (audit.persona_verdicts || []).map((row) =>
        '<tr><td>' + escapeHtml(row.persona) + '</td><td>' + escapeHtml(row.decision) + '</td><td>' + escapeHtml(row.reason) + '</td></tr>'
      ).join('');
      $("friction").innerHTML = (audit.friction_map || []).map((row) =>
        '<tr><td>' + escapeHtml(row.section) + '</td><td>' + escapeHtml(row.attention) + '</td><td>' + escapeHtml(row.friction) + '</td><td>' + escapeHtml(row.fix) + '</td></tr>'
      ).join('');
      const fp = audit.fix_pack || {};
      $("fpTitle").textContent = fp.title || "-";
      $("fpValue").textContent = fp.value_proposition || "-";
      $("fpDescription").textContent = fp.description || "-";
      $("fpCta").textContent = fp.cta || "-";
      $("results").scrollIntoView({behavior: "smooth", block: "start"});
    }

    async function runAudit() {
      $("loading").style.display = "block";
      try {
        const response = await fetch("/api/audit", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(productFromForm()),
        });
        const audit = await response.json();
        renderAudit(audit);
      } catch (error) {
        renderAudit({
          launch_status: "Düzeltme Gerekli",
          simulated_conversion_score: 55,
          main_blocker: "Analiz sırasında bağlantı sorunu oluştu.",
          executive_verdict: "Demo güvenli moda geçti. Ürün sayfası yine de temel güven ve bilgi eksikleri açısından değerlendirilebilir.",
          required_fixes: ["Garanti ve iade bilgisini netleştir.", "Fiyatı kanıtlayan ürün detayları ekle.", "Somut güven sinyalleri kullan."],
          persona_verdicts: [],
          friction_map: [],
          fix_pack: {},
        });
      } finally {
        $("loading").style.display = "none";
      }
    }

    $("auditForm").addEventListener("submit", (event) => {
      event.preventDefault();
      runAudit();
    });
    loadSample("earbuds", false);
  </script>
</body>
</html>`;
}

async function handleRequest(req, res) {
  if (req.url === "/favicon.ico") {
    send(res, 204, "");
    return;
  }

  if (req.method === "POST" && req.url === "/api/audit") {
    try {
      const body = await readBody(req);
      const product = normalizeProduct(JSON.parse(body || "{}"));
      const audit = await auditProduct(product);
      send(res, 200, JSON.stringify(audit), "application/json; charset=utf-8");
    } catch (error) {
      console.error(`Audit request failed safely: ${error.message}`);
      const fallback = sanitizeAudit(fallbackAudit(normalizeProduct({})), normalizeProduct({}));
      send(res, 200, JSON.stringify(fallback), "application/json; charset=utf-8");
    }
    return;
  }

  send(res, 200, html());
}

http.createServer(handleRequest).listen(port, "0.0.0.0", () => {
  console.log(`BuyerLab AI cPanel app listening on ${port}`);
  console.log(`AI mode: ${wantsMockMode() || !process.env.GEMINI_API_KEY ? "demo/fallback" : "live-gemini"}`);
});
