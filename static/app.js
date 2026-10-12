// CineMatch frontend — vanilla JS, no frameworks.

const searchInput = document.getElementById("search");
const suggestionsBox = document.getElementById("suggestions");
const noticeBox = document.getElementById("notice");
const resultsSection = document.getElementById("results-section");
const resultsTitle = document.getElementById("results-title");
const resultsSub = document.getElementById("results-sub");
const cardsBox = document.getElementById("cards");
const moodsBox = document.getElementById("moods");

const MOODS = [
  "feel-good", "funny", "romantic", "thriller", "mind-bending",
  "dark", "scary", "adventurous", "emotional", "epic", "sporty",
];

let debounce = null;

// Deterministic gradient poster from the title (no external images needed).
function posterStyle(title) {
  let h = 0;
  for (const ch of title) h = (h * 31 + ch.charCodeAt(0)) % 360;
  const h2 = (h + 60) % 360;
  return `background: linear-gradient(135deg, hsl(${h},55%,32%), hsl(${h2},60%,18%));`;
}

function initials(title) {
  return title.split(/[\s:–-]+/).filter(Boolean).slice(0, 2)
    .map(w => w[0]).join("").toUpperCase();
}

function cardHTML(m) {
  const tags = m.genres.map(g => `<span>${g}</span>`).join("");
  const sim = m.similarity !== undefined
    ? `<div class="sim">▸ ${Math.round(m.similarity * 100)}% match</div>` : "";
  return `
    <div class="card">
      <div class="poster" style="${posterStyle(m.title)}"><span>${initials(m.title)}</span></div>
      <div class="card-body">
        <h3>${m.title}</h3>
        <div class="yr">${m.year}</div>
        <div class="rating">★ ${m.rating.toFixed(1)}</div>
        <div class="genre-tags">${tags}</div>
        <p class="overview">${m.overview}</p>
        ${sim}
      </div>
    </div>`;
}

function showResults(title, sub, movies) {
  resultsTitle.textContent = title;
  resultsSub.textContent = sub;
  cardsBox.innerHTML = movies.map(cardHTML).join("");
  resultsSection.classList.remove("hidden");
  resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

function showNotice(html) {
  noticeBox.innerHTML = html;
  noticeBox.classList.remove("hidden");
}
function hideNotice() { noticeBox.classList.add("hidden"); }

async function getJSON(url) {
  const res = await fetch(url);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw err;
  }
  return res.json();
}

async function recommend(title) {
  hideNotice();
  suggestionsBox.classList.add("hidden");
  searchInput.value = title;
  try {
    const data = await getJSON(`/recommend?title=${encodeURIComponent(title)}&n=8`);
    if (!data.found) {
      // Cold start: unknown title -> "did you mean…"
      if (data.suggestions && data.suggestions.length) {
        const btns = data.suggestions
          .map(s => `<button onclick="recommend('${s.replace(/'/g, "\\'")}')">${s}</button>`)
          .join(" ");
        showNotice(`We couldn't find "<b>${title}</b>". Did you mean: ${btns}`);
      } else {
        showNotice(`We couldn't find "<b>${title}</b>" and have no close matches. Try another title!`);
      }
      resultsSection.classList.add("hidden");
      return;
    }
    showResults(
      `Because you watched ${data.movie.title}`,
      `${data.movie.year} · ${data.movie.genres.join(", ")} · ★ ${data.movie.rating.toFixed(1)}`,
      data.recommendations
    );
  } catch (e) {
    showNotice("Something went wrong talking to the API. Is the server running?");
  }
}

// ---- autocomplete ----
searchInput.addEventListener("input", () => {
  clearTimeout(debounce);
  const q = searchInput.value.trim();
  if (q.length < 2) { suggestionsBox.classList.add("hidden"); return; }
  debounce = setTimeout(async () => {
    try {
      const data = await getJSON(`/search?q=${encodeURIComponent(q)}&n=6`);
      if (!data.results.length) { suggestionsBox.classList.add("hidden"); return; }
      suggestionsBox.innerHTML = data.results
        .map(m => `<div data-title="${m.title.replace(/"/g, "&quot;")}"><span>${m.title}</span><span class="yr">${m.year}</span></div>`)
        .join("");
      suggestionsBox.classList.remove("hidden");
      suggestionsBox.querySelectorAll("div").forEach(el =>
        el.addEventListener("click", () => recommend(el.dataset.title)));
    } catch { /* ignore */ }
  }, 200);
});

searchInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && searchInput.value.trim()) {
    recommend(searchInput.value.trim());
  }
  if (e.key === "Escape") suggestionsBox.classList.add("hidden");
});

document.addEventListener("click", (e) => {
  if (!e.target.closest(".search-wrap")) suggestionsBox.classList.add("hidden");
});

// ---- mood pills ----
MOODS.forEach(mood => {
  const btn = document.createElement("button");
  btn.className = "pill";
  btn.textContent = mood;
  btn.addEventListener("click", async () => {
    hideNotice();
    try {
      const data = await getJSON(`/mood/${mood}?n=8`);
      showResults(
        `In the mood for ${mood}`,
        `Top picks across ${data.genres.join(", ")}`,
        data.recommendations
      );
    } catch (e) {
      const avail = (e.detail && e.detail.available_moods || []).join(", ");
      showNotice(`Unknown mood. Try: ${avail}`);
    }
  });
  moodsBox.appendChild(btn);
});

// ---- surprise ----
document.getElementById("surprise").addEventListener("click", async () => {
  hideNotice();
  try {
    const data = await getJSON("/surprise");
    showResults("🎲 Tonight's surprise pick", "Highly rated, chosen at random", [data.movie]);
  } catch {
    showNotice("Something went wrong talking to the API.");
  }
});

// Expose for inline onclick handlers.
window.recommend = recommend;
