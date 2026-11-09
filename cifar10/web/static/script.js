const input = document.getElementById("image-input");
const preview = document.getElementById("preview");
const button = document.getElementById("classify-btn");
const statusEl = document.getElementById("status");
const lists = {
  baseline: document.getElementById("results-baseline"),
  optimized: document.getElementById("results-optimized"),
};

function clearResults() {
  for (const ul of Object.values(lists)) ul.innerHTML = "";
}

input.addEventListener("change", () => {
  const file = input.files[0];
  clearResults();
  statusEl.textContent = "";
  if (!file) {
    preview.hidden = true;
    button.disabled = true;
    return;
  }
  preview.src = URL.createObjectURL(file);
  preview.hidden = false;
  button.disabled = false;
});

button.addEventListener("click", async () => {
  const form = new FormData();
  form.append("image", input.files[0]);
  button.disabled = true;
  statusEl.textContent = "Classifying...";
  clearResults();
  try {
    const resp = await fetch("/api/classify", { method: "POST", body: form });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error);
    for (const [name, ul] of Object.entries(lists)) {
      const res = data[name];
      if (res.error) {
        const li = document.createElement("li");
        li.textContent = `Error: ${res.error}`;
        ul.appendChild(li);
        continue;
      }
      const sorted = Object.entries(res.predictions).sort((a, b) => b[1] - a[1]);
      for (const [label, prob] of sorted) {
        const li = document.createElement("li");
        li.textContent = `${label}: ${(prob * 100).toFixed(1)}%`;
        ul.appendChild(li);
      }
    }
    statusEl.textContent = "";
  } catch (err) {
    statusEl.textContent = `Error: ${err.message}`;
  } finally {
    button.disabled = false;
  }
});