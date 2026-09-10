document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const activeModelTag = document.getElementById("activeModelTag");
  const queryInput = document.getElementById("queryInput");
  const submitQueryBtn = document.getElementById("submitQueryBtn");
  const dualModeToggle = document.getElementById("dualModeToggle");
  const ollamaToggle = document.getElementById("ollamaToggle");
  const suggestionChips = document.querySelectorAll(".chip");

  const placeholderState = document.getElementById("placeholderState");
  const singleResponseCard = document.getElementById("singleResponseCard");
  const singleAnswerBody = document.getElementById("singleAnswerBody");
  const singleModelBadge = document.getElementById("singleModelBadge");
  const singleLatencyBadge = document.getElementById("singleLatencyBadge");
  const singleCitationsBadge = document.getElementById("singleCitationsBadge");
  const regenerateBtn = document.getElementById("regenerateBtn");

  const dualResponseCard = document.getElementById("dualResponseCard");
  const candidateTextA = document.getElementById("candidateTextA");
  const candidateTextB = document.getElementById("candidateTextB");
  const voteABtn = document.getElementById("voteABtn");
  const voteBBtn = document.getElementById("voteBBtn");

  const evidenceList = document.getElementById("evidenceList");
  const evidenceCountBadge = document.getElementById("evidenceCountBadge");

  // Feedback Elements
  const thumbUpBtn = document.getElementById("thumbUpBtn");
  const thumbDownBtn = document.getElementById("thumbDownBtn");
  const starRatingGroup = document.getElementById("starRatingGroup");
  const acceptCitationsBtn = document.getElementById("acceptCitationsBtn");
  const rejectCitationsBtn = document.getElementById("rejectCitationsBtn");
  const taskSuccessCheckbox = document.getElementById("taskSuccessCheckbox");
  const correctionInput = document.getElementById("correctionInput");
  const submitCorrectionBtn = document.getElementById("submitCorrectionBtn");

  // Analytics Modal Elements
  const tabAnalyticsBtn = document.getElementById("tabAnalyticsBtn");
  const triggerRlhfBtn = document.getElementById("triggerRlhfBtn");
  const analyticsModal = document.getElementById("analyticsModal");
  const closeModalBtn = document.getElementById("closeModalBtn");
  const progressionTableBody = document.getElementById("progressionTableBody");
  const failureModeGrid = document.getElementById("failureModeGrid");
  const toast = document.getElementById("toast");

  // State
  let currentQueryId = null;
  let currentResponseId = null;
  let currentResponseAId = null;
  let currentResponseBId = null;
  let currentFeedbackState = {
    thumbs: 0,
    rating: null,
    citation_accepted: null,
    task_success: null
  };

  // 1. Initial Data Fetch
  loadModelInfo();

  // Suggestion Chips
  suggestionChips.forEach(chip => {
    chip.addEventListener("click", () => {
      queryInput.value = chip.getAttribute("data-query");
      submitQuery();
    });
  });

  // Submit Query
  submitQueryBtn.addEventListener("click", submitQuery);

  // Ollama Toggle Feedback
  ollamaToggle.addEventListener("change", async () => {
    if (ollamaToggle.checked) {
      showToast("Checking local Ollama connection on http://localhost:11434...");
      try {
        const res = await fetch("/api/ollama/status");
        const data = await res.json();
        if (data.available) {
          const modelName = data.configured_model || (data.models && data.models[0]) || "qwen";
          showToast(`Ollama active! Using model '${modelName}' for local generation.`);
          activeModelTag.textContent = `Ollama: ${modelName}`;
        } else {
          showToast("Ollama is not running on port 11434. Run 'ollama serve' in your terminal to serve via Ollama. (Auto-fallback to Neural Policy active).");
          activeModelTag.textContent = "Ollama (Offline)";
        }
      } catch (err) {
        showToast("Could not contact Ollama status endpoint. Fallback active.");
      }
    } else {
      loadModelInfo();
      showToast("Switched back to Neural Research Policy.");
    }
  });
  queryInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      submitQuery();
    }
  });

  async function submitQuery() {
    const query = queryInput.value.trim();
    if (!query) return;

    submitQueryBtn.disabled = true;
    submitQueryBtn.innerHTML = `<span>Synthesizing...</span>`;

    placeholderState.classList.add("hidden");
    singleResponseCard.classList.add("hidden");
    dualResponseCard.classList.add("hidden");

    try {
      const response = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: query,
          dual_response: dualModeToggle.checked,
          use_ollama: ollamaToggle.checked
        })
      });

      if (!response.ok) throw new Error("Failed to synthesize query");
      const data = await response.json();

      currentQueryId = data.query_id;
      renderEvidence(data.evidence);

      if (data.dual_mode) {
        renderDualResponses(data);
      } else {
        renderSingleResponse(data);
      }
    } catch (err) {
      showToast(`Error: ${err.message}`);
    } finally {
      submitQueryBtn.disabled = false;
      submitQueryBtn.innerHTML = `<span>Synthesize</span>
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <line x1="22" y1="2" x2="11" y2="13"></line>
          <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
        </svg>`;
    }
  }

  function renderSingleResponse(data) {
    currentResponseId = data.response_id;
    singleModelBadge.textContent = `Policy: ${data.model_version}`;
    singleLatencyBadge.textContent = `Latency: ${Math.round(data.latency_ms)} ms`;
    singleCitationsBadge.textContent = `Citations: ${data.citations ? data.citations.length : 0}`;

    singleAnswerBody.innerHTML = formatMarkdownWithCitations(data.response_text);
    attachCitationListeners(singleAnswerBody);

    resetFeedbackUI();
    singleResponseCard.classList.remove("hidden");
  }

  function renderDualResponses(data) {
    currentResponseAId = data.response_a.id;
    currentResponseBId = data.response_b.id;

    candidateTextA.innerHTML = formatMarkdownWithCitations(data.response_a.text);
    candidateTextB.innerHTML = formatMarkdownWithCitations(data.response_b.text);

    attachCitationListeners(candidateTextA);
    attachCitationListeners(candidateTextB);

    voteABtn.onclick = () => submitABPreference("A");
    voteBBtn.onclick = () => submitABPreference("B");

    dualResponseCard.classList.remove("hidden");
  }

  function renderEvidence(evidence) {
    if (!evidenceList) return;
    evidenceList.innerHTML = "";
    if (evidenceCountBadge) evidenceCountBadge.textContent = `${evidence ? evidence.length : 0} papers`;

    if (!evidence || evidence.length === 0) {
      evidenceList.innerHTML = `<div class="empty-evidence">No external papers retrieved for this query.</div>`;
      return;
    }

    evidence.forEach((doc) => {
      const idx = doc.citation_index || 1;
      const card = document.createElement("div");
      card.className = "evidence-item";
      card.id = `evidence-item-${idx}`;

      const authorsStr = doc.authors ? (Array.isArray(doc.authors) ? doc.authors.slice(0, 3).join(", ") : doc.authors) : "N/A";
      const scorePct = Math.round((doc.similarity_score || 0.5) * 100);

      card.innerHTML = `
        <div class="evidence-item-header">
          <span class="citation-idx-badge">[${idx}]</span>
          <h4 class="evidence-title">${escapeHtml(doc.title)}</h4>
        </div>
        <div class="evidence-meta">
          <span>${escapeHtml(authorsStr)}</span> • 
          <span style="color: var(--accent-cyan)">Sim: ${scorePct}%</span> • 
          <a href="${doc.url || '#'}" target="_blank" style="color: var(--text-secondary); text-decoration: underline;">arXiv</a>
        </div>
        <div class="evidence-text">${escapeHtml(doc.text || doc.excerpt || "")}</div>
      `;
      evidenceList.appendChild(card);
    });
  }

  function formatMarkdownWithCitations(text) {
    if (!text) return "";
    let formatted = escapeHtml(text);

    // Format headers
    formatted = formatted.replace(/^### (.*$)/gim, '<h3>$1</h3>');
    formatted = formatted.replace(/^## (.*$)/gim, '<h2>$1</h2>');
    formatted = formatted.replace(/^# (.*$)/gim, '<h1>$1</h1>');

    // Format References section bullet points with direct arXiv links beside title:
    // Matches: - [1] Title — [https://arxiv.org/abs/...](https://arxiv.org/abs/...)
    // or: - [1] Title (arXiv: 1234.5678)
    formatted = formatted.replace(/^\-\s*\[(\d+)\]\s*(.*?)(?:\s*[—–-]\s*\[?(https?:\/\/[^\s\)\]]+)\]?(?:\([^\)]+\))?|\s*\((arXiv:[^\)]+)\))?$/gim, (match, p1, p2, p3, p4) => {
      let url = p3 || "";
      let label = "arXiv Link";
      if (!url && p4) {
        const cleanId = p4.replace(/arxiv:\s*/i, "").trim();
        url = `https://arxiv.org/abs/${cleanId}`;
        label = `arXiv:${cleanId}`;
      } else if (url) {
        const matchId = url.match(/abs\/([^\/\s\?#]+)/);
        if (matchId) label = `arXiv:${matchId[1]}`;
      }
      const titleClean = p2.trim();
      const titleHtml = url
        ? `<a href="${url}" target="_blank" rel="noopener noreferrer" class="ref-paper-name" title="Open paper on arXiv">${titleClean}</a>`
        : `<span class="ref-paper-name">${titleClean}</span>`;
      const linkHtml = url 
        ? `<a href="${url}" target="_blank" rel="noopener noreferrer" class="ref-arxiv-link" title="Direct link to paper on arXiv">
             <span>${label}</span>
             <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
               <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path>
               <polyline points="15 3 21 3 21 9"></polyline>
               <line x1="10" y1="14" x2="21" y2="3"></line>
             </svg>
           </a>`
        : "";
      return `<li class="ref-item" id="ref-${p1}">
  <div class="ref-item-main">
    <span class="ref-num-badge">&#91;${p1}&#93;</span>
    ${titleHtml}
  </div>
  ${linkHtml}
</li>`;
    });

    // Handle generic markdown links [text](url)
    formatted = formatted.replace(/\[([^\]]+)\]\((https?:\/\/[^\)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer" class="ref-arxiv-link"><span>$1</span> <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path><polyline points="15 3 21 3 21 9"></polyline><line x1="10" y1="14" x2="21" y2="3"></line></svg></a>');

    // Handle bold **text**
    formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

    // Handle italic *text*
    formatted = formatted.replace(/\*(.*?)\*/g, '<em>$1</em>');

    // Standard Bullet points
    formatted = formatted.replace(/^\- (.*$)/gim, '<li>$1</li>');

    // Replace citations [1], [2] in body text with clickable chips that scroll to reference
    formatted = formatted.replace(/\[(\d+)\]/g, (match, p1) => {
      return `<a class="citation-ref" data-citation="${p1}" href="#ref-${p1}" title="Jump to reference [${p1}]">[${p1}]</a>`;
    });

    // Paragraphs
    formatted = formatted.replace(/\n\n/g, '<br><br>');
    return formatted;
  }

  function attachCitationListeners(container) {
    container.querySelectorAll(".citation-ref").forEach(ref => {
      const idx = ref.getAttribute("data-citation");
      if (!idx) return;

      ref.addEventListener("click", (e) => {
        const targetRef = document.getElementById(`ref-${idx}`);
        if (targetRef) {
          e.preventDefault();
          targetRef.scrollIntoView({ behavior: "smooth", block: "center" });
          targetRef.classList.add("highlight-pulse");
          setTimeout(() => targetRef.classList.remove("highlight-pulse"), 1800);
        }
      });
    });
  }

  function highlightEvidence(idx, active) {
    const el = document.getElementById(`evidence-item-${idx}`);
    if (el) {
      if (active) {
        el.classList.add("highlighted");
      } else {
        el.classList.remove("highlighted");
      }
    }
  }

  function scrollToEvidence(idx) {
    const el = document.getElementById(`evidence-item-${idx}`);
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }

  // Feedback Handling
  thumbUpBtn.addEventListener("click", () => {
    currentFeedbackState.thumbs = currentFeedbackState.thumbs === 1 ? 0 : 1;
    thumbUpBtn.classList.toggle("active-up", currentFeedbackState.thumbs === 1);
    thumbDownBtn.classList.remove("active-down");
    sendFeedback({ thumbs: currentFeedbackState.thumbs });
  });

  thumbDownBtn.addEventListener("click", () => {
    currentFeedbackState.thumbs = currentFeedbackState.thumbs === -1 ? 0 : -1;
    thumbDownBtn.classList.toggle("active-down", currentFeedbackState.thumbs === -1);
    thumbUpBtn.classList.remove("active-up");
    sendFeedback({ thumbs: currentFeedbackState.thumbs });
  });

  starRatingGroup.querySelectorAll(".star").forEach(star => {
    star.addEventListener("click", () => {
      const val = parseInt(star.getAttribute("data-val"));
      currentFeedbackState.rating = val;
      starRatingGroup.querySelectorAll(".star").forEach((s, idx) => {
        s.classList.toggle("active", idx < val);
      });
      sendFeedback({ rating: val });
    });
  });

  acceptCitationsBtn.addEventListener("click", () => {
    currentFeedbackState.citation_accepted = true;
    acceptCitationsBtn.classList.add("active");
    rejectCitationsBtn.classList.remove("active");
    sendFeedback({ citation_accepted: true });
  });

  rejectCitationsBtn.addEventListener("click", () => {
    currentFeedbackState.citation_accepted = false;
    rejectCitationsBtn.classList.add("active");
    acceptCitationsBtn.classList.remove("active");
    sendFeedback({ citation_accepted: false });
  });

  taskSuccessCheckbox.addEventListener("change", () => {
    const val = taskSuccessCheckbox.checked;
    currentFeedbackState.task_success = val;
    sendFeedback({ task_success: val });
  });

  regenerateBtn.addEventListener("click", () => {
    sendFeedback({ regenerated: true });
    submitQuery();
  });

  submitCorrectionBtn.addEventListener("click", () => {
    const corr = correctionInput.value.trim();
    if (corr) {
      sendFeedback({ user_correction: corr });
      correctionInput.value = "";
      showToast("User correction recorded in preference trajectory buffer.");
    }
  });

  async function sendFeedback(payload) {
    if (!currentResponseId) return;
    try {
      await fetch("/api/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          response_id: currentResponseId,
          ...payload
        })
      });
      showToast("Feedback signal updated.");
    } catch (e) {
      console.error(e);
    }
  }

  async function submitABPreference(preferred) {
    if (!currentQueryId || !currentResponseAId || !currentResponseBId) return;
    try {
      await fetch("/api/feedback/preference", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query_id: currentQueryId,
          response_a_id: currentResponseAId,
          response_b_id: currentResponseBId,
          preferred: preferred
        })
      });
      showToast(`Preference recorded: Response ${preferred} preferred.`);
      dualResponseCard.classList.add("hidden");
      placeholderState.classList.remove("hidden");
    } catch (err) {
      showToast(`Error recording preference: ${err.message}`);
    }
  }

  function resetFeedbackUI() {
    thumbUpBtn.classList.remove("active-up");
    thumbDownBtn.classList.remove("active-down");
    acceptCitationsBtn.classList.remove("active");
    rejectCitationsBtn.classList.remove("active");
    taskSuccessCheckbox.checked = false;
    starRatingGroup.querySelectorAll(".star").forEach(s => s.classList.remove("active"));
  }

  // Model & Analytics Modal
  async function loadModelInfo() {
    try {
      const res = await fetch("/api/models");
      if (res.ok) {
        const data = await res.json();
        activeModelTag.textContent = data.active_version;
      }
    } catch (e) {
      activeModelTag.textContent = "Base Policy";
    }
  }

  tabAnalyticsBtn.addEventListener("click", openAnalyticsModal);
  closeModalBtn.addEventListener("click", () => analyticsModal.classList.add("hidden"));
  analyticsModal.querySelector(".modal-backdrop").addEventListener("click", () => analyticsModal.classList.add("hidden"));

  async function openAnalyticsModal() {
    analyticsModal.classList.remove("hidden");
    try {
      const res = await fetch("/api/metrics");
      const data = await res.json();
      renderAnalytics(data.metrics_history, data.live_stats);
    } catch (e) {
      console.error(e);
    }
  }

  // Live Re-Evaluation Trigger
  const reEvalBtn = document.getElementById("reEvalBtn");
  if (reEvalBtn) {
    reEvalBtn.addEventListener("click", async () => {
      reEvalBtn.disabled = true;
      reEvalBtn.textContent = "Evaluating...";
      showToast("Running live benchmark evaluation against active model...");
      try {
        const res = await fetch("/api/evaluation/run", { method: "POST" });
        const data = await res.json();
        if (data.status === "success") {
          showToast(`Live Evaluation Complete! Gate ${data.gate_passed ? "PASSED (Promoted)" : "FAILED (Regressed)"}`);
          await openAnalyticsModal();
        } else {
          showToast("Evaluation failed: " + (data.detail || "Unknown error"));
        }
      } catch (e) {
        showToast("Error running live evaluation: " + e.message);
      } finally {
        reEvalBtn.disabled = false;
        reEvalBtn.innerHTML = `
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M23 4v6h-6"></path>
            <path d="M1 20v-6h6"></path>
            <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path>
          </svg>
          Run Live Evaluation
        `;
      }
    });
  }

  function renderAnalytics(history, liveStats) {
    if (liveStats) {
      const statQueries = document.getElementById("statQueries");
      const statFeedbacks = document.getElementById("statFeedbacks");
      const statPreferences = document.getElementById("statPreferences");
      const statActiveVersion = document.getElementById("statActiveVersion");

      if (statQueries) statQueries.textContent = liveStats.total_queries ?? 0;
      if (statFeedbacks) statFeedbacks.textContent = liveStats.total_feedbacks ?? 0;
      if (statPreferences) statPreferences.textContent = liveStats.total_preferences ?? 0;
      if (statActiveVersion) statActiveVersion.textContent = liveStats.active_version ?? "Active";
    }

    if (!history || history.length === 0) {
      progressionTableBody.innerHTML = `<tr><td colspan="8" class="text-center">No benchmark rounds executed yet. Click "Trigger RLHF" or "Run Live Evaluation".</td></tr>`;
      return;
    }

    const latest = history[history.length - 1];
    document.getElementById("kpiWinRate").textContent = `${(latest.win_rate * 100).toFixed(1)}%`;
    document.getElementById("kpiWinRateCi").textContent = `95% CI: [${(latest.win_rate_ci[0] * 100).toFixed(1)}%, ${(latest.win_rate_ci[1] * 100).toFixed(1)}%]`;

    document.getElementById("kpiReward").textContent = latest.avg_reward.toFixed(3);
    document.getElementById("kpiRewardCi").textContent = `95% CI: [${latest.avg_reward_ci[0].toFixed(3)}, ${latest.avg_reward_ci[1].toFixed(3)}]`;

    document.getElementById("kpiCitation").textContent = `${(latest.citation_accuracy * 100).toFixed(1)}%`;
    document.getElementById("kpiCitationCi").textContent = `95% CI: [${(latest.citation_accuracy_ci[0] * 100).toFixed(1)}%, ${(latest.citation_accuracy_ci[1] * 100).toFixed(1)}%]`;

    document.getElementById("kpiHallucination").textContent = `${(latest.hallucination_rate * 100).toFixed(1)}%`;

    // Render Table
    progressionTableBody.innerHTML = "";
    history.forEach(r => {
      const row = document.createElement("tr");
      let statusBadge = "";
      if (r.is_base) {
        statusBadge = `<span class="status-badge status-base">Base Reference</span>`;
      } else if (r.gate_passed) {
        statusBadge = `<span class="status-badge status-passed">Gate Passed (Promoted)</span>`;
      } else {
        statusBadge = `<span class="status-badge status-failed">Gate Regressed</span>`;
      }

      row.innerHTML = `
        <td><strong>${escapeHtml(r.round_name)}</strong></td>
        <td>${(r.win_rate * 100).toFixed(1)}% [${(r.win_rate_ci[0]*100).toFixed(1)}%, ${(r.win_rate_ci[1]*100).toFixed(1)}%]</td>
        <td>${r.avg_reward.toFixed(3)} [${r.avg_reward_ci[0].toFixed(3)}, ${r.avg_reward_ci[1].toFixed(3)}]</td>
        <td>${(r.citation_accuracy * 100).toFixed(1)}%</td>
        <td>${(r.groundedness * 100).toFixed(1)}%</td>
        <td>${(r.hallucination_rate * 100).toFixed(1)}%</td>
        <td>${(r.recall_at_k * 100).toFixed(1)}%</td>
        <td>${statusBadge}</td>
      `;
      progressionTableBody.appendChild(row);
    });

    // Render Failure Mode Breakdown
    failureModeGrid.innerHTML = "";
    if (latest.failure_analysis) {
      for (const [mode, val] of Object.entries(latest.failure_analysis)) {
        if (mode === "CLEAN_SUCCESS") continue;
        const card = document.createElement("div");
        card.className = "failure-card";
        const label = mode.replace(/_/g, " ");
        card.innerHTML = `
          <h4>${escapeHtml(label)}</h4>
          <div class="failure-rate">${val.percentage}%</div>
          <p style="font-size: 0.72rem; color: var(--text-muted);">Count: ${val.count} / ${latest.benchmark_size}</p>
        `;
        failureModeGrid.appendChild(card);
      }
    }
  }

  // Trigger RLHF Button
  triggerRlhfBtn.addEventListener("click", async () => {
    triggerRlhfBtn.disabled = true;
    triggerRlhfBtn.textContent = "Executing RLHF...";
    showToast("Triggering post-training RLHF round. Retraining Reward Model and running PPO optimization...");

    try {
      const res = await fetch("/api/rlhf/trigger", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({})
      });
      const data = await res.json();
      showToast(`RLHF Complete! Round: ${data.round_name} | Gate: ${data.gate_passed ? "PASSED" : "FAILED"}`);
      loadModelInfo();
      openAnalyticsModal();
    } catch (e) {
      showToast(`RLHF trigger error: ${e.message}`);
    } finally {
      triggerRlhfBtn.disabled = false;
      triggerRlhfBtn.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polyline points="23 4 23 10 17 10"></polyline>
          <polyline points="1 20 1 14 7 14"></polyline>
          <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path>
        </svg>
        Trigger RLHF
      `;
    }
  });

  function showToast(msg) {
    toast.textContent = msg;
    toast.classList.remove("hidden");
    setTimeout(() => {
      toast.classList.add("hidden");
    }, 4500);
  }

  function escapeHtml(text) {
    if (!text) return "";
    return text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }
});
