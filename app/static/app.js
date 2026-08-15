// HH Goa 2026 Voice RAG Frontend Client

document.addEventListener('DOMContentLoaded', () => {
  const micBtn = document.getElementById('mic-btn');
  const micText = document.getElementById('mic-text');
  const recordingIndicator = document.getElementById('recording-indicator');
  const queryInput = document.getElementById('query-input');
  const submitBtn = document.getElementById('submit-btn');
  const denseSlider = document.getElementById('dense-slider');
  const denseVal = document.getElementById('dense-val');

  const answerText = document.getElementById('answer-text');
  const langBadge = document.getElementById('lang-badge');
  const groundingBadge = document.getElementById('grounding-badge');
  const totalLatencyBadge = document.getElementById('total-latency-badge');
  const sourcesList = document.getElementById('sources-list');

  const healthStatus = document.getElementById('health-status');
  const healthText = document.getElementById('health-text');
  const docsPill = document.getElementById('docs-pill');
  const modelPill = document.getElementById('model-pill');

  let mediaRecorder = null;
  let audioChunks = [];
  let isRecording = false;

  // 1. Initial Health Check
  fetchHealth();

  async function fetchHealth() {
    try {
      const res = await fetch('/health');
      if (res.ok) {
        const data = await res.json();
        healthText.textContent = `Online (${data.status})`;
        healthStatus.querySelector('.status-dot').classList.add('active');
        docsPill.textContent = `${data.total_indexed_documents} Indexed Chunks`;
        modelPill.textContent = data.model_id;
      }
    } catch (e) {
      healthText.textContent = 'API Offline';
    }
  }

  // 2. Slider listener
  denseSlider.addEventListener('input', (e) => {
    denseVal.textContent = parseFloat(e.target.value).toFixed(1);
  });

  // 3. Sample query buttons
  document.querySelectorAll('.sample-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      queryInput.value = btn.getAttribute('data-query');
      executeQuery();
    });
  });

  // 4. Submit button
  submitBtn.addEventListener('click', executeQuery);
  queryInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      executeQuery();
    }
  });

  // 5. Text Query Execution
  async function executeQuery() {
    const query = queryInput.value.trim();
    if (!query) return;

    setLoading(true);

    try {
      const denseWeight = parseFloat(denseSlider.value);
      const res = await fetch('/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: query,
          dense_weight: denseWeight,
          bm25_weight: 1.0 - denseWeight,
          include_debug: true,
        }),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Server error');
      }

      const data = await res.json();
      renderResponse(data);
    } catch (err) {
      answerText.textContent = `Error: ${err.message}`;
      answerText.className = 'error-text';
    } finally {
      setLoading(false);
    }
  }

  // 6. Voice Recording
  micBtn.addEventListener('click', async () => {
    if (!isRecording) {
      startRecording();
    } else {
      stopRecording();
    }
  });

  async function startRecording() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioChunks = [];
      mediaRecorder = new MediaRecorder(stream);

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunks.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
        await sendAudioBlob(audioBlob);
        stream.getTracks().forEach(t => t.stop());
      };

      mediaRecorder.start();
      isRecording = true;
      micBtn.classList.add('recording');
      micText.textContent = 'Listening... Click to Stop';
      recordingIndicator.classList.remove('hidden');
    } catch (err) {
      alert('Microphone access error: ' + err.message);
    }
  }

  function stopRecording() {
    if (mediaRecorder && isRecording) {
      mediaRecorder.stop();
      isRecording = false;
      micBtn.classList.remove('recording');
      micText.textContent = 'Hold or Click to Speak';
      recordingIndicator.classList.add('hidden');
    }
  }

  async function sendAudioBlob(blob) {
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('file', blob, 'recording.wav');

      const res = await fetch('/voice/upload', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Voice processing error');
      }

      const data = await res.json();
      queryInput.value = data.query || '';
      renderResponse(data);
    } catch (err) {
      answerText.textContent = `Voice Error: ${err.message}`;
    } finally {
      setLoading(false);
    }
  }

  // 7. Render Response Details
  function renderResponse(data) {
    answerText.textContent = data.answer;
    answerText.className = '';

    // Language badge
    langBadge.textContent = `Lang: ${data.detected_language || 'Unknown'}`;

    // Grounding badge
    if (data.is_refusal) {
      groundingBadge.textContent = 'Grounding: Refusal / Low Context';
      groundingBadge.className = 'badge-grounding refusal';
    } else if (data.grounding && data.grounding.is_grounded) {
      groundingBadge.textContent = `Grounding: Grounded (${(data.grounding.grounding_score * 100).toFixed(0)}%)`;
      groundingBadge.className = 'badge-grounding grounded';
    } else {
      groundingBadge.textContent = 'Grounding: Unverified';
      groundingBadge.className = 'badge-grounding';
    }

    // Latency metrics
    const l = data.latency;
    totalLatencyBadge.textContent = `${l.total_ms.toFixed(1)} ms`;
    totalLatencyBadge.className = 'latency-pill ' + (l.total_ms <= 200 ? 'budget-pass' : 'budget-fail');

    document.getElementById('lat-stt').textContent = `${l.stt_ms.toFixed(1)} ms`;
    document.getElementById('lat-input').textContent = `${l.input_guardrails_ms.toFixed(1)} ms`;
    document.getElementById('lat-embed').textContent = `${l.query_embed_ms.toFixed(1)} ms`;
    document.getElementById('lat-vector').textContent = `${l.vector_retrieval_ms.toFixed(1)} ms`;
    document.getElementById('lat-bm25').textContent = `${l.bm25_retrieval_ms.toFixed(1)} ms`;
    document.getElementById('lat-fusion').textContent = `${l.hybrid_fusion_ms.toFixed(1)} ms`;
    document.getElementById('lat-llm').textContent = `${l.llm_generation_ms.toFixed(1)} ms`;
    document.getElementById('lat-grounding').textContent = `${l.grounding_check_ms.toFixed(1)} ms`;

    // Sources list
    sourcesList.innerHTML = '';
    if (!data.sources || data.sources.length === 0) {
      sourcesList.innerHTML = '<div class="empty-sources">No sources retrieved.</div>';
    } else {
      data.sources.forEach((src, idx) => {
        const item = document.createElement('div');
        item.className = 'source-item';
        item.innerHTML = `
          <div class="source-meta">
            <span>#${idx + 1} | Lang: <strong>${src.language}</strong></span>
            <span>Fused Score: <span class="score-tag">${src.score.toFixed(3)}</span> (Dense: ${src.dense_score || 0} | BM25: ${src.bm25_score || 0})</span>
          </div>
          <div class="source-text">${src.text}</div>
        `;
        sourcesList.appendChild(item);
      });
    }
  }

  function setLoading(loading) {
    if (loading) {
      submitBtn.disabled = true;
      submitBtn.querySelector('span').textContent = 'Processing...';
      answerText.textContent = 'Generating grounded answer...';
    } else {
      submitBtn.disabled = false;
      submitBtn.querySelector('span').textContent = 'Execute RAG Pipeline';
    }
  }
});
