// HH Goa 2026 Voice RAG Frontend Client
// Supports Real-Time Voice Streaming, Audio Queue Playback, and Barge-In Interruption

document.addEventListener('DOMContentLoaded', () => {
  const micBtn = document.getElementById('mic-btn');
  const micText = document.getElementById('mic-text');
  const interruptBtn = document.getElementById('interrupt-btn');
  const voiceStateIndicator = document.getElementById('voice-state-indicator');
  const voiceStateText = document.getElementById('voice-state-text');

  const queryInput = document.getElementById('query-input');
  const submitBtn = document.getElementById('submit-btn');
  const denseSlider = document.getElementById('dense-slider');
  const denseVal = document.getElementById('dense-val');

  const answerText = document.getElementById('answer-text');
  const langBadge = document.getElementById('lang-badge');
  const voiceBadge = document.getElementById('voice-badge');
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
  let currentSessionId = 'sess_' + Math.random().toString(36).substring(2, 9);

  // Audio Playback Queue State
  let audioContext = null;
  let audioQueue = [];
  let isPlayingAudio = false;
  let activeAudioSource = null;

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
        modelPill.textContent = `${data.model_id} (${data.tts_backend})`;
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

  // 4. Submit & Enter keys
  submitBtn.addEventListener('click', executeQuery);
  queryInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      executeQuery();
    }
  });

  // 5. Interruption Button
  interruptBtn.addEventListener('click', () => {
    interruptPlayback();
  });

  function setVoiceState(state) {
    voiceStateIndicator.className = `voice-state-indicator ${state.toLowerCase()}`;
    voiceStateText.textContent = state;

    if (state === 'SPEAKING') {
      interruptBtn.classList.remove('hidden');
    } else if (state === 'IDLE' || state === 'READY' || state === 'DONE') {
      interruptBtn.classList.add('hidden');
    }
  }

  function getSelectedMode() {
    const selected = document.querySelector('input[name="rag-mode"]:checked');
    return selected ? selected.value : 'voice';
  }

  // 6. Text Query Execution
  async function executeQuery() {
    const query = queryInput.value.trim();
    if (!query) return;

    interruptPlayback();
    setVoiceState('THINKING');
    setLoading(true);
    answerText.textContent = '';
    answerText.className = '';

    const mode = getSelectedMode();
    const denseWeight = parseFloat(denseSlider.value);

    try {
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
      setVoiceState('DONE');
    } catch (err) {
      answerText.textContent = `Error: ${err.message}`;
      answerText.className = 'error-text';
      setVoiceState('READY');
    } finally {
      setLoading(false);
    }
  }

  // 7. Voice Recording
  micBtn.addEventListener('click', async () => {
    if (!isRecording) {
      interruptPlayback();
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
        await handleVoiceBlob(audioBlob);
        stream.getTracks().forEach(t => t.stop());
      };

      mediaRecorder.start();
      isRecording = true;
      micBtn.classList.add('recording');
      micText.textContent = 'Listening... Click to Stop';
      setVoiceState('LISTENING');
    } catch (err) {
      alert('Microphone access error: ' + err.message);
      setVoiceState('READY');
    }
  }

  function stopRecording() {
    if (mediaRecorder && isRecording) {
      mediaRecorder.stop();
      isRecording = false;
      micBtn.classList.remove('recording');
      micText.textContent = 'Hold or Click to Speak';
    }
  }

  async function handleVoiceBlob(blob) {
    const reader = new FileReader();
    reader.readAsDataURL(blob);
    reader.onloadend = async () => {
      const base64Data = reader.result.split(',')[1];
      const mode = getSelectedMode();

      if (mode === 'voice') {
        await streamVoiceQuery(base64Data);
      } else {
        await sendVoiceJson(base64Data);
      }
    };
  }

  // 8. Streaming Voice SSE Execution
  async function streamVoiceQuery(base64Audio) {
    currentSessionId = 'sess_' + Math.random().toString(36).substring(2, 9);
    setVoiceState('THINKING');
    answerText.textContent = '';
    answerText.className = '';
    audioQueue = [];

    try {
      const denseWeight = parseFloat(denseSlider.value);
      const payload = {
        audio_base64: base64Audio,
        audio_format: 'wav',
        dense_weight: denseWeight,
        session_id: currentSessionId,
        stream: true,
        mode: 'voice',
      };

      const response = await fetch('/voice/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error('Streaming connection failed.');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split('\n\n');
        buffer = events.pop();

        for (const evt of events) {
          if (!evt.trim()) continue;
          const lines = evt.split('\n');
          let eventName = '';
          let dataStr = '';

          for (const line of lines) {
            if (line.startsWith('event: ')) eventName = line.replace('event: ', '').trim();
            if (line.startsWith('data: ')) dataStr = line.replace('data: ', '').trim();
          }

          if (dataStr) {
            try {
              const data = JSON.parse(dataStr);
              handleStreamEvent(eventName, data);
            } catch (e) {
              console.warn('SSE parse error:', e);
            }
          }
        }
      }
    } catch (err) {
      answerText.textContent = `Voice Error: ${err.message}`;
      setVoiceState('READY');
    }
  }

  function handleStreamEvent(event, data) {
    if (event === 'status') {
      setVoiceState(data.text);
    } else if (event === 'transcript') {
      queryInput.value = data.text;
    } else if (event === 'token') {
      if (data.delta) {
        answerText.textContent += data.delta;
      }
    } else if (event === 'audio_chunk') {
      setVoiceState('SPEAKING');
      if (data.audio_base64) {
        enqueueAudioChunk(data.audio_base64);
      }
    } else if (event === 'done') {
      if (data.latency) {
        updateLatencyGrid(data.latency);
      }
      if (!isPlayingAudio && audioQueue.length === 0) {
        setVoiceState('DONE');
      }
    } else if (event === 'error') {
      answerText.textContent = `Error: ${data.text}`;
      setVoiceState('READY');
    }
  }

  async function sendVoiceJson(base64Audio) {
    setLoading(true);
    setVoiceState('THINKING');
    try {
      const res = await fetch('/voice', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          audio_base64: base64Audio,
          audio_format: 'wav',
          mode: 'text',
        }),
      });
      const data = await res.json();
      renderResponse(data);
      setVoiceState('DONE');
    } catch (err) {
      answerText.textContent = `Voice error: ${err.message}`;
      setVoiceState('READY');
    } finally {
      setLoading(false);
    }
  }

  // 9. Web Audio Queue Player
  function getAudioContext() {
    if (!audioContext) {
      audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 24000 });
    }
    if (audioContext.state === 'suspended') {
      audioContext.resume();
    }
    return audioContext;
  }

  function enqueueAudioChunk(base64Pcm) {
    audioQueue.push(base64Pcm);
    if (!isPlayingAudio) {
      playNextAudioChunk();
    }
  }

  async function playNextAudioChunk() {
    if (audioQueue.length === 0) {
      isPlayingAudio = false;
      setVoiceState('DONE');
      return;
    }

    isPlayingAudio = true;
    const b64 = audioQueue.shift();
    const ctx = getAudioContext();

    try {
      const rawBytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
      const int16Array = new Int16Array(rawBytes.buffer);
      const float32Array = new Float32Array(int16Array.length);

      for (let i = 0; i < int16Array.length; i++) {
        float32Array[i] = int16Array[i] / 32768.0;
      }

      const audioBuffer = ctx.createBuffer(1, float32Array.length, 24000);
      audioBuffer.getChannelData(0).set(float32Array);

      activeAudioSource = ctx.createBufferSource();
      activeAudioSource.buffer = audioBuffer;
      activeAudioSource.connect(ctx.destination);

      activeAudioSource.onended = () => {
        playNextAudioChunk();
      };

      activeAudioSource.start();
    } catch (e) {
      console.warn('Audio decode/playback note:', e);
      playNextAudioChunk();
    }
  }

  // 10. Interruption / Barge-in
  function interruptPlayback() {
    if (activeAudioSource) {
      try {
        activeAudioSource.stop();
      } catch (e) {}
      activeAudioSource = null;
    }
    audioQueue = [];
    isPlayingAudio = false;
    setVoiceState('INTERRUPTED');

    // Notify server to cancel active TTS / generation
    fetch(`/voice/interrupt?session_id=${currentSessionId}`, { method: 'POST' }).catch(() => {});
    setTimeout(() => {
      setVoiceState('READY');
    }, 1200);
  }

  // 11. Render Utilities
  function renderResponse(data) {
    answerText.textContent = data.answer;
    langBadge.textContent = `Language: ${data.detected_language || 'Unknown'}`;
    if (voiceBadge && data.voice_type) {
      voiceBadge.textContent = `Voice: ${data.voice_type}`;
    }

    if (data.grounding) {
      groundingBadge.textContent = data.grounding.is_grounded ? 'Grounded (Verified)' : 'Ungrounded / Refusal';
      groundingBadge.className = data.grounding.is_grounded ? 'badge-grounding grounded' : 'badge-grounding ungrounded';
    }

    if (data.latency) {
      updateLatencyGrid(data.latency);
    }

    if (data.sources && data.sources.length > 0) {
      sourcesList.innerHTML = data.sources.map((s, idx) => `
        <div class="source-item">
          <div class="source-header">
            <span class="source-tag">#${idx + 1} (${s.language})</span>
            <span class="source-score">Score: ${(s.score * 100).toFixed(1)}%</span>
          </div>
          <div class="source-body">${escapeHtml(s.text)}</div>
        </div>
      `).join('');
    } else {
      sourcesList.innerHTML = '<div class="empty-sources">No source passages retrieved.</div>';
    }
  }

  function updateLatencyGrid(lat) {
    totalLatencyBadge.textContent = `${lat.total_ms.toFixed(1)} ms`;
    totalLatencyBadge.className = lat.total_ms <= 200.0 ? 'latency-pill fast' : 'latency-pill slow';

    const setVal = (id, val) => {
      const el = document.getElementById(id);
      if (el) el.textContent = `${val.toFixed(1)} ms`;
    };

    setVal('lat-stt', lat.stt_ms || 0.0);
    setVal('lat-first-audio', lat.first_audio_latency_ms || 0.0);
    setVal('lat-ttft', lat.llm_ttft_ms || 0.0);
    setVal('lat-vector', lat.vector_retrieval_ms || 0.0);
    setVal('lat-bm25', lat.bm25_retrieval_ms || 0.0);
    setVal('lat-fusion', lat.hybrid_fusion_ms || 0.0);
    setVal('lat-llm', lat.llm_generation_ms || 0.0);
    setVal('lat-grounding', lat.grounding_check_ms || 0.0);
  }

  function setLoading(loading) {
    if (loading) {
      submitBtn.disabled = true;
      submitBtn.classList.add('loading');
    } else {
      submitBtn.disabled = false;
      submitBtn.classList.remove('loading');
    }
  }

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }
});
