// Keep the scene mounted. Telemetry patches text/panels, not the scene or scroll host.
(() => {
  const studio = document.querySelector('.agent-studio');
  if (!studio) return;
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (reducedMotion) document.querySelectorAll('animateMotion').forEach(el => el.remove());
  const fallback = document.body.dataset.defaultAgent || 'manager';
  const valid = new Set([...studio.querySelectorAll('[data-agent]')].map(el => el.dataset.agent));
  let saved = {};
  try { saved = JSON.parse(sessionStorage.getItem('stock-studio-inspector') || '{}'); } catch (_) {}
  let selected = valid.has(saved.role) ? saved.role : fallback;
  let tab = ['current','history','messages','trace'].includes(saved.tab) ? saved.tab : 'current';
  function select(role, nextTab = tab) {
    selected = role;
    tab = nextTab;
    studio.querySelectorAll('[data-agent]').forEach(el => {
      el.classList.toggle('selected', el.dataset.agent === role);
      el.setAttribute('aria-pressed', String(el.dataset.agent === role));
    });
    studio.querySelectorAll('[data-detail]').forEach(panel => {
      panel.hidden = panel.dataset.detail !== role;
      panel.querySelectorAll('[data-tab]').forEach(el => el.setAttribute('aria-pressed', String(el.dataset.tab === tab)));
      panel.querySelectorAll('[data-content]').forEach(el => { el.hidden = el.dataset.content !== tab; });
      const replay = panel.querySelector('.replay-task');
      replay.disabled = !studio.querySelector(`.recorded-transfer[data-sender="${panel.dataset.detail}"],.recorded-transfer[data-receiver="${panel.dataset.detail}"]`);
    });
    try { sessionStorage.setItem('stock-studio-inspector', JSON.stringify({role, tab})); } catch (_) {}
  }
  function update(markup) {
    const next = new DOMParser().parseFromString(markup, 'text/html').querySelector('.agent-studio');
    if (!next) return;
    // Preserve the exact scroll offset of every inspector viewport. No focus,
    // scrollIntoView or document replacement occurs on a telemetry update.
    const sync = (selector, classes = false) => {
      const oldNodes = [...studio.querySelectorAll(selector)];
      const newNodes = [...next.querySelectorAll(selector)];
      oldNodes.forEach((old, index) => {
        const fresh = newNodes[index];
        if (!fresh) return;
        const top = old.scrollTop, left = old.scrollLeft;
        if (classes && old.className !== fresh.className) old.className = fresh.className;
        if (old.innerHTML !== fresh.innerHTML) old.innerHTML = fresh.innerHTML;
        if (old.scrollTop !== top) old.scrollTop = top;
        if (old.scrollLeft !== left) old.scrollLeft = left;
      });
    };
    // Buttons, avatars, desks and monitor elements retain their identity/animation.
    studio.querySelectorAll('[data-agent]').forEach(old => {
      const fresh = next.querySelector(`[data-agent="${old.dataset.agent}"]`);
      if (fresh) old.classList.toggle('working', fresh.classList.contains('working'));
    });
    ['.studio-team-state', '.studio-mode', '.studio-table small', '.studio-floor-note',
     '.studio-handoff-caption', '.station-bubble', '.task-content', '[data-tab]'].forEach(s => sync(s));
    sync('.station-status', true);
    sync('.studio-handoffs');
    if (reducedMotion) studio.querySelectorAll('animateMotion').forEach(el => el.remove());
    select(selected, tab);
  }
  studio.querySelectorAll('[data-agent]').forEach(button => {
    button.onclick = () => select(button.dataset.agent);
  });
  studio.querySelectorAll('[data-tab]').forEach(button => {
    button.onclick = () => select(selected, button.dataset.tab);
  });
  let replayTimer;
  studio.querySelectorAll('.replay-task').forEach(button => {
    button.onclick = () => {
      clearTimeout(replayTimer);
      const edges = [...studio.querySelectorAll('.recorded-transfer')];
      edges.forEach(edge => {
        edge.classList.remove('replaying');
        if (edge.dataset.sender !== selected && edge.dataset.receiver !== selected) return;
        edge.classList.add('replaying');
        edge.querySelectorAll('animateMotion').forEach(animation => animation.beginElement());
      });
      const caption = studio.querySelector('.studio-handoff-caption');
      caption.textContent = 'Xem lại message đã ghi nhận · không phải tác vụ đang chạy';
      replayTimer = setTimeout(() => {
        edges.forEach(edge => edge.classList.remove('replaying'));
        caption.textContent = 'Bấm avatar để xem công việc';
      }, 4200);
    };
  });
  // On first load show work/history rather than an empty current tab after completion.
  if (!saved.tab && !studio.querySelector('.working')) tab = 'history';
  select(selected);
  if (typeof channel !== 'undefined' && typeof window !== 'undefined') {
    window.addEventListener(channel, event => {
      if (typeof event.detail === 'string') update(event.detail);
    });
  }
})();
