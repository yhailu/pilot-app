// ApplyPilot web client helpers.
// Most page logic lives in inline Alpine components — this file only
// exposes generic SSE / fetch helpers and the small `topBar()` data
// model used by base.html.

(function () {
  'use strict';

  /**
   * Subscribe to an SSE endpoint with a single onEvent callback.
   * Returns the EventSource so the caller can .close() it on teardown.
   *
   *   const es = subscribeStream('/api/streams/abc', (evt) => { ... });
   *   es.close();
   */
  window.subscribeStream = function (url, onEvent) {
    const es = new EventSource(url);
    ['line', 'action', 'done', 'error'].forEach((name) => {
      es.addEventListener(name, (e) => onEvent({ event: name, data: e.data }));
    });
    es.onerror = () => onEvent({ event: 'error', data: 'connection lost' });
    return es;
  };

  /**
   * Top-bar component used by base.html. Pulls the current model + today's
   * cost so every page shows the same chips without each template re-implementing
   * the call.
   */
  window.topBar = function () {
    return {
      currentModel: '',
      costToday: 0,
      async init() {
        try {
          const m = await (await fetch('/api/models')).json();
          this.currentModel = m.current || '(default)';
        } catch (_) { /* ignore */ }
        try {
          const c = await (await fetch('/api/cost')).json();
          const today = new Date().toISOString().slice(0, 10);
          const bucket = (c.by_day || []).find((b) => b.date === today);
          this.costToday = bucket ? bucket.cost_usd : 0;
        } catch (_) { /* ignore */ }
      }
    };
  };

  /**
   * URL helper — encode each path segment, but leave the slashes that separate
   * segments alone. The job-detail route uses {url:path} so the URL itself
   * is appended verbatim, which means we DO want to encode the URL once.
   */
  window.jobDetailHref = function (jobUrl) {
    return '/jobs/' + encodeURIComponent(jobUrl);
  };

  // Small auto-init: hover-tooltip for any [data-tip] element. Cheap, no library.
  document.addEventListener('mouseover', (e) => {
    const el = e.target.closest('[data-tip]');
    if (!el) return;
    el.title = el.getAttribute('data-tip');
  });
})();
